""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer,\
    text_global_pool
from .utils import to_2tuple
import pdb
import math

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training
    attentional_pool: bool = False  # whether to use attentional pooler
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    pos_embed_type: str = 'learnable'
    timm_model_name: str = None
    timm_model_pretrained: bool = False
    timm_pool: str = 'avg'
    timm_proj: str = 'linear'
    timm_proj_bias: bool = False
    timm_drop: float = 0.
    timm_drop_path: Optional[float] = None
    
    # 补全所有 _build_vision_tower 需要的参数
    norm_kwargs: dict = None
    act_kwargs: dict = None
    no_ln_pre: bool = False
    final_ln_after_pool: bool = False
    pool_type: str = 'tok'
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None
    
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_class: type = None
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'
    hf_model_pretrained: bool = False
    
    # === 🚀 修复：这里是真正的 pool_type ===
    pool_type: str = 'argmax' 
    # ==================================
    
    proj_dir: str = 'text_to_img' 
    proj_bias: bool = False
    
    mlp_ratio: float = 4.0
    embed_cls: bool = False
    no_causal_mask: bool = False
    pad_id: int = 0
    norm_kwargs: dict = None
    act_kwargs: dict = None
    output_tokens: bool = False
# =====================================================================================

class PatchGCN(nn.Module):
    """
    轻量级动态图卷积模块：让图像的 196 个 Patch 根据语义相似度互相传递信息（拓扑推理）
    """
    def __init__(self, dim=768):
        super().__init__()
        self.msg_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # ⚠️ 【消融实验修改】：关闭零初始化，改成 1.0 (让模型一开始就强制承受 GCN 的信息冲击)
        # self.gamma_gcn = nn.Parameter(torch.zeros(1))  <-- 原来的保留注释掉
        self.gamma_gcn = nn.Parameter(torch.ones(1))     # <-- 消融实验用的纯净版

    def forward(self, x):
        attn = torch.bmm(x, x.transpose(1, 2)) / math.sqrt(x.size(-1))
        A = F.softmax(attn, dim=-1) 
        
        msg = torch.bmm(A, x) 
        msg = self.msg_proj(msg)
        
        # 如果 gamma_gcn 是 1，这里就是普通的 x + msg，没有任何前期保护了
        return x + self.gamma_gcn * msg


# ================= 🚀 纯净版：带有拓扑推理 (GCN) 的非对称适配器 =================
class TGSA(nn.Module):
    """
    非侵入式参数高效微调模块：利用文本语义动态高亮遥感图像中相关的空间实体。
    (已恢复为最纯净的单头点乘版本，作为 FGTA 的干净对照底座)
    """
    def __init__(self, text_dim=512, img_dim=768):
        super().__init__()
        # 1. 引入图卷积网络 (保留最强拓扑基本盘)
        self.gcn = PatchGCN(dim=img_dim)
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, img_dim),
            nn.GELU(),
            nn.Linear(img_dim, img_dim)
        )
        self.ln = nn.LayerNorm(img_dim)
        self.gamma = nn.Parameter(torch.ones(1))
        
    def forward(self, img_feat, text_global):
        B, N, D = img_feat.shape
        cls_token = img_feat[:, 0:1, :]
        spatial_tokens = img_feat[:, 1:, :] 
        
        # 1. 拓扑推理阶段 (GCN)
        spatial_tokens = self.gcn(spatial_tokens)
        
        # 2. 实体激发阶段 (单头纯净点乘)
        text_query = self.ln(self.text_proj(text_global)).unsqueeze(2)
        spatial_weights = torch.bmm(spatial_tokens, text_query) 
        spatial_mask = torch.sigmoid(spatial_weights)
        
        spatial_refined = spatial_tokens + self.gamma * (spatial_tokens * spatial_mask)
        return torch.cat([cls_token, spatial_refined], dim=1)
# ====================================================================================

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)
        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = 0.07,
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            # 固定为0，不参与训练：CLIP风格softmax loss不需要该偏置
            self.logit_bias = nn.Parameter(torch.zeros([]), requires_grad=False)
            
        # ================= 🚀 注册 TGSA 模块 =================
        self.tgsa = TGSA(text_dim=embed_dim, img_dim=768)
        # =====================================================

        self.itm_head = nn.Sequential(
            nn.Linear(embed_dim, 2)
        )
        self.momentum = nn.Parameter(torch.ones([]) * 0.995, requires_grad=False)
        self.visual_m = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text_m =  _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype).transformer
        self.text_proj_m = text.text_projection
        self.model_pairs = [[self.visual,self.visual_m],
                    [self.transformer,self.text_m],
                    [self.text_projection,self.text_proj_m],
                    ]   
        self._no_save_params = ['visual_m', 'text_m', 'text_proj_m']
        self._init_weight()
        self.copy_params()
    
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            if isinstance(model_pair[0], nn.Module) and isinstance(model_pair[1], nn.Module):
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data.copy_(param.data)  
                    param_m.requires_grad = False
            elif isinstance(model_pair[0], nn.Parameter) and isinstance(model_pair[1], nn.Parameter):
                model_pair[1].data.copy_(model_pair[0].data)
                model_pair[1].requires_grad = False
            else:
                raise TypeError(f"Unsupported type: {type(model_pair[0])} and {type(model_pair[1])}")
        
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:  
            main_component, momentum_component = model_pair
            
            if isinstance(main_component, nn.Module) and isinstance(momentum_component, nn.Module):
                for param, param_m in zip(main_component.parameters(), momentum_component.parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            
            elif isinstance(main_component, nn.Parameter) and isinstance(momentum_component, nn.Parameter):
                momentum_component.data = momentum_component.data * self.momentum + main_component.data * (1. - self.momentum)
            
            else:
                raise TypeError(
                    f"组件类型不匹配或不支持："
                    f"主组件类型 {type(main_component)}，"
                    f"动量组件类型 {type(momentum_component)}"
                )

    def _init_weight(self):
        for m in self.itm_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    
    def get_text_to_image_mapping(self, text_features, image_features, k):
        similarity = text_features @ image_features.T
        sorted_indices = torch.argsort(similarity, dim=-1, descending=True)
        topk_indices = sorted_indices[:,:k]
        return topk_indices
    
    def get_image_to_text_mapping(self, image_features, text_features, k):
        similarity = image_features @ text_features.T
        sorted_indices = torch.argsort(similarity, dim=-1, descending=True)
        topk_indices = sorted_indices[:,:k]
        return topk_indices

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False,embeds: bool = False):
        features,embeddings = self.visual(image)
        if embeds:
            return embeddings
        elif normalize:
            return F.normalize(features, dim=-1)
        else:
            return F.normalize(features, dim=-1)

    def encode_text(self, text, normalize: bool = False,embeds: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  
        text_embeds = x  # [batch_size, n_ctx, transformer.width]
        
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        if embeds:
            return text_embeds
        elif normalize:
            return  F.normalize(x, dim=-1)
        else:
            return F.normalize(x, dim=-1)
    
    def encode_mimage(self, image, normalize: bool = False,embeds: bool = False):
        features,embeddings = self.visual_m(image)
        if embeds:
            return  embeddings
        elif normalize:
            return F.normalize(features, dim=-1)
        else:
            return F.normalize(features, dim=-1)

    def encode_mtext(self, text, normalize: bool = False,embeds: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.text_m(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  
        text_embeds = x  # [batch_size, n_ctx, transformer.width]
        
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_proj_m is not None:
            if isinstance(self.text_proj_m, nn.Linear):
                x = self.text_proj_m(x)
            else:
                x = x @ self.text_proj_m
    
        if embeds:
            return text_embeds
        elif normalize:
            return  F.normalize(x, dim=-1)
        else:
            return F.normalize(x, dim=-1)
    
    def encode_weight_image(self, text, image_embs = None):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        itm_input = self.transformer(x, attn_mask=self.attn_mask,image_embs=image_embs,mode = "multi")
        return  itm_input

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits
    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            WeightsoftCEloss = None
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        img_locals = self.encode_image(image, embeds=True)
        device = image_features.device
        
        # 恢复 57.67% 的隐式对抗正则化：全局高亮
        img_locals = self.tgsa(img_locals, text_features)

        with torch.no_grad():
            self._momentum_update()
            image_feat_m = self.encode_mimage(image)
            text_feat_m = self.encode_mtext(text)
            image_feat_all = image_feat_m.t()
            text_feat_all = text_feat_m.t()
            logit_scale_clamped = torch.clamp(self.logit_scale, max=4.6052)
            sim_i2t_m = logit_scale_clamped.exp() * image_feat_m @ text_feat_all
            sim_t2i_m = logit_scale_clamped.exp() * text_feat_m @ image_feat_all
            
            # ================= 🚀 38.05% SOTA 核心：带严格阈值截断的自相似度软标签 =================
            sim_targets = torch.zeros(sim_i2t_m.size(), device=device)
            sim_targets.fill_diagonal_(1.0)
            
            # 1. 计算图像-图像自相似度 (必须加高阈值截断)
            sim_i2i = image_feat_m @ image_feat_m.t()  
            sim_i2i = torch.where(sim_i2i > 0.85, sim_i2i, torch.zeros_like(sim_i2i))    
            sim_i2i.fill_diagonal_(0.0)                
            
            # 2. 计算文本-文本自相似度
            sim_t2t = text_feat_m @ text_feat_m.t()    
            sim_t2t = torch.where(sim_t2t > 0.85, sim_t2t, torch.zeros_like(sim_t2t))
            sim_t2t.fill_diagonal_(0.0)
            
            # 3. 动态软标签生成：缔造 38.05% 奇迹的原版公式 (系数 0.5 + L1 归一化)
            alpha_label = 0.8 
            
            soft_targets_i2t = sim_targets + sim_t2t * 0.5 
            soft_targets_i2t = F.normalize(soft_targets_i2t, p=1, dim=1) 
            sim_i2t_targets = (1 - alpha_label) * F.softmax(sim_i2t_m, dim=1) + alpha_label * soft_targets_i2t
            
            soft_targets_t2i = sim_targets + sim_i2i * 0.5
            soft_targets_t2i = F.normalize(soft_targets_t2i, p=1, dim=1)
            sim_t2i_targets = (1 - alpha_label) * F.softmax(sim_t2i_m, dim=1) + alpha_label * soft_targets_t2i
            # =========================================================================================

            # ================= 🚀 终极杀器：负样本专属信息熵 (Negative-only Entropy) =================
            import math
            bs_size = sim_i2t_m.size(0)
            
            # 1. 图搜文方向的负熵
            prob_i2t = F.softmax(sim_i2t_m, dim=1)
            mask = torch.ones_like(prob_i2t) - torch.eye(bs_size, device=device) # 挖掉正样本
            prob_i2t_neg = prob_i2t * mask
            prob_i2t_neg = prob_i2t_neg / (prob_i2t_neg.sum(dim=1, keepdim=True) + 1e-8) # 重新归一化负样本
            entropy_i = -torch.sum(prob_i2t_neg * torch.log(prob_i2t_neg + 1e-8), dim=1)
            entropy_i_norm = entropy_i / math.log(bs_size - 1 + 1e-8) # 归一化
            
            # 2. 文搜图方向的负熵
            prob_t2i = F.softmax(sim_t2i_m, dim=1)
            prob_t2i_neg = prob_t2i * mask
            prob_t2i_neg = prob_t2i_neg / (prob_t2i_neg.sum(dim=1, keepdim=True) + 1e-8)
            entropy_t = -torch.sum(prob_t2i_neg * torch.log(prob_t2i_neg + 1e-8), dim=1)
            entropy_t_norm = entropy_t / math.log(bs_size - 1 + 1e-8)
            # =========================================================================================

        # ------------------------- 损失计算与组合 -------------------------
        pos_itm_input = self.encode_weight_image(text, img_locals)
        logits_per_image = self.logit_scale.exp() * image_features @ text_feat_all + self.logit_bias
        logits_per_text = self.logit_scale.exp() * text_features @ image_feat_all + self.logit_bias
        
        # 结合 NoE 信息熵动态权重计算 Loss
        if WeightsoftCEloss is not None:
            itc_loss = (WeightsoftCEloss(logits_per_image, sim_i2t_targets, ambiguity=entropy_i_norm) + 
                        WeightsoftCEloss(logits_per_text, sim_t2i_targets, text_to_image=True, ambiguity=entropy_t_norm)) / 2.0
        else:
            itc_loss = (F.cross_entropy(logits_per_image, sim_i2t_targets) + F.cross_entropy(logits_per_text, sim_t2i_targets)) / 2.0
            
        total_loss = itc_loss
        
        bs = image.size(0)
        with torch.no_grad():
            # ... 后续代码完全不变 ...
            # sim_i2t_m: [bs, bs] 图像->文本; sim_t2i_m: [bs, bs] 文本->图像
            # weights_i2t 用于从文本中采负样本（给每张图找困难负文本），应用 sim_i2t_m
            sim_i2t_slice = sim_i2t_m[:, :bs].clone()
            sim_i2t_slice = torch.nan_to_num(sim_i2t_slice, nan=0.0, posinf=0.0, neginf=0.0)
            weights_i2t = F.softmax(sim_i2t_slice, dim=1) + 1e-8
            weights_i2t.fill_diagonal_(0)
            # weights_t2i 用于从图像中采负样本（给每条文本找困难负图像），应用 sim_t2i_m
            sim_t2i_slice = sim_t2i_m[:, :bs].clone()
            sim_t2i_slice = torch.nan_to_num(sim_t2i_slice, nan=0.0, posinf=0.0, neginf=0.0)
            weights_t2i = F.softmax(sim_t2i_slice, dim=1) + 1e-8
            weights_t2i.fill_diagonal_(0)
        images_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            images_neg.append(img_locals[neg_idx])
        images_neg = torch.stack(images_neg,dim=0)   
        texts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            texts_neg.append(text[neg_idx])
        texts_neg = torch.stack(texts_neg,dim=0)
        neg_img_itm_input = self.encode_weight_image(texts_neg, img_locals)
        neg_txt_itm_input = self.encode_weight_image(text, images_neg)
        pos_labels = torch.ones(bs, device=device)
        neg_batch_labels = torch.zeros(2*bs, device=device)
        itm_labels = torch.cat([pos_labels, neg_batch_labels], dim=0).long()
        itm_logit = self.itm_head(torch.cat([pos_itm_input, neg_img_itm_input, neg_txt_itm_input], dim=0))
        itm_loss = nn.CrossEntropyLoss()(itm_logit,itm_labels)
        total_loss += itm_loss
        out_dict = {
            "total_loss": total_loss
        }
        return out_dict


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32
    
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys
    # print(f"missing keys: {missing_keys}")
    # print(f"unexpected keys: {unexpected_keys}")
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logging.info('Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict['positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg['mean']  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg
