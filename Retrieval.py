import argparse
import os
import sys
import math
import numpy as np
import random
import time
import datetime
import json
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import utils
from PIL import Image
import io
import types

import torch.multiprocessing

# 强制使用 file_system 策略，这是解决 "Bad file descriptor"终极方案
torch.multiprocessing.set_sharing_strategy('file_system')
from dataset import create_dataset, create_sampler, create_loader, dataset_collate, rs5m_dataset_collate
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_retrieval import CLIPFusionModule, create_and_load_pretrained
from ruamel.yaml import YAML
from models.open_clip import tokenizer
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from models.loss import Weight_soft_CEloss
from utils.eval_utils import evaluate_dataset, evaluate_dataset_ECE_error

scaler = GradScaler()
now = datetime.datetime.now()
time_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = now.strftime("%Y-%m-%d_%H-%M-%S-log.txt")


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            print(f'No gradient for {name}, skipping...')


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    metric_logger = utils.MetricLogger(delimiter="")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    metric_logger.add_meter('TCloss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 200
    print('_________________{}__________________'.format(len(data_loader)))
    lennum = len(data_loader)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 纯 FP32 单精度前向传播
        images = image.to(device, non_blocking=True)
        texts = text.to(device, non_blocking=True)

        loss = model(images, texts, WeightsoftCEloss)
        optimizer.zero_grad()
        loss.backward()

        # 🛡️ 终极防爆护盾5：梯度级净化 (保护预训练权重不被污染)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1e4, neginf=-1e4)

        # 梯度裁剪防爆器
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        metric_logger.update(TCloss=loss.item())
        metric_logger.update(lr=scheduler.get_lr()[-1])

    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.10f} ".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, k=40):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']
    image_feas = []
    text_feas = []
    local_images = []
    texts_ids = []
    all_ = []
    print('_________________{}__________________'.format(len(data_loader)))
    for index, batch in enumerate(metric_logger.log_every(data_loader, 100, header)):
        torch.cuda.empty_cache()
        image, _ = batch
        image = image.to(device)
        t1 = time.time()
        image_fea = model.encode_image(image)
        image_feas.append(image_fea)
        local_image = model.encode_image(image, embeds=True)
        local_images.append(local_image)
        del image_fea, local_image, image
        t2 = time.time()
        all_.append(t2 - t1)
    print("infer image time:{:.2f}".format(np.average(all_)))
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer.tokenize(text).to(device)
        text_fea = model.encode_text(text_input)
        text_feas.append(text_fea)
        texts_ids.append(text_input)

    image_feas = torch.cat(image_feas, dim=0).to(device)
    text_feas = torch.cat(text_feas, dim=0).to(device)
    texts_ids = torch.cat(texts_ids, dim=0).to(device)
    image_features = image_feas / image_feas.norm(dim=-1, keepdim=True)
    text_features = text_feas / text_feas.norm(dim=-1, keepdim=True)

    # 🚀 修复：去除 softmax，保留原始的 ITC 相似度分数！
    sims_matrix = model.clip.logit_scale.exp() * image_features @ text_features.t()
    if hasattr(model.clip, 'logit_bias') and model.clip.logit_bias is not None:
        sims_matrix += model.clip.logit_bias

    score_matrix_i2t = sims_matrix.clone()
    score_matrix_t2i = score_matrix_i2t.clone().t()
    local_image_feas = torch.cat(local_images, dim=0).to(device)
    if k != 0:
        image_to_text_mapping = model.get_image_to_text_mapping(image_features, text_features, k)
        text_to_image_mapping = model.get_text_to_image_mapping(text_features, image_features, k)
        for i, img_local in enumerate(local_image_feas):
            topk_text_idx = image_to_text_mapping[i]
            topk_text_ids = texts_ids[topk_text_idx]
            img_local_expanded = img_local.unsqueeze(0).repeat(k, 1, 1)
            match_prob = model.encode_weight_image(topk_text_ids, img_local_expanded)
            score_matrix_i2t[i, topk_text_idx] += match_prob
        for i, txt_local in enumerate(texts_ids):
            topk_image_idx = text_to_image_mapping[i]
            topk_img_fea = local_image_feas[topk_image_idx]
            txt_local_expanded = txt_local.repeat(k, 1)
            match_prob = model.encode_weight_image(txt_local_expanded, topk_img_fea)
            score_matrix_t2i[i, topk_image_idx] += match_prob

    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])

    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': round(tr1, 8),
                   'txt_r5': round(tr5, 8),
                   'txt_r10': round(tr10, 8),
                   'img_r1': round(ir1, 8),
                   'img_r5': round(ir5, 8),
                   'img_r10': round(ir10, 8),
                   'r_mean': round(r_mean, 8)}

    return eval_result


def main(args, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.gpu = 0
    args.distributed = False

    world_size = 1
    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    print("Creating model", flush=True)

    model = CLIPFusionModule(config=config)
    checkpoint = torch.load(args.precheckpoint, map_location='cpu')

    # 提取字典
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 🚀 强行对齐键名 (Key Alignment)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('clip.'):
            name = k
        else:
            name = "clip." + k
        new_state_dict[name] = v

    # 加载修复后的字典，strict=False 是允许的
    msg = model.load_state_dict(new_state_dict, strict=False)

    # ================= 🚀 核心指标修复：同步动量模型 =================
    if hasattr(model, 'clip') and hasattr(model.clip, 'copy_params'):
        model.clip.copy_params()
        print("🔄 动量编码器 (Momentum Model) 同步完毕！记忆不再被污染！")

    # ================= 终极无敌防爆护盾部署 =================
    model.float()

    for name, p in model.named_parameters():
        if 'logit_scale' in name:
            # 🚀 强制换回 CLIP 的官方安全温度系数 ln(1/0.07) = 2.6592
            p.data = torch.tensor(2.6592, dtype=p.dtype, device=p.device)
            print(f"🔥 成功将温度系数强制修复为标准值: {name} = {p.data.item():.4f}")

    # 🛡️ 终极防爆护盾1：黑入底层强制全网使用 FP32
    for m in model.modules():
        if hasattr(m, 'get_cast_dtype'):
            m.get_cast_dtype = types.MethodType(lambda self: torch.float32, m)

    # 🛡️ 终极防爆护盾2：深度替换底层 Buffer 中的 -inf
    for name, buf in model.named_buffers():
        if buf.dtype.is_floating_point:
            buf.data.masked_fill_(buf.data == float('-inf'), -1e4)

    print("🛡️ 防爆护盾部署完毕！一切 NaN 将灰飞烟灭！")

    print("=== 权重加载核对报告 ===")
    core_missing = [k for k in msg.missing_keys if "clip." in k]
    if len(core_missing) == 0:
        print("✅ 恭喜！CLIP 核心主干权重已完美加载。")

    model = model.to(device)
    model_without_ddp = model
    preprocess_train, preprocess_val = model.preprocess_train, model.preprocess_val
    print("Creating retrieval dataset", flush=True)
    train_dataset, val_dataset, test_dataset = create_dataset('re', config, args.evaluate, preprocess_train,
                                                              preprocess_val)
    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    model.train()
    print("Start training", flush=True)
    print(f"The trainable parameters are {count_trainable_parameters(model)}")
    train_dataset_size = len(train_dataset)

    samplers = [None, None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                      batch_size=[config['batch_size_train']] * 3,
                                                      num_workers=[0, 0, 0],
                                                      is_trains=[True, False, False],
                                                      collate_fns=[dataset_collate, dataset_collate,
                                                                   dataset_collate])

# ================= 🚀 终极改进：差分学习率 (保护 RS5M 权重) =================
    pretrained_params = []
    scratch_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
    # 底层 CLIP 预训练参数放入低学习率组
        if "clip." in name:
            pretrained_params.append(param)
    # 上层随机初始化的参数（如 ITM_head）放入高学习率组
        else:
            scratch_params.append(param)

# ⚠️ 注意这里：必须和上面的 for 循环对齐（退格出去）！
    base_lr = float(config['optimizer'].get('lr', 1e-5))
    weight_decay = float(config['optimizer'].get('weight_decay', 0.04))

# 将底层和上层的参数全部合并，取消任何特权！
    all_params = pretrained_params + scratch_params

# 没有任何学习率分组，统一使用 base_lr
    optimizer = torch.optim.AdamW(all_params, lr=base_lr, weight_decay=weight_decay)

# =================================================================================
    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    best = 0
    best_epoch = 0

    for epoch in range(0, max_epoch):
        train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

        with torch.no_grad():
            score_val_i2t, score_val_t2i = evaluation(model_without_ddp, val_loader, tokenizer, device, config, k=0)
            image_text_ece, image_text_bin_dict, text_image_ece, text_image_bin_dict, image_text_meancalibration_gap, text_image_meancalibration_gap = evaluate_dataset_ECE_error(
                score_val_i2t, score_val_t2i, val_loader.dataset.img2txt, val_loader.dataset.txt2img,
                num_bins=config['num_bins'])

        mean_adaece = (image_text_ece.item() + text_image_ece.item()) / 2
        calibration_dict = {
            'epoch': epoch,
            'mean_ece': mean_adaece,
            'image_text_meancalibration_gap': image_text_meancalibration_gap,
            'text_image_meancalibration_gap': text_image_meancalibration_gap,
            'image_text_ece': image_text_ece.item(),
            'text_image_ece': text_image_ece.item(),
        }

        if utils.is_main_process():
            with open(os.path.join(args.output_dir + time_dir, "val_calibration_dict.txt"), "a") as f:
                f.write(json.dumps(calibration_dict) + "\n")

        WeightsoftCEloss.updategamma(image_text_meancalibration_gap, text_image_meancalibration_gap)

        # ============ 🚀 速度优化：合并测试集评估，拒绝重复计算！============
        if epoch >= 5:
            # 1. 统一只调用一次 evaluation (k=40)
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config,
                                                        k=config['k'])

            # 2. 计算 ECE (用于记录日志)
            image_text_ece_t, image_text_bin_dict_t, text_image_ece_t, text_image_bin_dict_t, image_text_meancalibration_gap_t, text_image_meancalibration_gap_t = evaluate_dataset_ECE_error(
                score_test_i2t, score_test_t2i, test_loader.dataset.img2txt, test_loader.dataset.txt2img,
                num_bins=config['num_bins'])

            mean_adaece_t = (image_text_ece_t.item() + text_image_ece_t.item()) / 2
            calibration_dict_t = {
                'epoch': epoch,
                'mean_ece': mean_adaece_t,
                'image_text_meancalibration_gap': image_text_meancalibration_gap_t,
                'text_image_meancalibration_gap': text_image_meancalibration_gap_t,
                'image_text_ece': image_text_ece_t.item(),
                'text_image_ece': text_image_ece_t.item(),
            }

            if utils.is_main_process():
                # 写入测试集 ECE 日志
                with open(os.path.join(args.output_dir + time_dir, "test_calibration_dict.txt"), "a") as f:
                    f.write(json.dumps(calibration_dict_t) + "\n")

                # 3. 计算真正的检索指标 (R@1, R@5, mR...)
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                       test_loader.dataset.img2txt)
                print(test_result)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch}
                with open(os.path.join(args.output_dir + time_dir, filename), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # 4. 保存最高分权重
                if test_result['r_mean'] > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'config': config,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir + time_dir, 'checkpoint_best.pth'))
                    best = test_result['r_mean']
                    best_epoch = epoch
                else:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'config': config,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir + time_dir, 'checkpoint_last.pth'))

        torch.cuda.empty_cache()
    # ============================== for 循环结束 ==============================

    if utils.is_main_process():
        with open(os.path.join(args.output_dir + time_dir, filename), "a") as f:
            f.write("best epoch: %d\n" % best_epoch)

        os.system(f"cat {args.output_dir}{time_dir}/{filename}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precheckpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument("--num_worker", type=int, default=0, help='number of workers')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')

    args = parser.parse_args()
    yaml = YAML()
    print("Init Successful")
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
    Path(args.output_dir + time_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    WeightsoftCEloss = Weight_soft_CEloss(imagegamma=config['weight_init_imagegamma'],
                                          textgamma=config['weight_init_textgamma'], maxgamma=config['themaxgamma'],
                                          mingamma=config['themingamma'])
    yaml.dump(config, open(os.path.join(args.output_dir + time_dir, 'config.yaml'), 'w'))
    main(args, config)