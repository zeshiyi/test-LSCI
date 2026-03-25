import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections


class Weight_soft_CEloss(nn.Module):
    def __init__(self, reduction='mean', imagegamma=1.0, textgamma=1.0, alpha=0.3, maxgamma=2.0, mingamma=-1.0):
        super(Weight_soft_CEloss, self).__init__()
        self.reduction = reduction
        self.imagegamma = imagegamma
        self.textgamma = textgamma
        self.maxgamma = torch.tensor(maxgamma)
        self.mingamma = torch.tensor(mingamma)
        self.image_to_textgamma = torch.tensor(imagegamma)
        self.text_to_imagegamma = torch.tensor(textgamma)
        self.alpha = alpha
        self.update = lambda g, gap: self.update_gamma(g, gap, self.maxgamma, self.mingamma)

    def update_gamma(self, gamma, gap, maxgamma, mingamma):
        # 按照论文 Algorithm 1：γt+1 = γt * exp(CE_val) 当 γt >= 0
        # 但 CE_val 可能很小（0.01-0.05 量级），加 clamp 防止指数项爆炸
        gap_clamped = torch.clamp(gap, min=-0.5, max=0.5)
        if gamma >= 0:
            new_gamma = gamma * torch.exp(gap_clamped)
            new_gamma = torch.clamp(new_gamma, max=maxgamma)
        else:
            new_gamma = gamma * torch.exp(-gap_clamped)
            new_gamma = torch.clamp(new_gamma, min=mingamma)
        return new_gamma

    def _apply_switch(self, gamma, gamma_init):
        # Sth = 0.1 * gamma_init (论文中的切换阈值)
        sth = abs(gamma_init) * 0.1
        if abs(gamma) < sth:
            if gamma >= 0:
                # 从 focal 切到 inv-focal
                return torch.tensor(-sth)
            else:
                # 从 inv-focal 切回 focal
                return torch.tensor(sth)
        return gamma

    def updategamma(self, image_text_meancalibration_gap, text_image_meancalibration_gap):
        image_text_meancalibration_gap = torch.tensor(image_text_meancalibration_gap)
        text_image_meancalibration_gap = torch.tensor(text_image_meancalibration_gap)
        self.image_to_textgamma = self.update_gamma(
            self.image_to_textgamma, image_text_meancalibration_gap, self.maxgamma, self.mingamma)
        self.text_to_imagegamma = self.update_gamma(
            self.text_to_imagegamma, text_image_meancalibration_gap, self.maxgamma, self.mingamma)
        # 切换阈值检查（对两个方向独立处理）
        self.image_to_textgamma = self._apply_switch(self.image_to_textgamma, self.imagegamma)
        self.text_to_imagegamma = self._apply_switch(self.text_to_imagegamma, self.textgamma)
        print(f"update image_to_textgamma to {self.image_to_textgamma}")
        print(f"update text_to_imagegamma to {self.text_to_imagegamma}")
        return 0

    def forward(self, inputs, labels, text_to_image=False, ambiguity=None):
        inputs = torch.clamp(inputs, min=-100, max=100)
        pt = F.softmax(inputs, dim=1)
        eps = 1e-8
        pt_safe = torch.clamp(pt, eps, 1.0 - eps)
        log_pt = F.log_softmax(inputs, dim=1)

        if text_to_image:
            gamma = self.text_to_imagegamma
        else:
            gamma = self.image_to_textgamma

        gamma_val = float(gamma)

        # ================= 🚀 创新二：基于模糊度的实例级动态校准 =================
        if ambiguity is not None:
            # 熵值(ambiguity)越大，说明这段文本匹配到了很多不同的图片（多义性/模糊）。
            # 为了防止模型强行拟合模糊样本导致空间崩溃，我们用 (1.0 + ambiguity) 放大 gamma。
            # 在 Focal Loss 中，gamma 越大，算出的最终惩罚权重越小，模型对它越宽容。
            gamma_instance = gamma_val * (1.0 + ambiguity.view(-1, 1))
        else:
            gamma_instance = gamma_val
        # =========================================================================

        if gamma_val >= 0:
            # focal loss: -(|ỹ - P|^γ_i) * ỹ * log(P)
            diff = torch.abs(labels - pt_safe)
            weight = torch.clamp(diff, min=eps) ** gamma_instance * labels
        else:
            # inv-focal loss: -(|ỹ + P|^|γ_i|) * ỹ * log(P)
            if ambiguity is not None:
                gamma_mag = torch.abs(gamma_instance)
            else:
                gamma_mag = abs(gamma_val)
            diff = torch.clamp(labels + pt_safe, min=eps)
            weight = diff ** gamma_mag * labels

        loss = -torch.sum(weight * log_pt, dim=1)
        return loss.mean()