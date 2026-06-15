"""
类别制衡损失函数
================
- SegmentationLoss: CE + Dice (对抗前景/背景严重不平衡)
- AsymmetricLoss:  ASL (对抗多标签极度不平衡, 提升正样本 Recall)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """CE + Dice 联合分割损失

    在严重类别不平衡下, 纯 CE 会纵容模型全部预测背景。
    Dice Loss 直接优化类别间重叠度, 对前景小类更敏感。

    Args:
        num_classes:  类别数 (含背景, VOC = 21)
        ignore_index: 忽略的标签值 (VOC 边界 = 255)
        dice_weight:  Dice Loss 的权重系数
        smooth:       Dice 平滑项, 防止除零
    """

    def __init__(self, num_classes: int = 21, ignore_index: int = 255,
                 dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H, W) logits
            target: (B, H, W)    long tensor
        """
        ce_loss = self.ce(pred, target)

        # ---- Dice Loss (只算前景类别 1..C-1) ----
        softmax_pred = F.softmax(pred, dim=1)          # (B, C, H, W)
        target_one_hot = F.one_hot(target.clamp(0, self.num_classes - 1),
                                   self.num_classes)    # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # 构造有效区域 mask: 排除 ignore_index 像素
        valid_mask = (target != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
        softmax_pred = softmax_pred * valid_mask
        target_one_hot = target_one_hot * valid_mask

        # 逐类 Dice (跳过 index=0 背景)
        dice_loss = torch.tensor(0.0, device=pred.device)
        foreground_count = 0
        for c in range(1, self.num_classes):
            pred_c = softmax_pred[:, c]          # (B, H, W)
            tgt_c = target_one_hot[:, c]         # (B, H, W)
            intersection = (pred_c * tgt_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + tgt_c.sum(dim=(1, 2))
            dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = dice_loss + (1.0 - dice_c.mean())
            foreground_count += 1

        if foreground_count > 0:
            dice_loss = dice_loss / foreground_count

        return ce_loss + self.dice_weight * dice_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification

    论文: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., ICLR 2021)

    核心思想:
      - 对正样本 (稀有): 用正常甚至更低的 gamma_neg, 保留梯度
      - 对负样本 (大量): 用更高的 gamma_pos (实为 gamma_neg), 抑制易分负样本的梯度
      - 可选: probability shifting, 进一步削减弱负样本

    Args:
        gamma_neg: 负样本聚焦参数 (>=0, 越大越抑制易分负样本)
        gamma_pos: 正样本聚焦参数 (>=0, 通常设 0)
        clip:      probability shifting 阈值 (0=禁用, 常用 0.05)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) 原始 logits
            targets: (B, C) 多标签 0/1
        """
        # Sigmoid probabilities
        probs = torch.sigmoid(logits)

        # Probability shifting: 削减负样本概率，彻底丢弃简单负样本
        if self.clip and self.clip > 0:
            probs_neg = (probs - self.clip).clamp(min=0.0)
        else:
            probs_neg = probs

        # 计算不对称 focal 因子
        # 正样本损失: -targets * (1 - probs)^gamma_pos * log(probs)
        # 负样本损失: -(1-targets) * probs_neg^gamma_neg * log(1-probs)
        loss_pos = targets * F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        loss_neg = (1 - targets) * F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Focal modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)  # 正确类别概率
        focal_pos = (1 - p_t) ** self.gamma_pos
        focal_neg = probs_neg ** self.gamma_neg

        loss = focal_pos * loss_pos + focal_neg * loss_neg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
