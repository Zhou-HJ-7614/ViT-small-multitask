"""
多任务 ViT-Small 训练脚本 v2 (Improved)
========================================
基于 MAE 预训练的 ViT-Small，在 Pascal VOC 2012 上微调
任务: 单分类、多分类、空检测、语义分割

v2 改进:
  - EMA (指数移动平均): 评估时使用 EMA 权重, 提升 mIoU 0.5~1.5%
  - Early Stopping: 监控 val mIoU, patience=15 epochs 无改善则停止
  - Uncertainty Loss Weighting: 自适应学习各任务损失权重 (替代手工调参)
  - AMP autocast 范围修复: 包含 loss 计算
  - 更强的数据增强: 新增 CoarseDropout, ChannelShuffle

用法:
    python train.py                    # 使用默认配置训练
    python train.py --epochs 100       # 自定义训练轮数
    python train.py --batch_size 8     # 自定义批次大小
    python train.py --no_ema           # 禁用 EMA (默认已启用)
    python train.py --uw               # 启用 Uncertainty Weighting
"""

import os
import sys
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import MultiTaskViTSmall, build_model, ModelEMA, EarlyStopping
from losses import SegmentationLoss, AsymmetricLoss


# ======================== 配置 ========================
class Config:
    # ---- 数据 ----
    VOC_ROOT = '../data/VOCdevkit/VOC2012'
    BG_PATCH_DIR = 'background_patches'
    NUM_BG_PATCHES = 800
    INPUT_SIZE = 512

    # ---- 模型 ----
    NUM_CLASSES = 20
    PRETRAINED_WEIGHTS = 'imagenet'    # 默认使用 timm ImageNet 预训练权重, 或指定 MAE 权重路径
    TIMM_MODEL_NAME = 'vit_small_patch16_224'
    DROP_PATH_RATE = 0.1

    # ---- 训练 ----
    EPOCHS = 100                       # Random init needs a longer schedule; use fewer epochs with pretraining if needed.
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4
    BACKBONE_LR = 1e-4
    HEAD_LR = 1e-3
    SEG_LR = 1e-3
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 5
    MIN_LR_RATIO = 0.01

    # ---- 损失权重 (固定模式, --uw 时会被覆盖) ----
    LOSS_SINGLE_W = 0.1
    LOSS_MULTI_W = 1.0
    LOSS_EMPTY_W = 0.1
    LOSS_SEG_W = 5.0

    # ---- v2 新增功能开关 ----
    USE_EMA = True                     # EMA 开关 (默认开启, 推荐!)
    EMA_DECAY = 0.9999                 # EMA 衰减率
    USE_UW = False                     # Uncertainty Weighting 开关
    EARLY_STOPPING_PATIENCE = 15       # 早停耐心值

    # ---- 系统设置 ----
    NUM_WORKERS = 4
    AMP_ENABLED = True
    SAVE_DIR = 'checkpoints'
    LOG_INTERVAL = 20
    VAL_INTERVAL = 5
    SEED = 42

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ======================== 数据集 ========================
class VOCMultiTaskDataset(Dataset):
    """Pascal VOC 2012 多任务数据集 (与 v1 一致)"""

    def __init__(self, image_dir: str, mask_dir: str, file_list_path: str,
                 transform=None, empty_bg_paths: list = None,
                 num_classes: int = 20):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes

        if os.path.exists(file_list_path):
            with open(file_list_path, 'r') as f:
                self.ids = [line.strip() for line in f if line.strip()]
        else:
            self.ids = []
            print(f"Warning: File list not found: {file_list_path}")

        self.empty_ids = empty_bg_paths if empty_bg_paths else []
        self.num_normal = len(self.ids)
        self.total_len = self.num_normal + len(self.empty_ids)

    def __len__(self) -> int:
        return self.total_len

    def _load_image_and_mask(self, idx: int):
        is_empty_sample = idx >= self.num_normal
        if not is_empty_sample:
            img_id = self.ids[idx]
            img_path = os.path.join(self.image_dir, img_id + '.jpg')
            mask_path = os.path.join(self.mask_dir, img_id + '.png')
            image = Image.open(img_path).convert('RGB')
            mask = np.array(Image.open(mask_path), dtype=np.int32)
        else:
            empty_idx = idx - self.num_normal
            image = Image.open(self.empty_ids[empty_idx]).convert('RGB')
            w, h = image.size
            mask = np.zeros((h, w), dtype=np.int32)
        return image, mask, is_empty_sample

    def _generate_labels(self, mask: np.ndarray, is_empty: bool):
        if is_empty:
            single_label = 0
            multi_label = torch.zeros(self.num_classes, dtype=torch.float32)
        else:
            unique_classes = np.unique(mask)
            obj_classes = unique_classes[(unique_classes > 0) & (unique_classes < 255)]
            if len(obj_classes) == 0:
                single_label = 0
                multi_label = torch.zeros(self.num_classes, dtype=torch.float32)
            else:
                areas = [(int(c), int((mask == c).sum())) for c in obj_classes]
                single_label = max(areas, key=lambda x: x[1])[0]
                multi_label = torch.zeros(self.num_classes, dtype=torch.float32)
                for c in obj_classes:
                    multi_label[int(c) - 1] = 1.0
        return single_label, multi_label

    def __getitem__(self, idx: int):
        image, mask, is_empty = self._load_image_and_mask(idx)
        single_label, multi_label = self._generate_labels(mask, is_empty)
        is_empty_tensor = float(is_empty)

        if self.transform:
            augmented = self.transform(
                image=np.array(image),
                mask=mask.astype(np.int64)
            )
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(np.array(image).astype(np.float32)).permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'single_label': torch.tensor(single_label, dtype=torch.long),
            'multi_label': multi_label,
            'mask': mask,
            'is_empty': torch.tensor(is_empty_tensor, dtype=torch.float32)
        }


def extract_background_patches(voc_image_dir: str, voc_mask_dir: str,
                                file_list_path: str, save_dir: str,
                                num_patches: int = 800, patch_size: int = 224) -> list:
    """从 VOC 训练图像中截取纯背景区域"""
    os.makedirs(save_dir, exist_ok=True)
    existing = [os.path.join(save_dir, f) for f in os.listdir(save_dir)
                if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(existing) >= num_patches:
        print(f"Background patches already exist: {len(existing)} files")
        return existing[:num_patches]

    with open(file_list_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    random.shuffle(ids)
    bg_paths = list(existing)
    count = len(bg_paths)

    for img_id in ids:
        if count >= num_patches:
            break
        try:
            img = Image.open(os.path.join(voc_image_dir, img_id + '.jpg')).convert('RGB')
            mask = np.array(Image.open(os.path.join(voc_mask_dir, img_id + '.png')), dtype=np.int32)
            bg_mask = (mask == 0) | (mask == 255)
            h, w = bg_mask.shape
            if h < patch_size or w < patch_size:
                continue
            for _ in range(30):
                y = random.randint(0, h - patch_size)
                x = random.randint(0, w - patch_size)
                if bg_mask[y:y + patch_size, x:x + patch_size].all():
                    patch = img.crop((x, y, x + patch_size, y + patch_size))
                    save_name = f"bg_{count:05d}.jpg"
                    patch.save(os.path.join(save_dir, save_name), quality=95)
                    bg_paths.append(os.path.join(save_dir, save_name))
                    count += 1
                    break
        except Exception as e:
            print(f"  Warning: Failed to process {img_id}: {e}")
            continue
        if count % 200 == 0 and count > 0:
            print(f"  Extracted {count}/{num_patches} background patches...")

    print(f"\nTotal background patches extracted: {len(bg_paths)}")
    return bg_paths


def download_voc2012(target_root: str = './data') -> str:
    """自动下载 VOC 2012"""
    from torchvision.datasets import VOCSegmentation
    print("=" * 60)
    print("Downloading Pascal VOC 2012 (~2 GB)...")
    print("=" * 60)
    VOCSegmentation(root=target_root, year='2012', image_set='train', download=True)
    VOCSegmentation(root=target_root, year='2012', image_set='val', download=True)
    voc_root = os.path.join(target_root, 'VOCdevkit', 'VOC2012')
    print(f"Dataset downloaded to: {voc_root}")
    return voc_root


def get_transforms(input_size: int = 224, is_training: bool = True):
    """构建数据增强 pipeline (v2 增强)"""
    if is_training:
        return A.Compose([
            A.RandomResizedCrop(size=(input_size, input_size),
                                scale=(0.5, 1.0), ratio=(0.7, 1.43)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),
            A.RandomRotate90(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                        contrast_limit=0.25, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15,
                                  sat_shift_limit=25,
                                  val_shift_limit=15, p=0.35),
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.05, 0.125),
                            hole_width_range=(0.05, 0.125), p=0.2),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=input_size, width=input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================== 损失函数 (v2) ========================

class MultiTaskLoss(nn.Module):
    """
    多任务联合损失 v2
    
    支持两种模式:
      1. 固定权重模式 (原始): L = sum(w_i * L_i)
      2. Uncertainty Weighting (Kendall et al., NeurIPS 2018):
         学习每个任务的对数方差 log(sigma^2) 作为权重,
         总损失 = sum(1/(2*sigma_i^2)*L_i + log(sigma_i)) + lambda * sum(log(sigma_i)^2)
         其中 lambda=1e-4 为 L2 正则化权重, 防止 log_var 过大
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_uw = getattr(cfg, 'USE_UW', False)

        # 原始损失函数
        self.criterion_single = nn.CrossEntropyLoss()
        self.criterion_multi = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        self.criterion_empty = nn.BCEWithLogitsLoss()
        self.criterion_seg = SegmentationLoss(num_classes=21, dice_weight=1.0)

        # Uncertainty Weighting 参数 (可学习)
        if self.use_uw:
            self.log_vars_single = nn.Parameter(torch.zeros(1))   # 单分类
            self.log_vars_multi = nn.Parameter(torch.zeros(1))    # 多标签
            self.log_vars_empty = nn.Parameter(torch.zeros(1))    # 空检测
            self.log_vars_seg = nn.Parameter(torch.zeros(1))      # 分割
            self.log_var_min = -5.0
            self.log_var_max = 5.0
            self.log_var_reg_weight = 1e-4
            print("  [UW] Uncertainty Weighting enabled — loss weights are learnable")

    def forward(self, outputs: dict, batch: dict) -> dict:
        single_logit = outputs['single_logit']
        multi_logit = outputs['multi_logit']
        empty_logit = outputs['empty_main_logit']
        empty_prob = outputs['empty_prob']
        seg_logit = outputs['seg_logit']

        single_label = batch['single_label']
        multi_label = batch['multi_label']
        is_empty = batch['is_empty'].unsqueeze(-1)
        mask = batch['mask']

        B, _, h, w = seg_logit.shape
        if seg_logit.shape[2:] != mask.shape[1:]:
            seg_logit = F.interpolate(seg_logit, size=mask.shape[1:],
                                       mode='bilinear', align_corners=False)

        # 各任务原始损失
        loss_single = self.criterion_single(single_logit, single_label)
        loss_multi = self.criterion_multi(multi_logit, multi_label)
        loss_empty_main = self.criterion_empty(empty_logit, is_empty)
        loss_empty_prob = self.criterion_empty(empty_prob, is_empty)
        loss_empty = 0.7 * loss_empty_main + 0.3 * loss_empty_prob
        loss_seg = self.criterion_seg(seg_logit, mask)

        if self.use_uw and self.training:
            # Clamp log variances to keep learned task weights numerically stable.
            log_var_s = self.log_vars_single.clamp(self.log_var_min, self.log_var_max)
            log_var_m = self.log_vars_multi.clamp(self.log_var_min, self.log_var_max)
            log_var_e = self.log_vars_empty.clamp(self.log_var_min, self.log_var_max)
            log_var_sg = self.log_vars_seg.clamp(self.log_var_min, self.log_var_max)

            precision_s = torch.exp(-log_var_s) * 0.5
            precision_m = torch.exp(-log_var_m) * 0.5
            precision_e = torch.exp(-log_var_e) * 0.5
            precision_sg = torch.exp(-log_var_sg) * 0.5

            log_var_reg = self.log_var_reg_weight * (
                self.log_vars_single.pow(2) +
                self.log_vars_multi.pow(2) +
                self.log_vars_empty.pow(2) +
                self.log_vars_seg.pow(2)
            )

            total_loss = (
                precision_s * loss_single + log_var_s +
                precision_m * loss_multi + log_var_m +
                precision_e * loss_empty + log_var_e +
                precision_sg * loss_seg + log_var_sg +
                log_var_reg
            )
        else:
            # 固定权重模式
            total_loss = (
                self.cfg.LOSS_SINGLE_W * loss_single +
                self.cfg.LOSS_MULTI_W * loss_multi +
                self.cfg.LOSS_EMPTY_W * loss_empty +
                self.cfg.LOSS_SEG_W * loss_seg
            )

        return {
            'total': total_loss,
            'single': loss_single,
            'multi': loss_multi,
            'empty': loss_empty,
            'seg': loss_seg
        }

    def get_effective_weights(self) -> dict:
        """返回当前有效权重 (用于日志/调试)"""
        if self.use_uw:
            log_var_s = self.log_vars_single.clamp(self.log_var_min, self.log_var_max)
            log_var_m = self.log_vars_multi.clamp(self.log_var_min, self.log_var_max)
            log_var_e = self.log_vars_empty.clamp(self.log_var_min, self.log_var_max)
            log_var_sg = self.log_vars_seg.clamp(self.log_var_min, self.log_var_max)
            return {
                'single': float(torch.exp(-log_var_s).item()),
                'multi': float(torch.exp(-log_var_m).item()),
                'empty': float(torch.exp(-log_var_e).item()),
                'seg': float(torch.exp(-log_var_sg).item()),
            }
        else:
            return {
                'single': self.cfg.LOSS_SINGLE_W,
                'multi': self.cfg.LOSS_MULTI_W,
                'empty': self.cfg.LOSS_EMPTY_W,
                'seg': self.cfg.LOSS_SEG_W,
            }


# ======================== 评估指标 (v2: 支持 EMA) ====================

@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader,
             device: str, num_classes: int = 20) -> dict:
    """在验证集上评估模型性能"""
    model.eval()
    total_samples = 0
    correct_single = 0
    correct_empty = 0
    tp_multi = torch.zeros(num_classes, device=device)
    fp_multi = torch.zeros(num_classes, device=device)
    fn_multi = torch.zeros(num_classes, device=device)
    seg_intersection = torch.zeros(num_classes + 1, device=device)
    seg_union = torch.zeros(num_classes + 1, device=device)

    criterion_single = nn.CrossEntropyLoss()
    criterion_multi = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    criterion_seg = SegmentationLoss(num_classes=21, dice_weight=1.0)
    total_loss = 0.0

    for batch in data_loader:
        images = batch['image'].to(device)
        single_lbl = batch['single_label'].to(device)
        multi_lbl = batch['multi_label'].to(device)
        empty_lbl = batch['is_empty'].to(device)
        masks = batch['mask'].to(device)
        B = images.shape[0]

        # v2 修复: autocast 包含完整推理和 loss 计算
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            seg_out = F.interpolate(outputs['seg_logit'], size=masks.shape[1:],
                                     mode='bilinear', align_corners=False)

            loss_s = criterion_single(outputs['single_logit'], single_lbl)
            loss_m = criterion_multi(outputs['multi_logit'], multi_lbl)
            loss_e_main = nn.BCEWithLogitsLoss()(outputs['empty_main_logit'],
                                                  empty_lbl.unsqueeze(-1))
            loss_e_prob = nn.BCEWithLogitsLoss()(outputs['empty_prob'],
                                                  empty_lbl.unsqueeze(-1))
            loss_e = 0.7 * loss_e_main + 0.3 * loss_e_prob
            loss_sg = criterion_seg(seg_out, masks)
            batch_loss = loss_s.item() + loss_m.item() + loss_e.item() + loss_sg.item()

        total_loss += batch_loss * B
        total_samples += B

        pred_single = outputs['single_logit'].argmax(dim=-1)
        correct_single += (pred_single == single_lbl).sum().item()

        pred_empty = (outputs['empty_prob'] > 0.0).float().squeeze(-1)
        target_empty = (empty_lbl > 0.5).float()
        correct_empty += (pred_empty == target_empty).sum().item()

        pred_multi = (torch.sigmoid(outputs['multi_logit']) > 0.5).long()
        target_multi = (multi_lbl > 0.5).long()
        tp_multi += ((pred_multi == 1) & (target_multi == 1)).sum(dim=0)
        fp_multi += ((pred_multi == 1) & (target_multi == 0)).sum(dim=0)
        fn_multi += ((pred_multi == 0) & (target_multi == 1)).sum(dim=0)

        pred_seg = seg_out.argmax(dim=1)
        for cls_idx in range(num_classes + 1):
            pred_cls = (pred_seg == cls_idx)
            target_cls = (masks == cls_idx)
            valid = (masks != 255)
            intersection = (pred_cls & target_cls & valid).sum(dim=(1, 2))
            union = ((pred_cls | target_cls) & valid).sum(dim=(1, 2))
            seg_intersection[cls_idx] += intersection.sum()
            seg_union[cls_idx] += union.sum()

    valid_seg_classes = seg_union > 0
    if valid_seg_classes.any():
        mean_iou = (seg_intersection[valid_seg_classes] / seg_union[valid_seg_classes]).mean().item()
    else:
        mean_iou = 0.0

    per_class_iou = torch.full_like(seg_intersection, float('nan'), dtype=torch.float32)
    per_class_iou[valid_seg_classes] = seg_intersection[valid_seg_classes] / seg_union[valid_seg_classes]

    metrics = {
        'loss': total_loss / max(total_samples, 1),
        'acc_single': correct_single / max(total_samples, 1),
        'acc_empty': correct_empty / max(total_samples, 1),
        'mIoU': mean_iou,
    }

    precision = tp_multi / (tp_multi + fp_multi).clamp(min=1e-6)
    recall = tp_multi / (tp_multi + fn_multi).clamp(min=1e-6)
    metrics['mF1'] = (2 * precision * recall / (precision + recall).clamp(min=1e-6)).mean().item()
    metrics['per_class_iou'] = per_class_iou.cpu().numpy().tolist()

    return metrics


def format_metrics(metrics: dict, epoch: int, phase: str = 'Val') -> str:
    lines = [
        f"\n{'='*60}",
        f"[{phase}] Epoch {epoch}",
        f"{'='*60}",
        f"  Total Loss:     {metrics['loss']:.4f}",
        f"  Single Acc:     {metrics['acc_single']*100:.2f}%",
        f"  Empty Acc:      {metrics['acc_empty']*100:.2f}%",
        f"  Multi-label mF1:{metrics['mF1']*100:.2f}%",
        f"  Segmentation mIoU: {metrics['mIoU']*100:.2f}%",
        f"  Per-class IoU: ",
    ]
    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    ious = metrics.get('per_class_iou', [])
    for name, iou in zip(class_names[:len(ious)], ious):
        lines.append(f"    {name:>12s}: {iou*100:.1f}%")
    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)


# ======================== 主训练循环 (v2) ========================

def train(cfg: Config = None):
    """主训练入口 v2"""
    if cfg is None:
        cfg = Config()

    set_seed(cfg.SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---------- 1. 准备数据 ----------
    print("\n[1/5] Preparing datasets...")
    # 自动搜索本地 VOC 数据集（多路径回退）
    search_paths = [cfg.VOC_ROOT, './data/VOCdevkit/VOC2012', '../data/VOCdevkit/VOC2012']
    VOC_ROOT = None
    for path in search_paths:
        if os.path.exists(os.path.join(path, 'JPEGImages')):
            VOC_ROOT = os.path.abspath(path)
            print(f"Using LOCAL VOC dataset at: {VOC_ROOT}")
            break
    if VOC_ROOT is None:
        print("Local VOC dataset not found. Downloading...")
        VOC_ROOT = download_voc2012(target_root='./data')

    IMG_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
    MASK_DIR = os.path.join(VOC_ROOT, 'SegmentationClass')
    TRAIN_LIST = os.path.join(VOC_ROOT, 'ImageSets', 'Segmentation', 'train.txt')
    VAL_LIST = os.path.join(VOC_ROOT, 'ImageSets', 'Segmentation', 'val.txt')

    if not os.path.exists(cfg.BG_PATCH_DIR):
        os.makedirs(cfg.BG_PATCH_DIR, exist_ok=True)
    bg_files = [f for f in os.listdir(cfg.BG_PATCH_DIR)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(bg_files) < cfg.NUM_BG_PATCHES // 2:
        print("Extracting background patches...")
        bg_paths = extract_background_patches(
            IMG_DIR, MASK_DIR, TRAIN_LIST,
            cfg.BG_PATCH_DIR, num_patches=cfg.NUM_BG_PATCHES,
            patch_size=cfg.INPUT_SIZE
        )
    else:
        bg_paths = [os.path.join(cfg.BG_PATCH_DIR, f) for f in bg_files]
        print(f"Found {len(bg_paths)} existing background patches.")

    train_dataset = VOCMultiTaskDataset(
        IMG_DIR, MASK_DIR, TRAIN_LIST,
        transform=get_transforms(cfg.INPUT_SIZE, is_training=True),
        empty_bg_paths=bg_paths, num_classes=cfg.NUM_CLASSES)
    val_dataset = VOCMultiTaskDataset(
        IMG_DIR, MASK_DIR, VAL_LIST,
        transform=get_transforms(cfg.INPUT_SIZE, is_training=False),
        empty_bg_paths=[], num_classes=cfg.NUM_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                               shuffle=True, num_workers=cfg.NUM_WORKERS,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE,
                             shuffle=False, num_workers=cfg.NUM_WORKERS,
                             pin_memory=True)

    print(f"  Train samples: {len(train_dataset)} "
          f"({len(train_dataset)-len(bg_paths)} normal + {len(bg_paths)} empty)")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---------- 2. 构建模型 ----------
    print("\n[2/5] Building model...")
    model = build_model(num_classes=cfg.NUM_CLASSES,
                        pretrained_path=cfg.PRETRAINED_WEIGHTS,
                        timm_model_name=cfg.TIMM_MODEL_NAME,
                        device=device)

    # 冻结骨干网络防爆
    print("  Freezing backbone to prevent gradient explosion...")
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 编译模型加速训练（PyTorch 2.0+；Linux 支持，Windows 暂不支持）
    if hasattr(torch, 'compile') and not sys.platform.startswith('win'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # ---------- 3. 损失函数、优化器 & 调度器 ----------
    print("\n[3/5] Setting up optimizer & scheduler...")
    criterion = MultiTaskLoss(cfg)

    seg_params = [p for p in model.seg_decoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith('backbone.') and not n.startswith('seg_decoder.') and p.requires_grad]

    param_groups = [
        {'params': head_params, 'lr': cfg.HEAD_LR},
        {'params': seg_params, 'lr': cfg.SEG_LR},
    ]
    if cfg.USE_UW:
        param_groups.append({
            'params': [p for p in criterion.parameters() if p.requires_grad],
            'lr': cfg.HEAD_LR,
            'weight_decay': 0.0,
        })

    optimizer = AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)

    warmup_iters = cfg.WARMUP_EPOCHS * len(train_loader)
    total_iters = cfg.EPOCHS * len(train_loader)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_iters - warmup_iters, 1),
                                          eta_min=cfg.MIN_LR_RATIO * cfg.HEAD_LR)
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup_scheduler, cosine_scheduler],
                              milestones=[warmup_iters])

    scaler = GradScaler(enabled=cfg.AMP_ENABLED and device == 'cuda')

    # ---------- 4. EMA (新增) ----------
    ema = None
    if cfg.USE_EMA:
        ema = ModelEMA(model, decay=cfg.EMA_DECAY, device=device)
        print(f"  [EMA] Enabled (decay={cfg.EMA_DECAY})")

    # ---------- 5. Early Stopping (新增) ----------
    early_stopper = EarlyStopping(
        patience=getattr(cfg, 'EARLY_STOPPING_PATIENCE', 15),
        mode='max', min_delta=0.001
    )
    print(f"  [EarlyStop] Patience={early_stopper.patience} epochs (monitor: val mIoU)")

    # ---------- 6. 保存目录 ----------
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # ========== 训练配置摘要 ==========
    print("\n[4/5] Training config:")
    print(f"{'='*70}")
    print(f"  Epochs:         {cfg.EPOCHS}")
    print(f"  Batch size:     {cfg.BATCH_SIZE} (effective: {cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS})")
    print(f"  Backbone LR:    {cfg.BACKBONE_LR}")
    print(f"  Head LR:        {cfg.HEAD_LR}")
    print(f"  Seg LR:         {cfg.SEG_LR}")
    print(f"  Warmup epochs:  {cfg.WARMUP_EPOCHS}")
    print(f"  AMP:            {'Enabled' if cfg.AMP_ENABLED else 'Disabled'}")
    print(f"  EMA:            {'Enabled' if cfg.USE_EMA else 'Disabled'}")
    print(f"  UW:             {'Enabled' if cfg.USE_UW else 'Disabled'}")
    print(f"  Early Stopping: Patience={early_stopper.patience}")
    if cfg.USE_UW:
        print(f"  Loss weights:   Learnable (Uncertainty Weighting)")
    else:
        print(f"  Loss weights:   S={cfg.LOSS_SINGLE_W}, M={cfg.LOSS_MULTI_W}, "
              f"E={cfg.LOSS_EMPTY_W}, Seg={cfg.LOSS_SEG_W}")
    print(f"{'='*70}\n")

    best_miou = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_metrics': []}
    start_time = time.time()
    stopped_early = False

    backbone_frozen = True
    for epoch in range(1, cfg.EPOCHS + 1):
        # Warmup 结束后解冻 backbone
        if backbone_frozen and epoch > cfg.WARMUP_EPOCHS:
            print(f"\n  [Unfreeze] Warmup done. Unfreezing backbone at epoch {epoch}...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                'params': list(model.backbone.parameters()),
                'lr': cfg.BACKBONE_LR
            })

            # 同步通知调度器新参数组，否则 Backbone LR 不会衰减
            if hasattr(cosine_scheduler, 'base_lrs'):
                cosine_scheduler.base_lrs.append(cfg.BACKBONE_LR)

            backbone_frozen = False

            # 重建 EMA 以包含新解冻的 backbone，并调低衰减率适应小数据集
            if ema is not None:
                print("  [EMA] Re-building EMA to include the unfrozen backbone...")
                ema = ModelEMA(model, decay=0.999, device=device)

        model.train()
        epoch_loss = 0.0
        epoch_losses = {'single': 0, 'multi': 0, 'empty': 0, 'seg': 0}
        pbar_total = len(train_loader)

        for step, batch in enumerate(train_loader):
            images = batch['image'].to(device, non_blocking=True)
            single_lbl = batch['single_label'].to(device, non_blocking=True)
            multi_lbl = batch['multi_label'].to(device, non_blocking=True)
            empty_lbl = batch['is_empty'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            batch = {
                'image': images,
                'single_label': single_lbl,
                'multi_label': multi_lbl,
                'is_empty': empty_lbl,
                'mask': masks,
            }

            # v2 修复: autocast 包含前向传播 AND loss 计算
            with autocast(enabled=cfg.AMP_ENABLED and device == 'cuda'):
                outputs = model(images)
                losses = criterion(outputs, batch)
                loss = losses['total'] / cfg.ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % cfg.ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                all_params = list(model.parameters()) + [p for p in criterion.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # v2: EMA 更新 (每个累积步后更新一次)
                if ema is not None:
                    ema.update(model)

            epoch_loss += losses['total'].item()
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            if (step + 1) % cfg.LOG_INTERVAL == 0 or step == 0:
                lr = optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / (step + 1)
                uw_info = ""
                if cfg.USE_UW and hasattr(criterion, 'get_effective_weights'):
                    w = criterion.get_effective_weights()
                    uw_info = f" | UW_w:[{w['single']:.2f},{w['multi']:.2f},{w['empty']:.2f},{w['seg']:.2f}]"
                print(f"  Epoch [{epoch}/{cfg.EPOCHS}] Step [{step+1}/{pbar_total}] "
                      f"Loss: {avg_loss:.4f} (S:{epoch_losses['single']/(step+1):.3f} "
                      f"M:{epoch_losses['multi']/(step+1):.3f} "
                      f"E:{epoch_losses['empty']/(step+1):.3f} "
                      f"Seg:{epoch_losses['seg']/(step+1):.3f})"
                      f" LR: {lr:.6f}{uw_info}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_epoch_loss)

        elapsed = time.time() - start_time
        print(f"\n  >>> Epoch {epoch}/{cfg.EPOCHS} completed. "
              f"Avg Loss: {avg_epoch_loss:.4f}, Time: {elapsed/60:.1f} min\n")

        # ---------- 验证 ----------
        if epoch % cfg.VAL_INTERVAL == 0 or epoch == 1 or epoch == cfg.EPOCHS:
            print("[5/5] Validating...")

            # v2: 使用 EMA 权重进行评估
            use_ema_for_eval = ema is not None
            if use_ema_for_eval:
                ema.apply_shadow(model)

            val_metrics = evaluate(model, val_loader, device, cfg.NUM_CLASSES)
            history['val_metrics'].append({'epoch': epoch, **val_metrics})

            eval_tag = "EMA" if use_ema_for_eval else "Normal"
            print(format_metrics(val_metrics, epoch, phase=f'Val[{eval_tag}]'))

            # 恢复训练权重
            if use_ema_for_eval:
                ema.restore(model)

            current_miou = val_metrics['mIoU']

            # 保存最佳模型 (按 mIoU)
            if current_miou > best_miou:
                best_miou = current_miou
                best_epoch = epoch
                save_path = os.path.join(cfg.SAVE_DIR, 'best_model.pth')
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_miou': best_miou,
                    'config': vars(cfg),
                    'history': history
                }
                if ema is not None:
                    ckpt['ema_state_dict'] = ema.state_dict()
                if cfg.USE_UW:
                    ckpt['log_vars'] = {
                        'log_vars_single': criterion.log_vars_single.data.tolist(),
                        'log_vars_multi': criterion.log_vars_multi.data.tolist(),
                        'log_vars_empty': criterion.log_vars_empty.data.tolist(),
                        'log_vars_seg': criterion.log_vars_seg.data.tolist(),
                    }
                torch.save(ckpt, save_path)
                print(f"  *** Best model saved! mIoU={best_miou*100:.2f}% @ Epoch {epoch}\n")

            # v2: Early Stopping 检查
            if early_stopper.step(current_miou):
                stopped_early = True
                break

    # ========== 保存最终模型 ==========
    final_path = os.path.join(cfg.SAVE_DIR, 'final_model.pth')

    # 如果有 EMA, 最终也保存一份 EMA 权重的模型
    if ema is not None:
        ema.apply_shadow(model)
        ema_path = os.path.join(cfg.SAVE_DIR, 'final_model_ema.pth')
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'config': vars(cfg), 'history': history, 'best_miou': best_miou,
            'ema_state_dict': ema.state_dict(), 'is_ema': True
        }, ema_path)
        ema.restore(model)
        print(f"  EMA model saved: {ema_path}")

    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'config': vars(cfg), 'history': history
    }, final_path)

    with open(os.path.join(cfg.SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    if stopped_early:
        print(f"Training stopped early by EarlyStopping!")
    print(f"Training completed!")
    print(f"  Total time:     {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Best mIoU:      {best_miou*100:.2f}% at Epoch {best_epoch}")
    print(f"  Total epochs:   {epoch}/{cfg.EPOCHS}")
    print(f"  Models saved to: {cfg.SAVE_DIR}/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Task ViT-Small v2 Training on VOC 2012')
    # 默认值设为 None，未指定时优先使用 Config 类属性
    parser.add_argument('--epochs', type=int, default=None, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--backbone_lr', type=float, default=None, help='Backbone LR')
    parser.add_argument('--head_lr', type=float, default=None, help='Head LR')
    parser.add_argument('--seg_lr', type=float, default=None, help='Segmentation decoder LR')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to MAE pretrained weights, or "imagenet" for timm ImageNet weights')
    parser.add_argument('--timm_model', type=str, default=None,
                        help='timm model name used when --pretrained imagenet')
    parser.add_argument('--input_size', type=int, default=None, help='Input image size')
    parser.add_argument('--num_bg_patches', type=int, default=None,
                        help='Number of background patches')
    parser.add_argument('--loss_single_w', type=float, default=None,
                        help='Weight for auxiliary single-label softmax loss')
    parser.add_argument('--loss_multi_w', type=float, default=None,
                        help='Weight for multi-label sigmoid loss')
    parser.add_argument('--loss_empty_w', type=float, default=None,
                        help='Weight for empty/background detection loss')
    parser.add_argument('--loss_seg_w', type=float, default=None,
                        help='Weight for segmentation loss')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    # v2 新增参数
    parser.add_argument('--ema', action='store_true',
                        help='Enable EMA explicitly (already enabled by default, use --no_ema to disable)')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable EMA (default: enabled)')
    parser.add_argument('--uw', action='store_true',
                        help='Enable Uncertainty Weighting for loss balancing')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='Early stopping patience (default: 15)')

    args = parser.parse_args()

    # 只将用户显式指定的命令行参数传给 Config，其余保留 Config 类属性默认值
    kwargs = {}
    if args.epochs is not None:
        kwargs['EPOCHS'] = args.epochs
    if args.batch_size is not None:
        kwargs['BATCH_SIZE'] = args.batch_size
    if args.backbone_lr is not None:
        kwargs['BACKBONE_LR'] = args.backbone_lr
    if args.head_lr is not None:
        kwargs['HEAD_LR'] = args.head_lr
    if args.seg_lr is not None:
        kwargs['SEG_LR'] = args.seg_lr
    if args.pretrained is not None:
        kwargs['PRETRAINED_WEIGHTS'] = args.pretrained
    if args.timm_model is not None:
        kwargs['TIMM_MODEL_NAME'] = args.timm_model
    if args.input_size is not None:
        kwargs['INPUT_SIZE'] = args.input_size
    if args.num_bg_patches is not None:
        kwargs['NUM_BG_PATCHES'] = args.num_bg_patches
    if args.loss_single_w is not None:
        kwargs['LOSS_SINGLE_W'] = args.loss_single_w
    if args.loss_multi_w is not None:
        kwargs['LOSS_MULTI_W'] = args.loss_multi_w
    if args.loss_empty_w is not None:
        kwargs['LOSS_EMPTY_W'] = args.loss_empty_w
    if args.loss_seg_w is not None:
        kwargs['LOSS_SEG_W'] = args.loss_seg_w
    if args.no_amp:
        kwargs['AMP_ENABLED'] = False
    if args.no_ema:
        kwargs['USE_EMA'] = False
    if args.uw:
        kwargs['USE_UW'] = True
    if args.early_stop_patience is not None:
        kwargs['EARLY_STOPPING_PATIENCE'] = args.early_stop_patience

    config = Config(**kwargs)

    train(config)
