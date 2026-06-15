"""
多任务 ViT-Small 测试/推理脚本
===============================
支持:
  1. 单张图片推理 (分类 + 分割 + 空检测)
  2. 高分辨率图像: 滑动窗口 + 多尺度 + 分割精细化
  3. 批量测试评估 (mIoU, 准确率等)
  4. 结果可视化保存

用法:
    # 单张图片标准测试 (快速)
    python test.py --image path/to/image.jpg

    # 高分辨率图片 (滑动窗口+多尺度, 推荐!)
    python test.py --image large_image.jpg --sliding_window --multi_scale --vis

    # 批量评估验证集
    python test.py --eval --checkpoint checkpoints/best_model.pth

    # 自定义参数
    python test.py --image big.jpg --stride_ratio 0.5 --scales 0.75 1.0 1.25 1.5 --refine
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import MultiTaskViTSmall, build_model


# ======================== VOC 类别名 ========================
VOC_CLASS_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 分割结果配色 (RGB, 用于 PIL / matplotlib)
SEG_COLORS = [
    [0, 0, 0],          # 0: background - black
    [128, 0, 0],        # 1: aeroplane
    [0, 128, 0],        # 2: bicycle
    [128, 128, 0],      # 3: bird
    [0, 0, 128],        # 4: boat
    [128, 0, 128],      # 5: bottle
    [0, 128, 128],      # 6: bus
    [128, 128, 128],    # 7: car
    [64, 0, 0],         # 8: cat
    [192, 0, 0],        # 9: chair
    [64, 128, 0],       # 10: cow
    [192, 128, 0],      # 11: diningtable
    [64, 0, 128],       # 12: dog
    [192, 0, 128],      # 13: horse
    [64, 128, 128],     # 14: motorbike
    [192, 128, 128],    # 15: person
    [0, 64, 0],         # 16: pottedplant
    [128, 64, 0],       # 17: sheep
    [0, 192, 0],        # 18: sofa
    [128, 192, 0],      # 19: train
    [0, 64, 128]        # 20: tvmonitor
]


def load_model(checkpoint_path: str,
               pretrained_backbone: str = None,
               timm_model_name: str = 'vit_small_patch16_224',
               num_classes: int = 20,
               device: str = 'cuda') -> MultiTaskViTSmall:
    """
    加载训练好的模型 checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    model = build_model(
        num_classes=num_classes,
        pretrained_path=pretrained_backbone,
        timm_model_name=timm_model_name,
        device='cpu'
    )

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        miou = ckpt.get('best_miou', '?')
        print(f"  Loaded from epoch {epoch}, best mIoU={miou}")
    else:
        model.load_state_dict(ckpt)
        print("  Loaded raw state dict")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_single(model: nn.Module,
                   image_path: str,
                   input_size: int = 224,
                   device: str = 'cuda',
                   use_sliding_window: bool = False,
                   stride_ratio: float = 0.5,
                   scales=None,
                   seg_refine: bool = True) -> dict:
    """
    对单张图片进行完整的多任务推理

    支持两种模式:
      1. 标准模式 (use_sliding_window=False):
         直接 Resize 到 input_size 后推理, 速度快但高分辨率效果差

      2. 高分辨率模式 (use_sliding_window=True):
         滑动窗口 + 多尺度融合 + 分割精细化, 效果好但速度较慢

    Returns:
        dict 包含:
            single_class: int (0=空/背景, 1~20=物体类别)
            single_class_name: str
            multi_classes: list[int] 检测到的物体类别列表
            multi_class_names: list[str]
            is_empty_prob: float (0~1, larger means more likely empty/background)
            objectness_prob: float (0~1, larger means more likely containing objects)
            is_empty: bool
            seg_mask: np.ndarray (H, W) 像素级分割标签
            seg_mask_color: np.ndarray (H, W, 3) 彩色分割图
            inference_time: float 推理耗时(秒)
            mode: str 使用的推理模式
            original_image: np.ndarray (H, W, 3) 原始输入图像
    """
    if scales is None:
        scales = [1.0]

    device = next(model.parameters()).device

    # ---- 加载图片 ----
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    img_np = np.array(image)

    # ---- 判断使用哪种推理模式 ----
    is_high_res = use_sliding_window and (orig_h > input_size or orig_w > input_size)

    t_start = time.time()

    if is_high_res:
        # ========== 高分辨率模式: 滑动窗口 + 多尺度 ==========
        result = _predict_high_res(
            model, img_np, orig_h, orig_w,
            input_size, stride_ratio, scales, seg_refine, device
        )
        mode = f"SlidingWindow(scales={scales}, stride={stride_ratio})"
    else:
        # ========== 标准模式: 直接缩放 ==========
        result = _predict_standard(
            model, img_np, orig_h, orig_w, input_size, seg_refine, device
        )
        mode = "Standard(Resize)"

    t_end = time.time()
    result['inference_time'] = round(t_end - t_start, 3)
    result['mode'] = mode
    result['original_image'] = img_np

    return result


def _predict_standard(model, img_np, orig_h, orig_w,
                      input_size, seg_refine, device):
    """标准推理: 缩放到固定尺寸"""
    transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    input_tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

    with autocast(enabled=device == 'cuda'):
        outputs = model(input_tensor)

    return _parse_outputs(outputs, model, img_np, orig_h, orig_w,
                           input_size, seg_refine)


def _predict_high_res(model, img_np, orig_h, orig_w,
                      base_size, stride_ratio, scales, seg_refine, device):
    """高分辨率推理: 分类与分离管线

    步骤 A: 将大图 Resize 到 512x512, 单次前向取分类+空检测
    步骤 B: 空检测判定为 Empty 则直接返回全黑掩码, 提前结束
    步骤 C: 有物体则对原图做滑动窗口, 只取分割 Logits
    """
    normalize = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # ========== 步骤 A: Resize 到 512x512, 取分类 + 空检测 ==========
    cls_size = 512
    transform_cls = A.Compose([
        A.Resize(cls_size, cls_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    cls_tensor = transform_cls(image=img_np)['image'].unsqueeze(0).to(device)

    with autocast(enabled=device == 'cuda'):
        cls_outputs = model(cls_tensor)

    single_logit = cls_outputs['single_logit'][0]
    multi_logit = cls_outputs['multi_logit'][0]
    empty_prob = torch.sigmoid(cls_outputs['empty_prob'])[0].item()
    single_pred = single_logit.argmax(dim=-1).item()

    multi_probs = torch.sigmoid(multi_logit)
    multi_preds = (multi_probs > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
    multi_class_ids = [p + 1 for p in multi_preds]

    # ========== 步骤 B: 空检测判定 ==========
    if empty_prob > 0.5 or single_pred == 0:
        # 判定为空图, 直接返回全黑掩码, 提前结束
        seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        return {
            'single_class': single_pred,
            'single_class_name': VOC_CLASS_NAMES[single_pred],
            'multi_classes': multi_class_ids,
            'multi_class_names': [VOC_CLASS_NAMES[c] for c in multi_class_ids],
            'is_empty_prob': empty_prob,
            'objectness_prob': 1.0 - empty_prob,
            'is_empty': True,
            'seg_mask': seg_mask,
            'seg_mask_color': color_mask,
        }

    # ========== 步骤 C: 有物体 → 滑动窗口只做分割 ==========
    input_tensor = normalize(image=img_np)['image'].unsqueeze(0).to(device)  # (1,3,H,W)

    # 滑动窗口拼合分割 Logits (不取分类输出)
    nc = model.total_classes
    seg_acc = torch.zeros(1, nc, orig_h, orig_w, device=device, dtype=torch.float32)
    weight_map = torch.zeros(1, 1, orig_h, orig_w, device=device, dtype=torch.float32)
    gaussian = _create_gaussian_weight(base_size, sigma=base_size // 8)

    y_starts = list(range(0, max(orig_h - base_size, 0) + 1, int(base_size * stride_ratio)))
    x_starts = list(range(0, max(orig_w - base_size, 0) + 1, int(base_size * stride_ratio)))
    # 确保覆盖到边缘
    if not y_starts or y_starts[-1] != max(orig_h - base_size, 0):
        y_starts.append(max(orig_h - base_size, 0))
    if not x_starts or x_starts[-1] != max(orig_w - base_size, 0):
        x_starts.append(max(orig_w - base_size, 0))

    for ys in y_starts:
        for xs in x_starts:
            ye, xe = min(ys + base_size, orig_h), min(xs + base_size, orig_w)
            window = input_tensor[:, :, ys:ye, xs:xe]
            # pad 到 base_size x base_size
            ph, pw = base_size - (ye - ys), base_size - (xe - xs)
            if ph > 0 or pw > 0:
                window = F.pad(window, (0, pw, 0, ph), mode='reflect')

            with autocast(enabled=device == 'cuda'):
                out = model(window)

            ah, aw = ye - ys, xe - xs
            gw = gaussian[:, :, :ah, :aw].to(device)
            seg_acc[:, :, ys:ye, xs:xe] += out['seg_logit'][:, :, :ah, :aw] * gw
            weight_map[:, :, ys:ye, xs:xe] += gw

    seg_fused = seg_acc / weight_map.clamp(min=1e-6)

    # 多尺度 (如果指定了额外尺度)
    if len(scales) > 1 or scales[0] != 1.0:
        all_seg = [seg_fused]
        for scale in scales:
            if scale == 1.0:
                continue
            sh, sw = int(orig_h * scale), int(orig_w * scale)
            img_s = F.interpolate(input_tensor, size=(sh, sw), mode='bilinear', align_corners=False)
            scale_acc = torch.zeros(1, nc, sh, sw, device=device, dtype=torch.float32)
            scale_wmap = torch.zeros(1, 1, sh, sw, device=device, dtype=torch.float32)
            sy = list(range(0, max(sh - base_size, 0) + 1, int(base_size * stride_ratio)))
            sx = list(range(0, max(sw - base_size, 0) + 1, int(base_size * stride_ratio)))
            if not sy or sy[-1] != max(sh - base_size, 0):
                sy.append(max(sh - base_size, 0))
            if not sx or sx[-1] != max(sw - base_size, 0):
                sx.append(max(sw - base_size, 0))
            for yy in sy:
                for xx in sx:
                    ye2, xe2 = min(yy + base_size, sh), min(xx + base_size, sw)
                    w2 = img_s[:, :, yy:ye2, xx:xe2]
                    ph2, pw2 = base_size - (ye2 - yy), base_size - (xe2 - xx)
                    if ph2 > 0 or pw2 > 0:
                        w2 = F.pad(w2, (0, pw2, 0, ph2), mode='reflect')
                    with autocast(enabled=device == 'cuda'):
                        o2 = model(w2)
                    ah2, aw2 = ye2 - yy, xe2 - xx
                    gw2 = gaussian[:, :, :ah2, :aw2].to(device)
                    scale_acc[:, :, yy:ye2, xx:xe2] += o2['seg_logit'][:, :, :ah2, :aw2] * gw2
                    scale_wmap[:, :, yy:ye2, xx:xe2] += gw2
            scale_fused = scale_acc / scale_wmap.clamp(min=1e-6)
            scale_fused = F.interpolate(scale_fused, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            all_seg.append(scale_fused)
        seg_fused = torch.stack(all_seg).mean(0)

    seg_raw = seg_fused.argmax(dim=1).squeeze(0).cpu().numpy()

    if seg_refine:
        seg_mask = MultiTaskViTSmall._refine_segmentation(
            seg_raw.astype(np.uint8), orig_h, orig_w
        )
    else:
        seg_mask = seg_raw.astype(np.uint8)

    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls_id in range(len(SEG_COLORS)):
        color_mask[seg_mask == cls_id] = SEG_COLORS[cls_id]

    return {
        'single_class': single_pred,
        'single_class_name': VOC_CLASS_NAMES[single_pred],
        'multi_classes': multi_class_ids,
        'multi_class_names': [VOC_CLASS_NAMES[c] for c in multi_class_ids],
        'is_empty_prob': empty_prob,
        'objectness_prob': 1.0 - empty_prob,
        'is_empty': False,
        'seg_mask': seg_mask,
        'seg_mask_color': color_mask,
    }


def _create_gaussian_weight(size, sigma):
    """创建 2D 高斯权重用于滑动窗口拼合"""
    coord = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coord ** 2) / (2 * sigma ** 2))
    w2d = g.unsqueeze(1) * g.unsqueeze(0)
    return w2d.unsqueeze(0).unsqueeze(0)


def _parse_outputs(outputs, model, img_np, orig_h, orig_w,
                   input_size, seg_refine, is_high_res=False):
    """统一解析模型输出为结果字典

    Args:
        outputs: 模型输出字典。标准模式需包含 'single_logit', 'multi_logit',
                 'empty_prob', 'seg_logit'；高分辨率模式需包含 'seg_mask'。
    """

    # 单分类
    single_logit = outputs['single_logit'][0]
    single_pred = single_logit.argmax(dim=-1).item()   # 0~20

    # 多分类
    multi_logit = outputs['multi_logit'][0]
    multi_probs = torch.sigmoid(multi_logit)
    multi_preds = (multi_probs > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
    multi_class_ids = [p + 1 for p in multi_preds]  # 1~20

    # 空检测
    empty_prob = torch.sigmoid(outputs['empty_prob'])[0].item()
    objectness_prob = 1.0 - empty_prob

    # 分割掩码
    if is_high_res and 'seg_mask' in outputs:
        seg_mask = outputs['seg_mask']  # 已经是 np.uint8 精细化后的
    else:
        seg_logit = outputs['seg_logit']
        seg_upsampled = F.interpolate(seg_logit, size=(orig_h, orig_w),
                                       mode='bilinear', align_corners=False)
        seg_raw = seg_upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
        if seg_refine:
            seg_mask = MultiTaskViTSmall._refine_segmentation(
                seg_raw.astype(np.uint8), orig_h, orig_w
            )
        else:
            seg_mask = seg_raw.astype(np.uint8)

    # 彩色分割图
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls_id in range(len(SEG_COLORS)):
        color_mask[seg_mask == cls_id] = SEG_COLORS[cls_id]

    return {
        'single_class': single_pred,
        'single_class_name': VOC_CLASS_NAMES[single_pred],
        'multi_classes': multi_class_ids,
        'multi_class_names': [VOC_CLASS_NAMES[c] for c in multi_class_ids],
        'is_empty_prob': empty_prob,
        'objectness_prob': objectness_prob,
        'is_empty': empty_prob > 0.5 or single_pred == 0,
        'seg_mask': seg_mask,
        'seg_mask_color': color_mask,
    }


def visualize_prediction(result: dict, save_path: str = None):
    """
    可视化预测结果: 原图 + 分类信息 + 分割叠加
    """
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    try:
        font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=12)
    except Exception:
        font = None

    orig = result['original_image']
    color_seg = result['seg_mask_color']

    # 叠加显示: 原图 50% + 分割掩码 50%
    overlay = (orig.astype(np.float32) * 0.5 +
               color_seg.astype(np.float32) * 0.5).astype(np.uint8)

    # 判断是否使用高分辨率模式
    mode_info = f"Mode: {result.get('mode', 'Standard')}"
    time_info = f"Time: {result.get('inference_time', '?')}s"
    res_info = f"Resolution: {orig.shape[1]}x{orig.shape[0]}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig)
    axes[0].set_title(f'Original\n({res_info})', fontproperties=font, fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(color_seg)
    axes[1].set_title(f'Segmentation ({mode_info})',
                      fontproperties=font, fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    info_text = (
        f"Single Class: {result['single_class_name']} ({result['single_class']})\n"
        f"Multi Classes: {', '.join(result['multi_class_names']) if result['multi_class_names'] else 'None'}\n"
        f"Is Empty: {result['is_empty']} (prob={result['is_empty_prob']:.3f})\n"
        f"{mode_info} | {time_info}"
    )
    axes[2].text(0.05, 0.95, info_text, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].set_title('Overlay + Predictions', fontproperties=font, fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


@torch.no_grad()
def evaluate_dataset(model: nn.Module,
                     data_root: str,
                     input_size: int = 224,
                     batch_size: int = 16,
                     num_workers: int = 4,
                     device: str = 'cuda') -> dict:
    """
    在 VOC 2012 验证集上进行全面评估

    Returns:
        dict: 包含以下评估指标:
            loss: float 验证总损失
            acc_single: float 单分类准确率
            acc_empty: float 空检测准确率
            mIoU: float 平均 IoU
            mF1: float 多标签平均 F1
            per_class_iou: list 各类 IoU
    """
    from train import VOCMultiTaskDataset, get_transforms

    voc_root = os.path.join(data_root, 'VOCdevkit', 'VOC2012')
    val_list = os.path.join(voc_root, 'ImageSets', 'Segmentation', 'val.txt')

    dataset = VOCMultiTaskDataset(
        image_dir=os.path.join(voc_root, 'JPEGImages'),
        mask_dir=os.path.join(voc_root, 'SegmentationClass'),
        file_list_path=val_list,
        transform=get_transforms(input_size, is_training=False),
        num_classes=20
    )

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    from train import evaluate
    metrics = evaluate(model, loader, device, num_classes=20)

    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total Samples:     {len(dataset)}")
    print(f"  Validation Loss:   {metrics['loss']:.4f}")
    print(f"  Single-class Acc:  {metrics['acc_single']*100:.2f}%")
    print(f"  Empty Detection Acc: {metrics['acc_empty']*100:.2f}%")
    print(f"  Multi-label mF1:   {metrics['mF1']*100:.2f}%")
    print(f"  Segmentation mIoU: {metrics['mIoU']*100:.2f}%")

    print(f"\n  Per-Class IoU:")
    ious = metrics.get('per_class_iou', [])
    for idx, (name, iou) in enumerate(zip(VOC_CLASS_NAMES, ious)):
        bar = '#' * int(iou * 50)
        print(f"    {idx:>2d} {name:>14s}: {iou*100:6.2f}%  |{bar:<50}|")

    print("=" * 70 + "\n")

    return metrics


# ======================== 主入口 ========================

def main():
    parser = argparse.ArgumentParser(description='Multi-Task ViT-Small Testing/Evaluation')
    # ---- 基础参数 ----
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to MAE pretrained weights, or "imagenet" for timm ImageNet weights')
    parser.add_argument('--timm_model', type=str, default='vit_small_patch16_224',
                        help='timm model name used when --pretrained imagenet')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single image for prediction')
    parser.add_argument('--eval', action='store_true',
                        help='Run full evaluation on VOC2012 validation set')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory containing VOCdevkit')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Model input resolution (base window size)')
    parser.add_argument('--vis', action='store_true',
                        help='Visualize and save predictions')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save visualization results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')

    # ---- 高分辨率推理参数 (新增) ----
    parser.add_argument('--sliding_window', action='store_true',
                        help='Enable sliding window inference for high-res images (recommended!)')
    parser.add_argument('--stride_ratio', type=float, default=0.5,
                        help='Sliding stride / window size ratio (0.5=50%% overlap)')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0],
                        help='Multi-scale test scales, e.g. 0.75 1.0 1.25 1.5')
    parser.add_argument('--no_refine', action='store_true',
                        help='Disable segmentation post-refinement')
    parser.add_argument('--auto_high_res', action='store_true',
                        help='Automatically enable sliding_window when image > input_size')

    args = parser.parse_args()

    seg_refine = not args.no_refine

    # 加载模型
    model = load_model(
        checkpoint_path=args.checkpoint,
        pretrained_backbone=args.pretrained,
        timm_model_name=args.timm_model,
        device=args.device
    )

    # 模式选择
    if args.image is not None:
        # ===== 单张图片推理 =====
        print(f"\n{'='*60}")
        print(f" Image: {args.image}")
        print(f" Mode: {'Sliding Window' if (args.sliding_window or args.auto_high_res) else 'Standard'}")
        if args.sliding_window or args.auto_high_res:
            print(f" Scales: {args.scales}, Stride ratio: {args.stride_ratio}, Refine: {seg_refine}")
        print(f"{'='*60}")

        result = predict_single(
            model, args.image,
            input_size=args.input_size,
            device=args.device,
            use_sliding_window=(args.sliding_window or args.auto_high_res),
            stride_ratio=args.stride_ratio,
            scales=args.scales,
            seg_refine=seg_refine
        )

        # 打印结果
        print(f"\n{'─'*50}")
        print(f"  Single Class: {result['single_class_name']} "
              f"(ID: {result['single_class']})")
        print(f"  Multi Classes: {result['multi_class_names']}"
              f" ({result['multi_classes']})")
        print(f"  Empty: {'NOT Empty' if not result['is_empty'] else 'EMPTY'} "
              f"(prob={result['is_empty_prob']:.4f})")
        print(f"  Segmentation: {result['seg_mask'].shape}, "
              f"{len(np.unique(result['seg_mask']))} classes detected")
        print(f"  Inference: {result.get('mode','?')}, "
              f"{result.get('inference_time','?')}s")
        print(f"{'─'*50}\n")

        # 可视化
        if args.vis:
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            vis_path = os.path.join(args.save_dir, f'pred_{base_name}.png')
            visualize_prediction(result, vis_path)

        # 保存分割掩码
        mask_path = os.path.join(args.save_dir,
                                  f'mask_{os.path.splitext(os.path.basename(args.image))[0]}.png')
        os.makedirs(os.path.dirname(mask_path) if os.path.dirname(mask_path) else '.', exist_ok=True)
        Image.fromarray(result['seg_mask']).save(mask_path)
        print(f"Segmentation mask saved: {mask_path}")

    elif args.eval:
        # ===== 全量数据集评估 =====
        evaluate_dataset(
            model=model,
            data_root=args.data_root,
            input_size=args.input_size,
            batch_size=args.batch_size,
            device=args.device
        )

    else:
        # 默认模式: 对 test_pic 目录中的所有图片进行批量推理
        test_dir = 'test_pic'
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"\nRunning inference on {len(images)} images in '{test_dir}/'...")
            sw_mode = args.auto_high_res or args.sliding_window
            mode_str = "SlidingWindow" if sw_mode else "Standard"
            print(f"Mode: {mode_str} | Scales: {args.scales}\n")

            for img_file in sorted(images):
                img_path = os.path.join(test_dir, img_file)
                try:
                    result = predict_single(
                        model, img_path,
                        input_size=args.input_size,
                        device=args.device,
                        use_sliding_window=sw_mode,
                        stride_ratio=args.stride_ratio,
                        scales=args.scales,
                        seg_refine=seg_refine
                    )

                    status = "EMPTY" if result['is_empty'] else "OBJECT"
                    multi_str = ', '.join(result['multi_class_names']) if result['multi_class_names'] else '-'
                    t_str = f"{result.get('inference_time','?')}s"

                    print(f"  [{img_file:>30s}] "
                          f"Single={result['single_class_name']:>12s} | "
                          f"Multi=[{multi_str}] | "
                          f"Empty={status}({result['is_empty_prob']:.2f}) | "
                          f"{t_str}")

                    if args.vis:
                        base_name = os.path.splitext(img_file)[0]
                        vis_path = os.path.join(args.save_dir, f'pred_{base_name}.png')
                        visualize_prediction(result, vis_path)

                except Exception as e:
                    print(f"  [{img_file}] Error: {e}")

            print("\nDone!")
        else:
            print(f"No test images found at '{test_dir}/'. "
                  "Use --image <path> for single image or --eval for dataset evaluation.")


if __name__ == '__main__':
    main()
