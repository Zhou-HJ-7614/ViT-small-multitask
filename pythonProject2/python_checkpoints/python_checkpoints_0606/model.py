"""
Multi-Task ViT-Small 模型 (v2 - Improved)
==========================================
基于 MAE 预训练的 ViT-Small 骨干网络，实现四个任务：
  1. 单分类 (Single-class): Softmax, 21类 (20物体 + 1空/背景)
  2. 多分类 (Multi-label): Sigmoid, 20类物体
  3. 空分类 (Empty Detection): 二分类，判断图像是否包含任何物体
  4. 分割 (Segmentation): 像素级 21 类语义分割

预训练: MAE (Masked Autoencoder) on ImageNet-1k
高分辨率支持: 滑动窗口推理 + 多尺度融合
v2 改进: U-Net skip connection 分割 Decoder + 中间层特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm
from timm.models.vision_transformer import VisionTransformer
from typing import Tuple, Optional, Dict, List
import numpy as np


# ==================== 空检测头 ====================

class EmptyDetectionHead(nn.Module):
    """空分类检测头: 双路设计 (main + auxiliary)，输出该图像是否为"空"(无物体)"""

    def __init__(self, in_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.aux_branch = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.fusion = nn.Linear(2, 1)

    def forward(self, cls_feat, seg_feat_flat):
        main_logit = self.main_branch(cls_feat)
        aux_logit = self.aux_branch(seg_feat_flat)
        combined = torch.cat([torch.sigmoid(main_logit), torch.sigmoid(aux_logit)], dim=-1)
        final_logit = self.fusion(combined)
        return main_logit, final_logit


# ==================== PUP 分割 Decoder (Progressive UPsampling) ====================

class SegmentationDecoderPUP(nn.Module):
    """
    PUP (Progressive UPsampling) 分割 Decoder

    ViT 的中间层特征全都是低分辨率 (14x14) 的强语义特征,
    用 F.interpolate 强行做 U-Net Skip Connection 是无效的。
    改为纯粹的物理上采样: 利用 ConvTranspose2d 让网络学习
    如何把 14x14 逐步还原成 224x224。

    架构: 14x14 → 28x28 → 56x56 → 112x112 → 224x224
    """

    def __init__(self, in_dim: int = 384, num_classes: int = 21):
        super().__init__()
        # 14x14 -> 28x28
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        # 28x28 -> 56x56
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        # 56x56 -> 112x112
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        # 112x112 -> 224x224
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, patch_feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_feat_2d: (B, in_dim, H, W) — ViT 最终层 patch tokens reshape
        Returns:
            seg_logit: (B, num_classes, 4*H, 4*W)  即 224x224
        """
        x = self.up1(patch_feat_2d)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


# ==================== EMA (指数移动平均) ====================

class ModelEMA:
    """模型指数移动平均, 用于提升分割等任务的泛化性能"""

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = 'cpu'):
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow 权重为模型权重的深拷贝 (强制 FP32, 避免 AMP 下 FP16 下溢出)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float().to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """更新 shadow 权重: shadow = decay * shadow + (1-decay) * current"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data.to(self.device).float(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """将 shadow 权重应用到模型 (用于评估/保存)"""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model: nn.Module):
        """恢复原始权重 (评估后切回训练模式)"""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


# ==================== Early Stopping ====================

class EarlyStopping:
    """早停机制: 监控验证指标, 连续 patience 个 epoch 无改善则停止"""

    def __init__(self, patience: int = 15, mode: str = 'max',
                 min_delta: float = 0.001):
        """
        Args:
            patience: 容忍无改善的 epoch 数
            mode:     'max' (越大越好, 如 mIoU/Acc) 或 'min' (越小越好, 如 loss)
            min_delta: 最小改善幅度阈值
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop_flag = False

    def step(self, value: float) -> bool:
        """
        记录当前值, 返回是否应该停止训练
        Returns:
            True 表示应触发 early stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = False
        if self.mode == 'max':
            if value > self.best_value + self.min_delta:
                improved = True
        else:
            if value < self.best_value - self.min_delta:
                improved = True

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            print(f"    EarlyStopping: no improvement for {self.counter}/{self.patience} "
                  f"(best={self.best_value:.4f}, current={value:.4f})")

        if self.counter >= self.patience:
            self.should_stop_flag = True
            print(f"\n  *** Early Stopping triggered! No improvement for {self.patience} epochs ***\n")
            return True
        return False

    @property
    def should_stop(self) -> bool:
        return self.should_stop_flag


# ==================== 多任务主模型 ====================

class MultiTaskViTSmall(nn.Module):
    """
    多任务 ViT-Small 模型 v2 (Improved)
    =====================================
    骨干网络: ViT-Small (embed_dim=384, depth=12, heads=6)
    预训练:   MAE on ImageNet-1k (官方权重)

    任务头:
      - single_head:   单分类 (21 类, 含背景/空类)
      - multi_head:    多标签分类 (20 类物体)
      - empty_head:    空检测 (二分类: 有/无物体)
      - seg_decoder:   U-Net 风格语义分割 (21 类像素级, 带 skip connection)

    v2 变更:
      - 分割 Decoder 升级为 SegmentationDecoderV2 (U-Net skip connections)
      - extract_features 返回中间层特征用于 skip connections
      - 支持 intermediate_layer_indices 配置
    """

    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, num_classes: int = 20,
                 pretrained_backbone: Optional[str] = None,
                 drop_path_rate: float = 0.1,
                 timm_model_name: str = 'vit_small_patch16_224'):
        """
        Args:
            num_classes:          物体类别数 (Pascal VOC = 20)
            pretrained_backbone:  MAE 预训练权重路径
            drop_path_rate:       DropPath 正则化概率
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = 384
        self.total_classes = num_classes + 1

        # ========== 1. 骨干网络 ==========
        if pretrained_backbone == 'imagenet':
            print(f"Loading ImageNet pretrained backbone from timm: {timm_model_name}")
            self.backbone = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0,
                drop_path_rate=drop_path_rate,
                dynamic_img_size=True,
            )
        else:
            self.backbone = VisionTransformer(
                img_size=224, patch_size=16, in_chans=3,
                num_classes=0, embed_dim=self.embed_dim,
                depth=12, num_heads=6, mlp_ratio=4.,
                qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=drop_path_rate,
            )

            if pretrained_backbone is not None:
                self.load_mae_weights(pretrained_backbone)

        # 暴力解除 patch_embed 的固定输入尺寸限制，以支持高分辨率输入 (如 512x512)
        self.backbone.patch_embed.img_size = None
        if hasattr(self.backbone.patch_embed, 'strict_img_size'):
            self.backbone.patch_embed.strict_img_size = False

        # ========== 2. 分类特征提取器 ==========
        self.cls_feature_extractor = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )

        # ========== 3. 任务头 ==========
        self.single_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, self.total_classes)
        )
        self.multi_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.empty_head = EmptyDetectionHead(in_dim=256, hidden_dim=128)

        # ========== 4. PUP 分割 Decoder (Progressive UPsampling) ==========
        self.seg_decoder = SegmentationDecoderPUP(
            in_dim=self.embed_dim,
            num_classes=self.total_classes
        )

        self._initialize_heads()

    def _initialize_heads(self):
        """初始化所有任务头: 分类头用 Xavier, 分割头用 Kaiming"""
        for module in [self.single_head, self.multi_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # 初始化分割头
        for m in self.seg_decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def load_mae_weights(self, ckpt_path: str):
        """加载 MAE 官方预训练权重"""
        print(f"Loading MAE pretrained weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint

        cleaned = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ['backbone.', 'module.backbone.', 'module.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            if any(s in new_key for s in ['decoder', 'mask_token', 'norm_final']):
                continue
            cleaned[new_key] = v

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print("  MAE weights loaded successfully!")

    def _get_pos_embed(self, h_patch: int, w_patch: int) -> torch.Tensor:
        """Return absolute position embeddings, interpolated for non-224 inputs."""
        pos_embed = self.backbone.pos_embed
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        old_tokens = patch_pos.shape[1]
        old_size = int(old_tokens ** 0.5)

        if old_size * old_size != old_tokens:
            raise ValueError(f"Cannot reshape position embedding with {old_tokens} patch tokens")

        if old_size == h_patch and old_size == w_patch:
            return pos_embed

        patch_pos = patch_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(h_patch, w_patch),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h_patch * w_patch, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def extract_intermediate_features(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        提取骨干网络特征

        Returns:
            cls_feat:      (B, 256) 分类特征
            final_patches: (B, N, D) 最终层 patch tokens
            patch_hw:      (h_patch, w_patch) patch 网格尺寸
        """
        B, _, h, w = x.shape

        # padding 到 16 的倍数
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        h_patch, w_patch = (h + pad_h) // 16, (w + pad_w) // 16

        # 手动逐层前向
        x_patch = self.backbone.patch_embed(x)
        # dynamic_img_size=True 时返回 (B, H, W, D)，需要 flatten 成 (B, N, D)
        if x_patch.dim() == 4:
            B2, Hp, Wp, D = x_patch.shape
            x_patch = x_patch.reshape(B2, Hp * Wp, D)
        B2, N, D = x_patch.shape

        cls_token = self.backbone.cls_token.expand(B2, -1, -1)  # (B, 1, D)
        x_full = torch.cat((cls_token, x_patch), dim=1)           # (B, N+1, D)
        x_full = x_full + self._get_pos_embed(h_patch, w_patch)
        x_full = self.backbone.pos_drop(x_full)

        # 直接过所有 transformer blocks (不再收集中间层)
        for block in self.backbone.blocks:
            x_full = block(x_full)

        # 最终 norm
        x_full = self.backbone.norm(x_full)

        # 最终 patch tokens 和 CLS
        final_tokens = x_full[:, 1:, :]   # (B, N, D)
        pooled = final_tokens.mean(dim=1) # (B, D)
        cls_feat = self.cls_feature_extractor(pooled)  # (B, 256)

        return cls_feat, final_tokens, (h_patch, w_patch)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """兼容旧接口的快速特征提取"""
        cls_feat, final_tokens, _ = self.extract_intermediate_features(x)
        return cls_feat, final_tokens

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        B, _, h, w = x.shape

        # 特征提取
        cls_feat, patch_tokens, patch_hw = self.extract_intermediate_features(x)
        _, N, D = patch_tokens.shape

        # ---- 分类分支 ----
        single_logit = self.single_head(cls_feat)
        multi_logit = self.multi_head(cls_feat)

        # ---- 空检测分支 ----
        seg_pooled = patch_tokens.mean(dim=1)
        seg_pooled = self.cls_feature_extractor[0](seg_pooled)
        empty_main_logit, empty_prob = self.empty_head(cls_feat, seg_pooled)

        # ---- 分割分支 (PUP: Progressive UPsampling) ----
        H_patch, W_patch = patch_hw
        patch_feat_2d = patch_tokens.transpose(1, 2).reshape(B, D, H_patch, W_patch)
        seg_logit = self.seg_decoder(patch_feat_2d)
        # 调整到输入尺寸 (非 224 时 PUP 输出可能不完全匹配)
        if seg_logit.shape[2:] != (h, w):
            seg_logit = F.interpolate(seg_logit, size=(h, w), mode='bilinear', align_corners=False)

        return {
            'single_logit': single_logit,
            'multi_logit': multi_logit,
            'empty_main_logit': empty_main_logit,
            'empty_prob': empty_prob,
            'seg_logit': seg_logit
        }

    # ==================== 高分辨率推理 ====================

    @torch.no_grad()
    def inference_high_res(self, image, base_size=224, stride_ratio=0.5,
                           scales=None, use_amp=True, seg_refine=True):
        """高分辨率滑动窗口推理 (与 v1 接口一致)"""
        if scales is None:
            scales = [1.0]

        device = image.device
        B, C, orig_h, orig_w = image.shape
        assert B == 1

        all_seg_logits, all_cls_single, all_cls_multi, all_empty = [], [], [], []

        for scale in scales:
            if scale != 1.0:
                sh, sw = int(orig_h * scale), int(orig_w * scale)
                img_s = F.interpolate(image, size=(sh, sw), mode='bilinear', align_corners=False)
            else:
                img_s, sh, sw = image, orig_h, orig_w

            window_results = self._sliding_window_inference(
                img_s, base_size, int(base_size * stride_ratio), device, use_amp)
            all_cls_single.append(self._aggregate_single_logits(window_results['single_logits']))
            all_cls_multi.append(self._topk_mean_logits(window_results['multi_logits'], topk_ratio=0.25))

            # Empty logit is inverted to objectness before aggregation so small objects
            # are not washed out by many background windows in high-resolution images.
            objectness_logits = -window_results['empty_logits']
            all_empty.append(-self._topk_mean_logits(objectness_logits, topk_ratio=0.25))

            fused = window_results['seg_fused']
            if scale != 1.0 or seg_refine:
                fused = F.interpolate(fused, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            all_seg_logits.append(fused)

        single_logit = torch.stack(all_cls_single).mean(0)
        multi_logit = torch.stack(all_cls_multi).mean(0)
        empty_avg = torch.stack(all_empty).mean(0)
        seg_avg = torch.stack(all_seg_logits).mean(0)
        seg_raw = seg_avg.argmax(1).squeeze(0).cpu()

        if seg_refine:
            seg_mask = self._refine_segmentation(seg_raw.numpy().astype(np.uint8), orig_h, orig_w)
        else:
            seg_mask = seg_raw.numpy().astype(np.uint8)

        return {
            'single_logit': single_logit, 'multi_logit': multi_logit,
            'empty_main_logit': empty_avg, 'empty_prob': empty_avg,
            'seg_mask': seg_mask, 'seg_logit_full': seg_avg,
        }

    @staticmethod
    def _topk_mean_logits(logits: torch.Tensor, topk_ratio: float = 0.25) -> torch.Tensor:
        """Aggregate window logits with top-k mean along the window dimension."""
        if logits.dim() != 2:
            raise ValueError(f"Expected window logits with shape (N, C), got {tuple(logits.shape)}")
        num_windows = logits.shape[0]
        k = max(1, min(num_windows, int(np.ceil(num_windows * topk_ratio))))
        values = logits.topk(k, dim=0).values
        return values.mean(dim=0, keepdim=True)

    @staticmethod
    def _aggregate_single_logits(logits: torch.Tensor, topk_ratio: float = 0.25) -> torch.Tensor:
        """Confidence-weighted aggregation for single-label window logits."""
        if logits.dim() != 2:
            raise ValueError(f"Expected window logits with shape (N, C), got {tuple(logits.shape)}")
        num_windows = logits.shape[0]
        k = max(1, min(num_windows, int(np.ceil(num_windows * topk_ratio))))
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        top_idx = confidence.topk(k, dim=0).indices
        selected_logits = logits.index_select(0, top_idx)
        selected_conf = confidence.index_select(0, top_idx).unsqueeze(-1)
        weights = selected_conf / selected_conf.sum(dim=0, keepdim=True).clamp(min=1e-6)
        return (selected_logits * weights).sum(dim=0, keepdim=True)

    def _sliding_window_inference(self, image, base_size, stride, device, use_amp):
        """滑动窗口推理核心"""
        B, C, h, w = image.shape
        nc = self.total_classes
        y_starts = list(range(0, max(h - base_size, 0) + 1, stride))
        x_starts = list(range(0, max(w - base_size, 0) + 1, stride))
        if not y_starts or y_starts[-1] != max(h - base_size, 0):
            y_starts.append(max(h - base_size, 0))
        if not x_starts or x_starts[-1] != max(w - base_size, 0):
            x_starts.append(max(w - base_size, 0))

        seg_acc = torch.zeros(B, nc, h, w, device=device, dtype=torch.float32)
        weight_map = torch.zeros(1, 1, h, w, device=device, dtype=torch.float32)
        cls_s, cls_m, cls_e = [], [], []
        gaussian = self._create_gaussian_weight(base_size, sigma=base_size // 8)

        for ys in y_starts:
            for xs in x_starts:
                ye, xe = min(ys + base_size, h), min(xs + base_size, w)
                window = image[:, :, ys:ye, xs:xe]
                ph, pw = base_size - (ye - ys), base_size - (xe - xs)
                if ph > 0 or pw > 0:
                    window = F.pad(window, (0, pw, 0, ph), mode='reflect')

                with autocast(enabled=use_amp and device == 'cuda'):
                    out = self.forward(window)

                cls_s.append(out['single_logit'])
                cls_m.append(out['multi_logit'])
                cls_e.append(out['empty_main_logit'])

                ah, aw = ye - ys, xe - xs
                gw = gaussian[:, :, :ah, :aw].to(device)
                seg_acc[:, :, ys:ye, xs:xe] += out['seg_logit'][:, :, :ah, :aw] * gw
                weight_map[:, :, ys:ye, xs:xe] += gw

        seg_fused = seg_acc / weight_map.clamp(min=1e-6)
        return {'seg_fused': seg_fused,
                'single_logits': torch.cat(cls_s, 0),
                'multi_logits': torch.cat(cls_m, 0),
                'empty_logits': torch.cat(cls_e, 0)}

    @staticmethod
    def _create_gaussian_weight(size, sigma):
        coord = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-(coord ** 2) / (2 * sigma ** 2))
        w2d = g.unsqueeze(1) * g.unsqueeze(0)
        return w2d.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _refine_segmentation(mask, h, w, min_area_ratio=0.001, kernel_sizes=None):
        """分割掩码后处理: 形态学 + 连通域过滤"""
        try:
            from scipy import ndimage
        except ImportError:
            print("Warning: scipy not found, skipping segmentation refinement")
            return mask

        if kernel_sizes is None:
            kernel_sizes = [3, 5]

        refined = mask.copy()
        unique_classes = np.unique(refined)
        total_pixels = h * w
        min_area = int(total_pixels * min_area_ratio)

        for cls_id in unique_classes:
            if cls_id == 0:
                continue
            binary = (refined == cls_id).astype(np.uint8)
            for ks in kernel_sizes:
                binary = ndimage.binary_closing(binary, structure=np.ones((ks, ks), dtype=np.uint8))
            binary = ndimage.binary_opening(binary, structure=np.ones((3, 3), dtype=np.uint8))

            labeled, n = ndimage.label(binary)
            if n > 0:
                sizes = ndimage.sum(binary, labeled, range(n + 1))
                large = np.isin(labeled, np.where(sizes >= min_area)[0])
                refined[refined == cls_id] = 0
                refined[large.astype(np.uint8) > 0] = cls_id

        return refined


def build_model(num_classes: int = 20,
                pretrained_path: Optional[str] = None,
                timm_model_name: str = 'vit_small_patch16_224',
                device: str = 'cuda') -> MultiTaskViTSmall:
    """构建并返回模型的便捷函数"""
    model = MultiTaskViTSmall(
        num_classes=num_classes,
        pretrained_backbone=pretrained_path,
        timm_model_name=timm_model_name,
        drop_path_rate=0.1
    )
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built. Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  Frozen: {name}")
    return model


if __name__ == '__main__':
    model = build_model(pretrained_path=None, device='cpu')
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
