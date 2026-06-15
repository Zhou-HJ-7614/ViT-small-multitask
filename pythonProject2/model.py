"""
Multi-Task ViT-Small 模型 (v4 - Query-Based Top-Down)
=======================================================
基于 MAE 预训练的 ViT-Small 骨干网络，实现四个任务：
  1. 单分类 (Single-class): Softmax, 21类 (20物体 + 1空/背景)
  2. 多分类 (Multi-label): Sigmoid, 20类物体
  3. 空分类 (Empty Detection): 二分类，判断图像是否包含任何物体
  4. 分割 (Segmentation): 像素级 21 类语义分割

预训练: MAE (Masked Autoencoder) on ImageNet-1k
高分辨率支持: 滑动窗口推理 + 多尺度融合
v4 变更 (Query-Based Top-Down):
  - ClassQueryDecoder: 21 个可学习 Query 通过 Cross-Attention 提取类别特征
  - PixelDecoder (FPN): 提取 ViT 第 3/6/9/12 层特征，输出 256-dim 像素嵌入
  - 动态掩码生成: torch.einsum('b c d, b d h w -> b c h w') 结合 Query 与 Pixel Embedding
  - 移除独立的 single_head/multi_head，分类与分割共享语义
  - FPN Top-Down 尺寸对齐保险: 双线性插值防止奇数分辨率崩溃
  - Boundary Dice Loss: 强化物体边缘轮廓精度
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


# ==================== Query 路由模块 (新增) ====================

class ClassQueryDecoder(nn.Module):
    """
    类别查询解码器 (Top-Down 核心)
    利用 Cross-Attention 让 21 个类别的 Query 去图像中提取专属特征
    """
    def __init__(self, in_dim=384, query_dim=256, num_classes=21):
        super().__init__()
        self.num_classes = num_classes

        # 1. 可学习的类别查询向量 (Class Queries): (21, 256)
        self.query_embed = nn.Parameter(torch.randn(num_classes, query_dim))

        # 2. 维度投影对齐
        self.proj_k = nn.Linear(in_dim, query_dim)
        self.proj_v = nn.Linear(in_dim, query_dim)

        # 3. 交叉注意力机制 (Queries 找图像)
        self.cross_attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim)
        )

        # 4. 基于更新后的 Query 预测该类别是否存在
        self.class_predictor = nn.Linear(query_dim, 1)

    def forward(self, patch_tokens):
        """
        patch_tokens: (B, N, 384) 骨干网络输出的图像 Patch 序列
        """
        B = patch_tokens.shape[0]
        # (B, 21, 256)
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        # Keys 和 Values 是图像特征
        k = self.proj_k(patch_tokens)
        v = self.proj_v(patch_tokens)

        # Cross-Attention: 类别 Queries 提取全图属于自己的特征
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        q = self.norm(q + attn_out)
        q = q + self.mlp(q)  # 更新后的 Queries (B, 21, 256)

        # 分类 Logits (B, 21)
        class_logits = self.class_predictor(q).squeeze(-1)

        return q, class_logits


# ==================== Pixel Decoder (纯粹的像素特征上采样) ====================

class PixelDecoder(nn.Module):
    """
    像素特征解码器 (Query-Based 架构的配套模块)

    提取 ViT 第 3、6、9、12 层特征，构建特征金字塔，
    最终输出高分辨率的像素嵌入 (Pixel Embeddings)，供 Query 点乘生成掩码。

    架构:
      - Lateral (1x1 conv + 物理重构):
        layer 3  -> stride=4  -> 56x56
        layer 6  -> stride=2  -> 28x28
        layer 9  -> stride=1  -> 14x14
        layer 12 -> stride=2  -> 7x7  (塔尖)
      - Top-down: 7x7 -> up -> 14x14 + 14x14 -> up -> 28x28 + 28x28 -> up -> 56x56 + 56x56
      - Final: 56x56 -> stride=4 -> 224x224, 输出 256 维像素嵌入
    """

    def __init__(self, in_dim: int = 384, fpn_dim: int = 256):
        super().__init__()
        # ---- Lateral connections ----
        self.lat3 = nn.Sequential(
            nn.Conv2d(in_dim, fpn_dim, kernel_size=1, bias=False),
            nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)
        )
        self.lat6 = nn.Sequential(
            nn.Conv2d(in_dim, fpn_dim, kernel_size=1, bias=False),
            nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)
        )
        self.lat9 = nn.Sequential(
            nn.Conv2d(in_dim, fpn_dim, kernel_size=1, bias=False),
            nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)
        )
        self.lat12 = nn.Sequential(
            nn.Conv2d(in_dim, fpn_dim, kernel_size=1, bias=False),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)
        )

        # ---- Top-down ----
        self.up12 = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.up9 = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.smooth6 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)
        )
        self.up6 = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.smooth3 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True)
        )

        # ---- Final: 输出 256 维像素嵌入 (不输出类别) ----
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(fpn_dim, 128, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1)
        )

    def forward(self, f3: torch.Tensor, f6: torch.Tensor,
                f9: torch.Tensor, f12: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f3, f6, f9, f12: (B, in_dim, H, W) — 各层 patch tokens reshape
        Returns:
            pixel_embeds: (B, 256, 4*4*H, 4*4*W)  即 224x224 (当 H=W=14)
        """
        p3 = self.lat3(f3)    # (B, fpn_dim, 56, 56)
        p6 = self.lat6(f6)    # (B, fpn_dim, 28, 28)
        p9 = self.lat9(f9)    # (B, fpn_dim, 14, 14)
        p12 = self.lat12(f12) # (B, fpn_dim, 7, 7)

        # --- 修复：Top-down 融合加入尺寸对齐保险 ---
        p12_up = self.up12(p12)
        if p12_up.shape[2:] != p9.shape[2:]:
            p12_up = F.interpolate(p12_up, size=p9.shape[2:], mode='bilinear', align_corners=False)
        p9 = p9 + p12_up

        p9_up = self.up9(p9)
        if p9_up.shape[2:] != p6.shape[2:]:
            p9_up = F.interpolate(p9_up, size=p6.shape[2:], mode='bilinear', align_corners=False)
        p6 = self.smooth6(p6 + p9_up)   # (B, fpn_dim, 28, 28)

        p6_up = self.up6(p6)
        if p6_up.shape[2:] != p3.shape[2:]:
            p6_up = F.interpolate(p6_up, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.smooth3(p3 + p6_up)   # (B, fpn_dim, 56, 56)

        pixel_embeds = self.up_final(p3)  # (B, 256, 224, 224)
        return pixel_embeds


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
    多任务 ViT-Small 模型 v4 (Query-Based Top-Down)
    ================================================
    骨干网络: ViT-Small (embed_dim=384, depth=12, heads=6)
    预训练:   MAE on ImageNet-1k (官方权重)

    任务头:
      - query_decoder: Query-Based 分类 (21 类 Cross-Attention 路由)
      - empty_head:    空检测 (二分类: 有/无物体)
      - pixel_decoder: 像素特征解码器 (输出 256 维像素嵌入)

    v4 变更:
      - 引入 ClassQueryDecoder: 21 个可学习 Query 通过 Cross-Attention 提取类别特征
      - 引入 PixelDecoder: 纯粹输出高分辨率像素嵌入，不做分类
      - 分割掩码由 Query 与 Pixel Embedding 点乘动态生成 (einsum)
      - 分类与分割共享语义，实现真正的 Top-Down 架构
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

        # ========== 3. Query 路由 + 空检测 ==========
        self.query_decoder = ClassQueryDecoder(
            in_dim=self.embed_dim, query_dim=256, num_classes=self.total_classes
        )
        self.empty_head = EmptyDetectionHead(in_dim=256, hidden_dim=128)

        # ========== 4. Pixel Decoder (纯粹的像素特征上采样) ==========
        self.pixel_decoder = PixelDecoder(
            in_dim=self.embed_dim,
            fpn_dim=256
        )

        self._initialize_heads()

    def _initialize_heads(self):
        """初始化所有任务头: Query/Linear 用 Xavier, Conv 用 Kaiming"""
        # Query Decoder 中的 Linear 层用 Xavier
        for m in self.query_decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Empty Head 中的 Linear 层用 Xavier
        for m in self.empty_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Pixel Decoder 中的 Conv 层用 Kaiming
        for m in self.pixel_decoder.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        torch.Tensor, torch.Tensor, Tuple[int, int], Dict[int, torch.Tensor]]:
        """
        提取骨干网络特征（含中间层）

        Returns:
            cls_feat:      (B, 256) 分类特征
            final_patches: (B, N, D) 最终层 patch tokens
            patch_hw:      (h_patch, w_patch) patch 网格尺寸
            intermediate:  dict[int, Tensor] 第 3,6,9,12 层 2D 特征
        """
        B, _, h, w = x.shape

        # padding 到 16 的倍数
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        h_patch, w_patch = (h + pad_h) // 16, (w + pad_w) // 16

        # 手动逐层前向
        x_patch = self.backbone.patch_embed(x)
        # dynamic_img_size=True 时返回 (B, C, H, W)，需要 flatten 成 (B, N, C)
        if x_patch.dim() == 4:
            # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
            x_patch = x_patch.flatten(2).transpose(1, 2)
        B2, N, D = x_patch.shape

        cls_token = self.backbone.cls_token.expand(B2, -1, -1)  # (B, 1, D)
        x_full = torch.cat((cls_token, x_patch), dim=1)           # (B, N+1, D)
        x_full = x_full + self._get_pos_embed(h_patch, w_patch)
        x_full = self.backbone.pos_drop(x_full)

        # 逐层前向并收集第 3、6、9、12 层特征 (0-based index: 2,5,8,11)
        intermediate = {}
        collect_indices = {2, 5, 8, 11}
        for i, block in enumerate(self.backbone.blocks):
            x_full = block(x_full)
            if i in collect_indices:
                feat = x_full[:, 1:, :].transpose(1, 2).reshape(B2, D, h_patch, w_patch)
                intermediate[i] = feat

        # 最终 norm
        x_full = self.backbone.norm(x_full)

        # 最终 patch tokens 和 CLS
        final_tokens = x_full[:, 1:, :]   # (B, N, D)
        pooled = final_tokens.mean(dim=1) # (B, D)
        cls_feat = self.cls_feature_extractor(pooled)  # (B, 256)

        return cls_feat, final_tokens, (h_patch, w_patch), intermediate

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """兼容旧接口的快速特征提取"""
        cls_feat, final_tokens, _, _ = self.extract_intermediate_features(x)
        return cls_feat, final_tokens

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播 (Query-Based Top-Down)"""
        B, _, h, w = x.shape

        # 特征提取（含中间层）
        cls_feat, patch_tokens, patch_hw, intermediate = self.extract_intermediate_features(x)

        # ---- 1. Top-Down 第一步：Query 路由分类 ----
        updated_queries, class_logits = self.query_decoder(patch_tokens)
        single_logit = class_logits                # (B, 21)
        multi_logit = class_logits[:, 1:]          # (B, 20)

        # ---- 2. 空检测分支 ----
        seg_pooled = patch_tokens.mean(dim=1)
        seg_pooled = self.cls_feature_extractor(seg_pooled)
        empty_main_logit, empty_prob = self.empty_head(cls_feat, seg_pooled)

        # ---- 3. Top-Down 第二步：生成像素级画布 ----
        f3 = intermediate[2]
        f6 = intermediate[5]
        f9 = intermediate[8]
        f12 = intermediate[11]
        pixel_embeds = self.pixel_decoder(f3, f6, f9, f12)  # (B, 256, H, W)
        if pixel_embeds.shape[2:] != (h, w):
            pixel_embeds = F.interpolate(pixel_embeds, size=(h, w), mode='bilinear', align_corners=False)

        # ---- 4. Top-Down 终极结合：动态掩码生成 ----
        seg_logit = torch.einsum('b c d, b d h w -> b c h w', updated_queries, pixel_embeds)

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
