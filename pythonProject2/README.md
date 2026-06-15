# Multi-Task ViT-Small on Pascal VOC 2012

基于 **ViT-Small** 骨干网络，在 **Pascal VOC 2012** 数据集上微调，实现**四任务联合学习**。
骨干预训练支持 **timm ImageNet** (默认) 或 **MAE (Masked Autoencoder)**：

| 任务 | 输出 | 说明 |
|------|------|------|
| **单分类** | 21 类 Softmax | 检测图像中主要物体类别（20 类物体 + 背景/空） |
| **多分类** | 20 类 Sigmoid | 检测图像中所有出现的物体 |
| **空检测** | 二分类 | 判断图像是否包含任何物体（无目标检测） |
| **分割** | 21 类像素级 | 语义分割，输出逐像素类别标签 |

> **版本说明：** `pythonProject2\pythonProject2` 下的四个 `.py` 文件（`model.py`, `train.py`, `losses.py`, `test.py`）是**初步完成了 future work 的版本，尚未经过完整测试**。
> 
> `checkpoints/` 目录中的模型权重 与 `python_checkpoints/` 目录中的程序**是匹配的**（均为未完成 future work 前的版本），**可以配套使用**。
> 
> 当前 `pythonProject2\pythonProject2` 下的新版代码（v4 Query-Based Top-Down）与 `checkpoints/` 中的旧权重**架构不兼容，无法直接加载**。

## 项目结构

```
pythonProject2/
├── pythonProject2/
│   ├── model.py              # ViT-Small 多任务模型定义 (v4 Query-Based Top-Down, 未测试)
│   ├── train.py              # 训练脚本 (v3 Copy-Paste + Query-Based, 未测试)
│   ├── losses.py             # 损失函数定义 (CE + Dice + Boundary Dice, ASL)
│   ├── test.py               # 测试/推理脚本 (v4 兼容, 未测试)
│   ├── README.md             # 本文件
│   ├── requirements.txt      # Python 依赖
│   ├── background_patches/   # 空分类训练样本 (从 VOC 背景截取)
│   ├── checkpoints/          # 旧版模型权重 (v3 之前, 与 python_checkpoints/ 中的旧程序匹配, 可配套使用)
│   ├── python_checkpoints/   # 旧版程序备份 (未完成 future work 前的版本, 与 checkpoints/ 中的旧权重匹配, 可配套使用)
│   └── test_pic/             # 测试图片样例
├── data/
│   └── VOCdevkit/VOC2012/    # Pascal VOC 2012 数据集
├── results/                  # 推理结果保存目录
├── mae_pretrain_vit_small.pth# MAE 预训练权重 (官方, ~60 MB, 可选)
└── train_log.txt             # 训练日志
```

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 关键依赖版本建议:
#   torch>=2.0.0, torchvision>=0.15.0
#   timm>=0.9.0
#   albumentations>=1.3.0
#   scipy>=1.7.0
```

## 预训练权重说明

默认情况下，模型通过 **timm** 自动加载 **ImageNet-1k 监督预训练权重**（无需手动下载）。如果你想使用 **MAE (Masked Autoencoder)** 自监督预训练权重，可按以下步骤手动下载并指定路径。

### 下载 MAE 权重 (可选)

从 Facebook Research 的 MAE 官方 GitHub 下载:

1. 访问: https://github.com/facebookresearch/mae
2. 找到 **ViT-Small** 权重下载链接:
   ```
   mae_pretrain_vit_small.pth  (~60 MB)
   ```
3. 将权重文件放到项目根目录:
   ```
   pythonProject2/
   └── mae_pretrain_vit_small.pth
   ```

### 通过命令行快速下载 MAE 权重

```bash
# 在项目目录下执行:
curl -o mae_pretrain_vit_small.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_small.pth

# 或使用 wget:
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_small.pth
```

```bash
# 在项目目录下执行:
curl -o mae_pretrain_vit_small.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_small.pth

# 或使用 wget:
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_small.pth
```

> **注意**: 默认情况下模型会自动通过 timm 加载 ImageNet 预训练权重，无需额外下载。若指定了 MAE 权重路径则加载 MAE 权重；若关闭预训练（`--pretrained None`）则随机初始化。
> **注意**: `checkpoints/` 中的旧权重与 `python_checkpoints/` 中的旧程序**相互匹配**，可以配套使用。当前 v4 代码与旧权重不兼容，请勿混用。

## 使用方法

### 1. 训练

```bash
# 基础训练 (使用默认参数)
python train.py

# 自定义参数
python train.py --epochs 100 --batch_size 8 --backbone_lr 5e-5

# 指定 MAE 预训练权重
python train.py --pretrained mae_pretrain_vit_small.pth

# 显存不足时减小 batch size 并关闭 AMP
python train.py --batch_size 4 --no_amp
```

**默认训练配置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 总训练轮数 |
| `--batch_size` | 8 | 批次大小 |
| `--backbone_lr` | 1e-4 | 骨干网络学习率 |
| `--head_lr` | 1e-3 | 任务头学习率 |
| `--seg_lr` | 1e-3 | 分割 Decoder 学习率 |
| `--input_size` | 512 | 输入图像尺寸 |
| `--num_bg_patches` | 800 | 背景块提取数量 |
| `--pretrained` | 'imagenet' | 预训练来源 ('imagenet' 或 MAE 路径) |
| `--copy_paste_prob` | 0.5 | Copy-Paste 数据增强概率 (不可命令行调节, 在代码中修改) |

**训练流程自动处理:**
1. 自动检测 / 下载 VOC 2012 数据集
2. 从 VOC 训练集截取背景块作为空分类样本
3. Copy-Paste 数据增强: 50% 概率将随机 Donor 图的前景像素粘贴到当前图
4. Linear Warmup (5 epochs) + Cosine Annealing 学习率调度
5. AMP 混合精度加速
6. EMA 指数移动平均 (默认开启)
7. Early Stopping (patience=15, 监控 val mIoU)
8. 每 5 个 epoch 验证一次，自动保存最佳模型

### 2. 测试 & 推理

```bash
# 对单张图片推理
python test.py --image path/to/image.jpg --vis

# 批量测试 test_pic 目录中的图片
python test.py --vis

# 在验证集上评估 (计算 mIoU、准确率等指标)
python test.py --eval --checkpoint checkpoints/best_model.pth
```

**测试输出示例:**
```
  Single Classification: dog (class ID: 7)
  Multi-label Classes:   ['dog', 'person'] ([7, 15])
  Empty Probability:    0.9521 -> NOT Empty
  Segmentation Shape:    (480, 640)
```

## 模型架构详解 (v4 Query-Based Top-Down)

### 骨干网络: ViT-Small (timm ImageNet / MAE 预训练)

```
Input Image (224×224×3)
        ↓
Patch Embedding (patch_size=16) → 196 patches
        ↓
ViT Encoder (12 layers, embed_dim=384, heads=6)
  ├── [timm ImageNet 预训练] ← 默认 (ImageNet-1k 监督学习)
  └── [MAE 预训练权重] ← 可选 (ImageNet-1k 自监督学习)
  └── Stochastic Depth (drop_path=0.1)
        ↓
Features: CLS token + Patch tokens (197 × 384)
```

### 任务头 (v4 架构)

```
Patch tokens (B, N, 384)
        ↓
┌─────────────────────────────────────┐
│   ClassQueryDecoder (Cross-Attention) │
│   - 21 个可学习 Query (B, 21, 256)   │
│   - Query  attend to  Patch tokens   │
│   - 输出: updated_queries (B,21,256) │
│            class_logits (B, 21)      │
└─────────────────────────────────────┘
        ↓
  single_logit = class_logits          (B, 21)   单分类
  multi_logit  = class_logits[:, 1:] (B, 20)   多分类
        ↓
PixelDecoder (FPN) ───────────────────────────────────────┐
  - 提取 ViT 层 3/6/9/12 → lateral + top-down 融合        │
  - 输出 pixel_embeds (B, 256, H, W)                      │
        ↓                                                  │
  seg_logit = einsum('b c d, b d h w -> b c h w',         │
                     updated_queries, pixel_embeds)          │
  动态掩码生成 (B, 21, H, W)                               │
        ↓                                                  │
EmptyDetectionHead (main + auxiliary)                     │
  - empty_main_logit, empty_prob (B, 1)                   │
```

**v4 关键改进:**
- **Query-Based 分类**: 用 21 个可学习 Query 通过 Cross-Attention 提取类别特征，取代独立分类头
- **PixelDecoder**: 纯粹输出 256-dim 像素嵌入，不做分类
- **动态掩码生成**: `torch.einsum` 将 Query 与 Pixel Embedding 点乘生成分割结果
- **FPN 尺寸对齐保险**: 在 Top-Down 融合中加入 `F.interpolate`，防止奇数分辨率输入导致尺寸不匹配崩溃

**各模块参数量估算:**
| 模块 | 参数量 |
|------|--------|
| ViT-Small Backbone | ~22M |
| ClassQueryDecoder | ~0.3M |
| PixelDecoder (FPN) | ~2M |
| EmptyDetectionHead | ~0.1M |
| **总计** | **~24.4M** |

## 损失函数

### 分割损失: CE + Dice + Boundary Dice

```python
L_seg = L_CE + 1.0 * L_Dice + 0.5 * L_BoundaryDice
```

- **CE Loss**: 处理 21 类分类
- **Dice Loss**: 直接优化前景/背景重叠度，对抗类别不平衡
- **Boundary Dice Loss**: 对边界掩码计算 Dice，强迫网络关注边缘轮廓精度

### 多标签损失: AsymmetricLoss (ASL)

```python
L_multi = ASL(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
```

- 对负样本施加更高权重，提升正样本 Recall，对抗多标签极度不平衡

### 联合损失 (固定权重 / Uncertainty Weighting)

```python
# 固定权重模式 (默认)
L_total = 0.1*L_single + 1.0*L_multi + 0.1*L_empty + 5.0*L_seg

# Uncertainty Weighting (--uw 启用)
L_total = Σ (1/(2*σ_i²) * L_i + log(σ_i)) + λ * Σ log(σ_i)²
```

## 空分类数据生成策略

本项目采用**真实背景裁剪**而非合成数据来训练空分类能力:

```
VOC 2012 训练图像
        ↓
读取对应的 Segmentation mask
        ↓
定位背景区域 (mask == 0 或 255)
        ↓
随机采样 224×224 的纯背景块
        ↓
保存为独立图像 → background_patches/bg_XXXXX.jpg
```

**优势:**
- 样本来自真实自然图像，分布与 VOC 一致
- 相比合成纯色/棋盘格，domain gap 更小
- 模型学到的是"无目标"的真实语义特征

## 训练技巧

### 1. 分层学习率
```python
Backbone (ViT):  lr = 1e-4    # 低学习率, 保护预训练特征
Heads (all):     lr = 1e-3    # 高学习率, 快速适应新任务
SegDecoder:      lr = 1e-3    # 分割 Decoder 学习率
```

### 2. Copy-Paste 数据增强
```python
# 在 __getitem__ 中，50% 概率触发
# 1. 随机选取 Donor 图
# 2. Resize Donor 到当前图尺寸 (mask 用 NEAREST 防止非法类别)
# 3. 将 Donor 的前景像素 (1~20类) 直接覆盖到当前图
```
- 打破物体的背景共现偏差 (如椅子总在桌子旁)
- 无需复杂边缘融合，直接像素覆盖即可提升 mIoU

### 3. 学习率调度
```
Epoch:  |----warmup----|----------cosine annealing----------|
LR:          ↑ linear         ↓ cosine → min_lr (1% of peak)
```

### 4. 混合精度训练 (AMP)
- 使用 FP16 加速前向/反向传播
- 显存占用减少约 50%
- 训练速度提升约 30-40%

### 5. EMA (指数移动平均)
- 评估时应用 EMA 权重，mIoU 可提升 0.5~1.5%
- 默认开启，衰减率 0.9999

## 预期性能指标

> **注意**: 以下为旧版 (v3 之前) 在 Pascal VOC 2012 validation set 上的参考表现。当前 v4 代码尚未经过完整测试，实际性能待验证。

| 指标 | 参考值 (v3-) |
|------|--------------|
| 单分类准确率 | ~92-96% |
| 多标签 mF1 | ~80-88% |
| 空检测准确率 | ~90-95% |
| 分割 mIoU | ~65-75% |

> 实际性能取决于: 是否使用预训练权重、训练 epochs 数量、超参数调优

## 常见问题

### Q: 显存不够怎么办?
A: 减小 `--batch_size`, 或添加 `--no_amp` 关闭混合精度后进一步降低 batch_size 到 4 或 2。

### Q: 训练速度很慢?
A: 确保 AMP 开启 (默认开启), 增加 `num_workers=4` (默认), 或减小 input_size。

### Q: mIoU 不理想?
A: 1) 确保预训练权重已正确加载（默认 timm ImageNet）；2) 增加训练轮数到 100-150；3) 调整分割损失权重 `LOSS_SEG_W`。

### Q: checkpoints/ 中的旧权重能加载吗?
A: **不能**直接加载到当前 v4 代码。但 `checkpoints/` 中的旧权重与 `python_checkpoints/` 中的旧程序**是匹配的，可以配套使用**。如需使用旧权重，请将 `python_checkpoints/` 中的对应旧程序拷贝到当前目录并运行。

### Q: 如何在新数据集上使用?
A: 修改 `train.py` 中的 `VOC_ROOT` 和数据集类, 或继承 `VOCMultiTaskDataset` 自定义你的 Dataset。注意调整 `num_classes` 和类别名列表。
