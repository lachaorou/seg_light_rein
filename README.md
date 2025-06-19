# Seg Light Rein - 轻量化语义分割实验平台

## 项目概述

这是一个专为高精度/轻量化兼顾的语义分割研究而设计的模块化实验平台。项目基于PyTorch构建，支持多种主干网络、创新机制（如Rein）、分割头的插拔式组合，并提供完整的训练、评估、对比实验流程。

## 🌟 主要特性

### 🔧 模块化设计
- **主干网络**: MobileNetV3 (Small/Large)，可轻松扩展更多轻量网络
- **创新机制**: Rein自适应激活机制，支持可插拔集成
- **分割头**: DeepLabV3/DeepLabV3+，支持多尺度特征融合
- **统一接口**: 配置化模型构建，支持任意组合

### � 训练系统
- **多种损失函数**: CrossEntropy、Focal、Dice、Lovasz等
- **灵活优化器**: SGD、Adam、AdamW，支持自定义学习率调度
- **混合精度训练**: 自动混合精度加速训练
- **自动保存**: 最佳模型、训练历史、实验配置自动归档

### 📊 实验管理
- **配置系统**: YAML/JSON配置文件管理实验参数
- **自动对比**: 多实验结果自动生成对比表
- **指标追踪**: mIoU、准确率、参数量、FLOPs等全面指标
- **可视化**: 训练曲线、分割结果可视化

### 🎯 多数据集支持
- **VOC2012**: 经典21类语义分割，适合快速验证
- **ADE20K**: 150类室内外场景，适合全面评估
- **Cityscapes**: 19类街景分割，适合高精度测试
- **统一接口**: 一键切换数据集，自动适配类别数

### 🚀 实验流程
- **渐进式验证**: VOC2012(快速)→ADE20K(全面)→Cityscapes(精度)
- **批量实验**: 自动化多数据集实验和性能对比
- **结果归档**: 实验结果自动保存和对比表生成

### 🧠 创新机制
- 🚀 **轻量化导向**: 重点研究MobileNetV3、PIDNet、MobileViT等轻量主干
- 🧠 **创新机制**: 集成Rein、Token Merging、注意力等前沿机制
- 📈 **实验管理**: 自动化实验配置、结果记录、对比表生成

## 目录结构

```
seg_light_rein/
├── configs/           # 实验配置文件
├── models/
│   ├── backbones/     # 主干网络
│   ├── rein/          # Rein机制及创新模块
│   ├── heads/         # 分割头
│   └── aspp.py        # 特征增强模块
├── datasets/          # 数据集处理
├── scripts/           # 训练、评估、可视化脚本
├── utils/             # 工具函数
├── results/           # 实验结果
└── Documents/         # 文档与实验记录
```

## 快速开始

### 1. 环境配置
```bash
# TODO: 补充环境安装命令
```

### 2. 基线实验
```bash
# MobileNetV3基线
python scripts/train.py --config configs/mobilenetv3_baseline.yaml

# MobileNetV3 + Rein
python scripts/train.py --config configs/mobilenetv3_rein.yaml
```

### 3. 查看实验结果
```bash
python scripts/generate_comparison_table.py --results_dir results/
```

### 4. 多数据集批量实验

运行渐进式多数据集实验：

```bash
# 运行完整的多数据集批量实验（VOC2012→ADE20K→Cityscapes）
python run_batch_experiments.py

# 实验完成后查看结果报告
ls experiments/batch_experiment_report/
```

### 5. 单独运行特定数据集实验

```bash
# VOC2012快速验证
python train_complete.py --config configs/voc2012_quick_test.yaml

# ADE20K全面评估
python train_complete.py --config configs/ade20k_full_eval.yaml

# Cityscapes高精度测试
python train_complete.py --config configs/cityscapes_precision_test.yaml
```

## 数据集配置

### 支持的数据集格式

| 数据集         | 类别数 | 适用场景   | 推荐用途     |
| -------------- | ------ | ---------- | ------------ |
| **VOC2012**    | 21     | 快速验证   | 算法原型测试 |
| **ADE20K**     | 150    | 全面评估   | 泛化能力验证 |
| **Cityscapes** | 19     | 高精度测试 | 最终性能评估 |

### 数据集准备

```bash
# 创建数据集目录
mkdir -p datasets/{VOC2012,ADE20K,Cityscapes}

# VOC2012格式示例
VOC2012/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/Segmentation/

# ADE20K格式示例
ADE20K/
├── images/{training,validation}/
└── annotations/{training,validation}/

# Cityscapes格式示例
Cityscapes/
├── leftImg8bit/{train,val,test}/
└── gtFine/{train,val,test}/
```

详细格式说明参见 [Documents/数据集格式对比分析.md](Documents/数据集格式对比分析.md)

## 实验流程建议

### 🚀 渐进式实验策略

**阶段1: 快速验证（1-2小时）**
- 数据集：VOC2012
- 分辨率：256×256
- 目标：验证模型架构正确性

**阶段2: 全面评估（4-8小时）**
- 数据集：ADE20K
- 分辨率：512×512
- 目标：验证泛化能力和多类别性能

**阶段3: 精度测试（8-16小时）**
- 数据集：Cityscapes
- 分辨率：1024×512
- 目标：追求最高精度

## 支持的模块

### 主干网络 (Backbones)
- [x] MobileNetV3
- [ ] PIDNet
- [ ] MobileViT
- [ ] EfficientNet-Lite
- [ ] GhostNet

### 创新机制 (Rein)
- [x] Rein机制
- [ ] Token Merging
- [ ] Channel/Spatial Attention
- [ ] Dynamic Head

### 分割头 (Heads)
- [x] FCN Head
- [ ] OCR Head
- [ ] UPerHead
- [ ] Mask2Former Head

### 特征增强 (Enhancement)
- [x] ASPP
- [x] Lightweight ASPP
- [x] CBAM
- [ ] Transformer Block

## 实验规划

### 阶段1: 基线建立
- [ ] 主流轻量主干基线实验
- [ ] baseline对比表建立
- [ ] 评估指标体系完善

### 阶段2: Rein机制研究
- [ ] 单点/多点插入对比
- [ ] 不同参数配置消融
- [ ] 与其它机制组合实验

### 阶段3: 创新点归纳
- [ ] 系统性消融实验
- [ ] 性能提升分析
- [ ] 论文创新点整理

## 实验记录

详见 [Documents/baseline与对比表.md](Documents/baseline与对比表.md)

## 贡献指南

1. 遵循模块化设计原则
2. 添加新模块时需更新配置系统
3. 实验结果需记录于对比表
4. 代码需包含完整注释和文档

## 参考文献

- Rein机制: arXiv:2312.04265
- DeepLabV3+: ECCV 2018
- MobileNetV3: ICCV 2019
- PIDNet: arXiv:2206.02066

---

*Last updated: 2025-06-19*
