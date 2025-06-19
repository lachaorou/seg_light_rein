# seg_light_rein 项目使用指南

## 项目概述

seg_light_rein 是一个专注于**轻量化高精度语义分割**的研究框架，基于模块化设计，支持：

- 🚀 **轻量化主干网络**：MobileNetV3系列等
- 🧠 **创新注意力机制**：Rein自适应激活机制
- 🎯 **多种分割头**：DeepLabV3+、FCN等
- 📊 **完整实验流程**：配置化训练、自动评估、结果对比
- 🔧 **模块化设计**：主干、机制、分割头可插拔组合

## ✅ 系统验证状态

**所有核心功能已通过测试验证：**

1. ✅ **模型构建**：支持多种主干+分割头组合
2. ✅ **数据集处理**：VOC2012支持（含dummy数据测试）
3. ✅ **训练流程**：完整训练、验证、保存功能
4. ✅ **配置系统**：YAML配置文件支持
5. ✅ **实验管理**：自动归档、对比表生成
6. ✅ **Rein机制**：注意力机制集成（小batch问题已知）

## 🚀 快速开始

### 1. 环境配置

```bash
# 确保CUDA环境可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查依赖包
python -c "import torch, torchvision, numpy, tqdm, yaml; print('Dependencies OK')"
```

### 2. 系统测试

```bash
# 运行完整系统测试
python test_system.py

# 预期输出：所有测试通过，只有Rein机制会有小batch警告
```

### 3. 快速实验

```bash
# 运行3轮快速验证实验
python train_complete.py --config configs/quick_test_v2.yaml

# 查看实验结果
ls experiments/quick_test_*/results/
```

## 📁 项目结构

```
seg_light_rein/
├── configs/                    # 实验配置文件
│   ├── mobilenetv3_small_baseline.yaml    # 基线配置
│   ├── mobilenetv3_small_rein.yaml        # Rein机制配置
│   └── quick_test_v2.yaml                 # 快速测试配置
├── models/                     # 模型实现
│   ├── backbones/             # 主干网络
│   │   ├── mobilenetv3.py     # MobileNetV3实现
│   │   └── mobilenetv3_real.py # 真实MobileNetV3
│   ├── heads/                 # 分割头
│   │   ├── deeplabv3plus_head.py  # DeepLabV3+头
│   │   └── fcn_head.py        # FCN头
│   ├── mechanisms/            # 注意力机制
│   │   └── rein_mechanism.py  # Rein机制
│   └── unified_model_builder.py  # 统一模型构建器
├── datasets/                  # 数据集处理
│   └── voc_dataset.py        # VOC2012数据集
├── training/                  # 训练框架
│   └── advanced_trainer.py   # 高级训练器
├── utils/                     # 工具函数
│   ├── metrics.py            # 评估指标
│   ├── logger.py             # 日志系统
│   └── dataset.py            # 数据工具
├── experiments/               # 实验结果（自动生成）
├── train_complete.py         # 完整训练脚本
├── test_system.py           # 系统测试脚本
└── README.md                # 项目说明
```

## 🔧 配置系统

### 基本配置格式

```yaml
# 实验名称
experiment_name: my_experiment

# 模型配置
model:
  backbone:
    name: mobilenetv3_small        # 主干网络
    pretrained: false             # 是否使用预训练
    rein_insertion_points: []     # Rein插入点
    rein_config: {}              # Rein配置

  head:
    name: deeplabv3plus          # 分割头类型
    num_classes: 21              # 类别数
    dropout_ratio: 0.1           # Dropout比例

  aux_head:
    enabled: false               # 是否启用辅助头

# 数据配置
data:
  root_dir: "/path/to/voc2012"   # 数据集路径
  image_size: [512, 512]         # 输入分辨率
  ignore_index: 255              # 忽略标签

# 训练配置
training:
  epochs: 100                    # 训练轮数
  batch_size: 8                  # 批大小

  optimizer:
    name: sgd                    # 优化器
    lr: 0.01                     # 学习率
    momentum: 0.9                # 动量
    weight_decay: 1e-4           # 权重衰减

  lr_scheduler:
    name: poly                   # 学习率调度器
    power: 0.9                   # poly调度器参数
```

### 支持的组件

**主干网络 (Backbone):**
- `mobilenetv3_small`: MobileNetV3-Small
- `mobilenetv3_large`: MobileNetV3-Large

**分割头 (Head):**
- `deeplabv3plus`: DeepLabV3+头
- `fcn`: FCN头

**注意力机制 (Mechanisms):**
- `rein`: 自适应激活机制

## 🧪 实验管理

### 运行实验

```bash
# 使用配置文件运行实验
python train_complete.py --config configs/your_config.yaml

# 实验会自动创建目录：experiments/{experiment_name}_{timestamp}/
```

### 实验目录结构

```
experiments/experiment_name_timestamp/
├── configs/
│   └── config.yaml           # 保存的配置
├── checkpoints/
│   ├── best_model.pth       # 最佳模型
│   └── latest_model.pth     # 最新模型
├── logs/
│   └── train.log            # 训练日志
└── results/
    ├── training_history.yaml    # 训练历史
    ├── experiment_summary.yaml  # 实验总结
    └── comparison_table.md      # 对比表
```

### 对比表自动生成

系统会自动生成markdown格式的对比表：

```markdown
| Experiment | Backbone          | Head          | Mechanisms | Best mIoU | Parameters (M) | Final Epoch | Batch Size | Learning Rate | Image Size | Final Train Loss | Final Val Loss |
| ---------- | ----------------- | ------------- | ---------- | --------- | -------------- | ----------- | ---------- | ------------- | ---------- | ---------------- | -------------- |
| baseline   | mobilenetv3_small | deeplabv3plus | none       | 0.0162    | 6.41           | 2           | 8          | 0.01          | 256x256    | 3.39             | 3.08           |
```

## 🔬 扩展开发

### 添加新主干网络

1. 在 `models/backbones/` 中实现新主干：

```python
class NewBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # 实现主干网络

    def forward(self, x):
        # 返回多尺度特征
        return {
            'low_level': low_level_features,    # 低级特征
            'high_level': high_level_features   # 高级特征
        }
```

2. 在 `unified_model_builder.py` 中注册：

```python
self.backbone_registry['new_backbone'] = NewBackbone
```

### 添加新分割头

1. 在 `models/heads/` 中实现新分割头：

```python
class NewHead(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        # 实现分割头

    def forward(self, features):
        # features是主干网络输出的特征字典
        return output  # [B, num_classes, H, W]
```

2. 在 `unified_model_builder.py` 中注册：

```python
self.head_registry['new_head'] = NewHead
```

### 添加新注意力机制

1. 在 `models/mechanisms/` 中实现新机制
2. 参考 `rein_mechanism.py` 的插入方式
3. 更新配置文件格式

## 📊 评估指标

系统支持以下评估指标：

- **mIoU**: 平均交并比
- **Pixel Accuracy**: 像素准确率
- **Mean Accuracy**: 平均准确率
- **FwIoU**: 频率加权交并比

## 🎯 实验建议

### 基线实验

1. **MobileNetV3-Small + DeepLabV3+**: 轻量化基线
2. **MobileNetV3-Large + DeepLabV3+**: 精度基线

### 创新实验

1. **Rein机制消融**: 对比有无Rein的效果
2. **插入点分析**: 不同插入位置的影响
3. **多机制组合**: Rein + 其他注意力机制

### 实用技巧

- 使用较小的`image_size`进行快速验证
- 设置`eval_interval=1`密切监控训练过程
- 启用`mixed_precision=true`加速训练
- 调整`batch_size`适配GPU内存

## 🐛 常见问题

### 1. BatchNorm错误
**问题**: `Expected more than 1 value per channel when training`
**解决**: 增加batch_size或在小batch测试时忽略此警告

### 2. CUDA内存不足
**解决**:
- 减小`batch_size`
- 减小`image_size`
- 启用`mixed_precision`

### 3. 数据集路径错误
**解决**:
- 检查`data.root_dir`设置
- 使用dummy数据进行测试

## 📈 性能优化

### 训练加速
- 使用混合精度训练
- 适当的batch_size设置
- 数据预处理优化

### 内存优化
- 梯度检查点
- 适当的图像分辨率
- 模型并行

## 🎉 总结

seg_light_rein 提供了一个完整的轻量化语义分割研究框架，具备：

✅ **完整的实验流程**: 从配置到结果对比
✅ **模块化设计**: 易于扩展和实验
✅ **自动化管理**: 实验归档和对比表生成
✅ **高度可配置**: YAML配置文件支持
✅ **性能优化**: 混合精度、多GPU支持

**系统已就绪，可开始真实数据集实验！** 🚀

---

更多详细信息请参考项目中的代码注释和配置文件示例。
