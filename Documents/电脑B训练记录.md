# 电脑B训练实验记录

## 🎯 训练目标
- **对比实验**: MobileNetV3-Small 基线 vs Rein增强版
- **数据集**: VOC2012 (真实数据集或dummy数据)
- **目标**: 验证Rein机制在长时间训练下的效果

## 📋 实验配置

### 实验1: MobileNetV3-Small 基线
- **配置文件**: `configs/server_mobilenetv3_small_baseline_100epochs.yaml`
- **主干网络**: MobileNetV3-Small
- **分割头**: DeepLabV3+
- **创新机制**: 无
- **预期参数量**: ~6.41M
- **训练轮数**: 100 epochs
- **批次大小**: 16 (服务器可调整)
- **学习率**: 0.01 → 0.0001 (cosine调度)

### 实验2: MobileNetV3-Small + Rein
- **配置文件**: `configs/server_mobilenetv3_small_rein_100epochs.yaml`
- **主干网络**: MobileNetV3-Small
- **分割头**: DeepLabV3+
- **创新机制**: Rein (Stage 3,4插入)
- **Rein配置**:
  - embed_dim: 256
  - num_heads: 8
  - dropout: 0.1
- **预期参数量**: ~6.41M
- **训练轮数**: 100 epochs
- **预期提升**: 35-50% mIoU (基于电脑A验证)

## 🚀 启动命令

```bash
# 在电脑B上执行
cd seg_light_rein

# 启动基线实验
nohup python train_complete.py --config configs/server_mobilenetv3_small_baseline_100epochs.yaml > baseline_train.log 2>&1 &

# 启动Rein实验 (可以同时跑，如果GPU内存够)
nohup python train_complete.py --config configs/server_mobilenetv3_small_rein_100epochs.yaml > rein_train.log 2>&1 &

# 查看进程
ps aux | grep python

# 查看实时日志
tail -f baseline_train.log
tail -f rein_train.log
```

## 📊 结果记录表

### 实验进度跟踪
| 实验 | 开始时间 | 当前Epoch | 最佳mIoU | 当前Loss | 预计完成时间 | 状态     |
| ---- | -------- | --------- | -------- | -------- | ------------ | -------- |
| 基线 | ______   | ___/100   | ______   | ______   | ______       | ⏸️ 待开始 |
| Rein | ______   | ___/100   | ______   | ______   | ______       | ⏸️ 待开始 |

### 最终结果对比
| 指标         | 基线    | Rein    | 提升    |
| ------------ | ------- | ------- | ------- |
| 最佳mIoU     | ______  | ______  | ______% |
| 最终训练Loss | ______  | ______  | ______  |
| 最终验证Loss | ______  | ______  | ______  |
| 训练时间     | ______h | ______h | ______  |
| 参数量       | ______M | ______M | ______  |

## 🔍 关键观察点

### 训练稳定性
- [ ] 是否有Loss突然跳跃？
- [ ] 学习率调度是否合理？
- [ ] 内存使用是否正常？

### Rein机制效果
- [ ] 相比基线，mIoU提升多少？
- [ ] 训练时间是否明显增加？
- [ ] 是否出现过拟合现象？

### 可能需要调整的参数
- [ ] 如果效果不好，可以调整Rein的embed_dim
- [ ] 如果训练太慢，可以减少batch_size
- [ ] 如果过拟合，可以增加dropout

## 📝 实验日志

### [日期] - 实验开始
- 时间: ______
- GPU: ______
- 数据集路径: ______
- 备注: ______

### [日期] - 进度检查
- 基线进度: ______
- Rein进度: ______
- 初步观察: ______

### [日期] - 实验完成
- 最终结果: ______
- 结论: ______
- 下一步计划: ______
