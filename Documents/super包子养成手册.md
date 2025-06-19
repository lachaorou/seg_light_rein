# 新计划

## 项目结构与模块化建议

- 采用“主干-机制-分割头”三级目录，所有模块均可独立维护和扩展。
- backbone（主干）、R机制、ASPP、分割头均模块化，支持插拔和参数化组合。
- 插入点通过配置文件/参数指定，避免硬编码，提升可维护性。

## 主流轻量主干与创新机制推荐

### 轻量主干推荐
- MobileNetV3
- EfficientNet-Lite
- MobileViT
- PIDNet
- GhostNet
- EdgeNeXt
- RepVGG
- ShuffleNetV2
- SwiftFormer

### 可插拔机制/创新点
- Token Merging/Pruning
- Prompt/Adapter机制
- Dynamic Head/Attention
- Lightweight Transformer Block
- RepLKNet Block
- 轻量注意力（ECA、CBAM、SE等）
- 结构重参数化（RepVGG思想）
- 轻量解码头（Lightweight OCR/UPerHead）

### 损失函数/训练技巧
- OHEM Loss、Dice Loss、Lovasz Loss
- Label Smoothing、Mixup/CutMix
- EMA、Lookahead、Ranger优化器

## 六大步骤详细执行指令（含baseline与评估指标）

### 步骤1：主干选择与适配
- 操作：在models/backbones/下实现或迁移主流轻量主干（如MobileNetV3、PIDNet、MobileViT等）。
- baseline：以官方repo/论文公开mIoU、参数量、FPS为基线，记录于Documents/baseline与对比表.md。
- 评估指标：mIoU、参数量、FLOPs、FPS、推理延迟、显存占用。
- 备案：每次主干实验结果与baseline对比，归档config、日志、指标。

### 步骤2：R机制/创新结构模块化
- 操作：在models/rein/下实现R机制及其它创新结构，设计统一接口，支持插拔。
- baseline：无R机制（纯主干+ASPP）为基线，或以论文权威结果为基线。
- 评估指标：mIoU提升、参数量变化、推理速度变化。
- 备案：对比有无R机制的实验结果，归档对比表。

### 步骤3：分割头/ASPP等模块化
- 操作：在models/heads/和models/aspp.py下实现可替换分割头和ASPP/其它特征增强模块。
- baseline：以FCN Head、ASPP为基线，或主流分割头公开结果。
- 评估指标：mIoU、参数量、推理速度。
- 备案：对比不同分割头/增强模块的实验结果。

### 步骤4：配置文件与批量实验脚本
- 操作：在configs/和scripts/下实现参数化配置和批量实验脚本，支持自动归档。
- baseline：以主流配置为基线，记录参数变动对结果的影响。
- 评估指标：实验可复现性、自动归档完整性。
- 备案：每次实验config、日志、指标自动归档。

### 步骤5：系统性消融实验与对比表生成
- 操作：设计消融实验（如主干、R机制、分割头、损失函数等单独/组合对比），自动生成对比表。
- baseline：每组实验均有明确对照组（如无R机制、无ASPP、不同分割头等）。
- 评估指标：mIoU提升、参数量变化、速度变化。
- 备案：对比表、可视化结果归档于Documents/实验记录与可视化对比表.md。

### 步骤6：创新点归纳与论文写作
- 操作：归纳创新点、撰写创新点与结论，整理实验对比和提升点。
- baseline：以主流SOTA和自身baseline为对照，突出创新提升。
- 评估指标：创新点提升幅度、适用场景、论文可发表性。
- 备案：创新点、对比表、论文草稿归档。

## baseline与备案模板

### baseline对比表模板（Markdown格式）
| 主干        | 机制 | 分割头  | mIoU(%) | Params(M) | FPS | 论文/来源     | 备注       |
| ----------- | ---- | ------- | ------- | --------- | --- | ------------- | ---------- |
| MobileNetV2 | 无   | ASPP    | 71.3    | 2.1       | 65  | 官方repo/论文 | baseline   |
| MobileNetV3 | 无   | ASPP    | 72.5    | 1.9       | 70  | 论文/公开代码 |            |
| PIDNet-S    | 无   | FCNHead | 75.2    | 3.0       | 80  | 论文/官方repo |            |
| MobileViT-S | 无   | FCNHead | 76.1    | 2.8       | 60  | 论文/官方repo |            |
| MobileNetV2 | Rein | ASPP    | 73.8    | 2.3       | 62  | 本项目实验    | +R机制提升 |
| ...         | ...  | ...     | ...     | ...       | ... | ...           | ...        |

### 评估指标说明
- mIoU：主流分割精度指标，越高越好。
- Params：参数量，越小越轻量。
- FPS：推理速度，越高越快。
- FLOPs、显存占用、推理延迟等可选补充。

---

# 已经完成的实验及试验记录及经验心得

...（原有内容请整体移动至此专栏下）...
