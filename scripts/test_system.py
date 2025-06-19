"""
快速测试训练流程
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from configs.config_manager import ConfigManager


def test_config_system():
    """测试配置系统"""
    print("=== 测试配置系统 ===")

    config_manager = ConfigManager()

    # 测试基线配置生成
    baseline_configs = config_manager.create_baseline_configs()

    for name, config in baseline_configs.items():
        print(f"配置 {name}:")
        print(f"  主干: {config.backbone.name}")
        print(f"  R机制: {'启用' if config.rein.enabled else '禁用'}")
        print(f"  分割头: {config.head.name}")
        print(f"  类别数: {config.head.num_classes}")
        print()


def test_model_building():
    """测试模型构建"""
    print("=== 测试模型构建 ===")

    from models.model_builder import ModelBuilder
    from configs.config_manager import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load_config('configs/mobilenetv3_baseline.yaml')

    model_builder = ModelBuilder()
    model = model_builder.build_model(config)

    # 测试前向传播
    x = torch.randn(1, 3, 512, 512)

    print("模型前向传播测试...")
    with torch.no_grad():
        output = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    return model


def test_training_components():
    """测试训练组件"""
    print("=== 测试训练组件 ===")

    from utils.dataset import SegmentationDataset
    from utils.metrics import SegmentationMetrics
    from torch.utils.data import DataLoader

    # 测试数据集
    dataset = SegmentationDataset(split='train', debug=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"数据集大小: {len(dataset)}")

    # 测试一个batch
    for images, labels in dataloader:
        print(f"图像批次形状: {images.shape}")
        print(f"标签批次形状: {labels.shape}")
        break

    # 测试评估指标
    metrics = SegmentationMetrics(num_classes=21)

    # 模拟预测和真实标签
    pred = torch.randint(0, 21, (2, 512, 512))
    target = torch.randint(0, 21, (2, 512, 512))

    metrics.update(pred, target)
    result = metrics.get_metrics()

    print("评估指标测试:")
    for key, value in result.items():
        if key != 'per_class_iou':
            print(f"  {key}: {value:.4f}")


def main():
    """主测试函数"""
    print("开始测试 seg_light_rein 训练系统...")
    print("="*50)

    try:
        # 测试配置系统
        test_config_system()

        # 测试模型构建
        model = test_model_building()

        # 测试训练组件
        test_training_components()

        print("="*50)
        print("✅ 所有测试通过！训练系统准备就绪。")
        print("\n接下来可以运行:")
        print("python scripts/train.py --config configs/mobilenetv3_baseline.yaml --debug")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
