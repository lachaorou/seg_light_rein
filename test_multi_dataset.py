#!/usr/bin/env python3
"""
多数据集框架测试脚本
验证universal_dataset.py的功能
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from datasets.universal_dataset import create_dataset

def test_dataset_creation():
    """测试数据集创建"""
    print("🧪 测试数据集创建功能...")

    # 测试参数
    test_configs = [
        {
            'name': 'VOC2012',
            'type': 'voc2012',
            'root_dir': './datasets/VOC2012',
            'expected_classes': 21
        },
        {
            'name': 'ADE20K',
            'type': 'ade20k',
            'root_dir': './datasets/ADE20K',
            'expected_classes': 150
        },
        {
            'name': 'Cityscapes',
            'type': 'cityscapes',
            'root_dir': './datasets/Cityscapes',
            'expected_classes': 19
        }
    ]

    results = []

    for config in test_configs:
        print(f"\n📁 测试 {config['name']} 数据集...")

        try:
            # 创建数据集
            dataset = create_dataset(
                dataset_type=config['type'],
                root_dir=config['root_dir'],
                split='train',
                image_size=(256, 256)
            )

            print(f"✅ {config['name']} 数据集创建成功")
            print(f"   - 数据集大小: {len(dataset)}")
            print(f"   - 预期类别数: {config['expected_classes']}")
            print(f"   - 实际类别数: {dataset.num_classes}")

            # 检查类别数是否正确
            if dataset.num_classes == config['expected_classes']:
                print(f"   - ✅ 类别数正确")
            else:
                print(f"   - ⚠️ 类别数不匹配")

            # 测试数据加载
            if len(dataset) > 0:
                try:
                    image, mask = dataset[0]
                    print(f"   - 图像形状: {image.shape}")
                    print(f"   - 标签形状: {mask.shape}")
                    print(f"   - 图像数据类型: {image.dtype}")
                    print(f"   - 标签数据类型: {mask.dtype}")
                    print(f"   - ✅ 数据加载正常")
                except Exception as e:
                    print(f"   - ❌ 数据加载失败: {e}")
            else:
                print(f"   - ⚠️ 数据集为空（使用dummy数据）")

            results.append({
                'name': config['name'],
                'success': True,
                'dataset': dataset
            })

        except Exception as e:
            print(f"❌ {config['name']} 数据集创建失败: {e}")
            results.append({
                'name': config['name'],
                'success': False,
                'error': str(e)
            })

    return results

def test_dataloader():
    """测试DataLoader"""
    print("\n🧪 测试DataLoader功能...")

    try:
        # 创建一个测试数据集
        dataset = create_dataset(
            dataset_type='voc2012',
            root_dir='./datasets/VOC2012',
            split='train',
            image_size=(256, 256)
        )

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # 避免多进程问题
        )

        print(f"✅ DataLoader创建成功")
        print(f"   - 批次大小: {dataloader.batch_size}")
        print(f"   - 数据集大小: {len(dataloader.dataset)}")

        # 测试一个批次
        for batch_idx, (images, masks) in enumerate(dataloader):
            print(f"   - 批次 {batch_idx}:")
            print(f"     * 图像批次形状: {images.shape}")
            print(f"     * 标签批次形状: {masks.shape}")
            print(f"     * 图像数据范围: [{images.min():.3f}, {images.max():.3f}]")
            print(f"     * 标签数据范围: [{masks.min()}, {masks.max()}]")

            if batch_idx >= 1:  # 只测试2个批次
                break

        print(f"   - ✅ DataLoader工作正常")
        return True

    except Exception as e:
        print(f"❌ DataLoader测试失败: {e}")
        return False

def test_config_integration():
    """测试配置文件集成"""
    print("\n🧪 测试配置文件集成...")

    try:
        from configs.config_manager import ConfigManager

        # 测试配置文件
        config_files = [
            'configs/voc2012_quick_test.yaml',
            'configs/ade20k_full_eval.yaml',
            'configs/cityscapes_precision_test.yaml'
        ]

        for config_file in config_files:
            config_path = project_root / config_file

            if config_path.exists():
                print(f"📄 测试配置文件: {config_file}")

                try:
                    # 直接读取YAML文件
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)

                    # 验证必要字段
                    required_fields = ['dataset', 'model', 'training', 'experiment']
                    missing_fields = []

                    for field in required_fields:
                        if field not in config_data:
                            missing_fields.append(field)

                    if not missing_fields:
                        print(f"   - ✅ 配置文件格式正确")
                        print(f"   - 数据集类型: {config_data.get('dataset', {}).get('type', 'unknown')}")
                        print(f"   - 实验名称: {config_data.get('experiment', {}).get('name', 'unknown')}")
                    else:
                        print(f"   - ⚠️ 缺少字段: {missing_fields}")

                except Exception as e:
                    print(f"   - ❌ 配置文件解析失败: {e}")
            else:
                print(f"📄 配置文件不存在: {config_file}")

        print(f"✅ 配置文件集成测试完成")
        return True

    except Exception as e:
        print(f"❌ 配置文件集成测试失败: {e}")
        return False

def generate_test_report(dataset_results, dataloader_success, config_success):
    """生成测试报告"""
    print("\n" + "="*60)
    print("📊 多数据集框架测试报告")
    print("="*60)

    # 数据集测试结果
    print("\n🗂️ 数据集测试结果:")
    successful_datasets = 0
    for result in dataset_results:
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"   - {result['name']}: {status}")
        if result['success']:
            successful_datasets += 1

    print(f"\n数据集成功率: {successful_datasets}/{len(dataset_results)} ({successful_datasets/len(dataset_results)*100:.1f}%)")

    # DataLoader测试结果
    dataloader_status = "✅ 成功" if dataloader_success else "❌ 失败"
    print(f"\n🔄 DataLoader测试: {dataloader_status}")

    # 配置文件测试结果
    config_status = "✅ 成功" if config_success else "❌ 失败"
    print(f"⚙️ 配置文件测试: {config_status}")

    # 总体评估
    total_tests = len(dataset_results) + 2
    successful_tests = successful_datasets + (1 if dataloader_success else 0) + (1 if config_success else 0)

    print(f"\n📈 总体测试结果: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")

    if successful_tests == total_tests:
        print("\n🎉 所有测试通过！多数据集框架工作正常。")
        recommendation = """
🚀 下一步建议:
1. 准备真实数据集并放置到正确目录
2. 运行单数据集实验验证功能
3. 执行批量实验测试完整流程
4. 根据实验结果调优模型和参数
        """
    elif successful_tests >= total_tests * 0.7:
        print("\n⚠️ 大部分测试通过，但有部分问题需要解决。")
        recommendation = """
🔧 建议修复:
1. 检查失败的数据集配置
2. 确认数据集路径和格式
3. 验证配置文件完整性
4. 修复后重新测试
        """
    else:
        print("\n❌ 多个测试失败，需要检查框架配置。")
        recommendation = """
🚨 紧急修复:
1. 检查项目依赖和环境
2. 验证代码完整性
3. 确认配置文件格式
4. 重新安装或重构代码
        """

    print(recommendation)

    return successful_tests == total_tests

def main():
    """主函数"""
    print("🔬 多数据集框架功能测试")
    print("="*60)

    # 运行测试
    dataset_results = test_dataset_creation()
    dataloader_success = test_dataloader()
    config_success = test_config_integration()

    # 生成报告
    all_passed = generate_test_report(dataset_results, dataloader_success, config_success)

    return 0 if all_passed else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
