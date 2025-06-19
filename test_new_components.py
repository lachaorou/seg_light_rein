"""
测试新添加的组件和功能
包括MobileViT、OCR Head、Token Merging等
"""
import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from typing import Dict


def test_mobilevit():
    """测试MobileViT主干网络"""
    print("🧪 Testing MobileViT...")

    try:
        from models.backbones.mobilevit import mobilevit_xs, mobilevit_s

        # 测试MobileViT-XS
        model_xs = mobilevit_xs()
        x = torch.randn(2, 3, 512, 512)
        features = model_xs(x)

        print(f"✅ MobileViT-XS forward pass successful")
        print(f"   Feature shapes:")
        for name, feat in features.items():
            print(f"     {name}: {feat.shape}")

        # 测试特征通道
        channels = model_xs.get_feature_channels()
        print(f"   Feature channels: {channels}")

        # 测试MobileViT-S
        model_s = mobilevit_s()
        features_s = model_s(x)
        print(f"✅ MobileViT-S forward pass successful")

        return True

    except Exception as e:
        print(f"❌ MobileViT test failed: {e}")
        return False


def test_ocr_head():
    """测试OCR分割头"""
    print("\\n🧪 Testing OCR Head...")

    try:
        from models.heads.ocr_head import build_ocr_head

        # 创建测试特征
        features = {
            'high_level_features': torch.randn(2, 2048, 32, 32)
        }

        # 创建OCR头
        ocr_head = build_ocr_head(
            in_channels=2048,
            num_classes=21,
            ocr_mid_channels=512,
            ocr_key_channels=256
        )

        # 前向传播
        outputs = ocr_head(features)

        print(f"✅ OCR Head forward pass successful")
        print(f"   Output shapes:")
        for name, output in outputs.items():
            print(f"     {name}: {output.shape}")

        # 测试损失函数
        targets = torch.randint(0, 21, (2, 1024, 1024))
        loss_fn = ocr_head.get_loss_function()

        # 上采样输出以匹配目标尺寸
        upsampled_outputs = {
            'out': torch.nn.functional.interpolate(
                outputs['out'], size=(1024, 1024),
                mode='bilinear', align_corners=False
            ),
            'aux': torch.nn.functional.interpolate(
                outputs['aux'], size=(1024, 1024),
                mode='bilinear', align_corners=False
            )
        }

        loss = loss_fn(upsampled_outputs, targets)
        print(f"   Loss computation successful: {loss.item():.4f}")

        return True

    except Exception as e:
        print(f"❌ OCR Head test failed: {e}")
        return False


def test_token_merging():
    """测试Token Merging机制"""
    print("\\n🧪 Testing Token Merging...")

    try:
        from models.mechanisms.token_merging import build_tome_module, ToMeBlock

        # 测试ToMe模块
        tome_block = build_tome_module(
            embed_dim=256,
            merge_ratio=0.3,
            merge_mode="mean"
        )

        # 测试4D输入（特征图）
        x_4d = torch.randn(2, 256, 32, 32)
        y_4d = tome_block(x_4d)
        print(f"✅ ToMe 4D input test successful")
        print(f"   Input shape: {x_4d.shape}")
        print(f"   Output shape: {y_4d.shape}")

        # 测试3D输入（序列）
        x_3d = torch.randn(2, 1024, 256)
        y_3d = tome_block(x_3d)
        print(f"✅ ToMe 3D input test successful")
        print(f"   Input shape: {x_3d.shape}")
        print(f"   Output shape: {y_3d.shape}")

        return True

    except Exception as e:
        print(f"❌ Token Merging test failed: {e}")
        return False


def test_unified_model_builder():
    """测试统一模型构建器"""
    print("\\n🧪 Testing Unified Model Builder...")

    try:
        from models.unified_model_builder import ModelBuilder

        builder = ModelBuilder()

        # 测试主干网络注册
        print(f"   Available backbones: {list(builder.backbone_registry.keys())}")
        print(f"   Available heads: {list(builder.head_registry.keys())}")
        print(f"   Available mechanisms: {list(builder.mechanism_registry.keys())}")

        # 测试构建配置
        config = {
            'model': {
                'backbone': {
                    'name': 'mobilevit_xs',
                    'pretrained': False
                },
                'head': {
                    'name': 'deeplabv3plus',
                    'num_classes': 21,
                    'in_channels': 384  # MobileViT-XS输出通道
                }
            }
        }

        model = builder.build_model(config)

        # 测试前向传播
        x = torch.randn(1, 3, 512, 512)
        outputs = model(x)

        print(f"✅ Model building and forward pass successful")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Output shapes:")
        for name, output in outputs.items():
            print(f"     {name}: {output.shape}")

        # 测试模型信息
        model_info = model.get_model_info()
        param_count = model.count_parameters()

        print(f"   Model info: {model_info}")
        print(f"   Parameters: {param_count}")

        return True

    except Exception as e:
        print(f"❌ Unified Model Builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_configs():
    """测试实验配置文件"""
    print("\\n🧪 Testing Experiment Configs...")

    try:
        import yaml
        from pathlib import Path

        config_dir = Path("configs")
        config_files = list(config_dir.glob("*.yaml"))

        print(f"   Found {len(config_files)} config files")

        valid_configs = 0
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # 检查必要字段
                required_fields = ['experiment_name', 'model', 'data', 'training']
                missing_fields = [field for field in required_fields if field not in config]

                if missing_fields:
                    print(f"   ⚠️  {config_file.name}: missing fields {missing_fields}")
                else:
                    valid_configs += 1

            except Exception as e:
                print(f"   ❌ {config_file.name}: {e}")

        print(f"✅ Config validation completed: {valid_configs}/{len(config_files)} valid")
        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("🚀 Starting Component Tests...\\n")

    tests = [
        ("MobileViT", test_mobilevit),
        ("OCR Head", test_ocr_head),
        ("Token Merging", test_token_merging),
        ("Unified Model Builder", test_unified_model_builder),
        ("Experiment Configs", test_experiment_configs),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # 输出总结
    print(f"\\n{'='*50}")
    print("📋 Test Summary:")
    print(f"{'='*50}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20} {status}")

    print(f"\\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! System is ready for experiments.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
