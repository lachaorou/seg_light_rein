"""
统一的模型构建器
整合主干网络、创新机制、分割头等组件
支持配置化构建和插拔式组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import importlib
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from backbones.mobilenetv3_real import build_mobilenet_v3
from backbones.pidnet import pidnet_s, pidnet_m, pidnet_l
from backbones.mobilevit import mobilevit_xs, mobilevit_s
from heads.deeplabv3plus_head import build_aspp_head
from mechanisms.rein_mechanism import ReinInserter, build_rein_module
from mechanisms.token_merging import build_tome_module


class SegmentationModel(nn.Module):
    """统一的语义分割模型"""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        mechanisms: Optional[Dict] = None,
        aux_head: Optional[nn.Module] = None
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.aux_head = aux_head
        self.mechanisms = mechanisms or {}

        # 记录模型信息
        self._model_info = {
            'backbone_type': type(backbone).__name__,
            'head_type': type(head).__name__,
            'mechanisms': list(self.mechanisms.keys()) if mechanisms else [],
            'has_aux_head': aux_head is not None
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            预测结果字典
        """
        input_size = x.shape[-2:]

        # 提取特征
        if hasattr(self.backbone, 'extract_features'):
            # 如果backbone有extract_features方法
            features = self.backbone.extract_features(x)
        else:
            # 否则直接forward
            features = self.backbone(x)

        # 根据head类型处理特征
        if hasattr(self.head, 'forward') and len(self.head.forward.__code__.co_varnames) > 2:
            # DeepLabV3+需要高级和低级特征
            if isinstance(features, (list, tuple)):
                high_level = features[-1]
                low_level = features[0] if len(features) > 1 else features[-1]
            else:
                # 如果只有单一特征，需要从backbone提取多级特征
                high_level = features
                low_level = self._extract_low_level_features(x)

            pred = self.head(high_level, low_level)
        else:
            # DeepLabV3或其他头只需要高级特征
            if isinstance(features, (list, tuple)):
                features = features[-1]
            pred = self.head(features)

        # 上采样到输入尺寸
        pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=False)

        results = {'pred': pred}

        # 辅助头
        if self.aux_head is not None and self.training:
            if isinstance(features, (list, tuple)):
                aux_feat = features[-2] if len(features) > 1 else features[-1]
            else:
                aux_feat = features
            aux_pred = self.aux_head(aux_feat)
            aux_pred = F.interpolate(aux_pred, size=input_size, mode='bilinear', align_corners=False)
            results['aux'] = aux_pred
        return results

    def _extract_low_level_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取低级特征（用于DeepLabV3+）"""
        # 这里需要根据具体的backbone实现
        if hasattr(self.backbone, 'features'):
            # MobileNetV3风格
            features = self.backbone.features
            # 提取早期特征：第一个卷积层+第一个倒残差块
            # features[:4] 包含：Conv2d -> BN -> Hardswish -> InvertedResidual
            x = features[:4](x)  # 前4层，对于MobileNetV3-Small输出16通道
            return x
        else:
            # 其他情况的简单处理
            conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1).to(x.device)
            with torch.no_grad():
                return conv(x)

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return self._model_info.copy()

    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())

        result = {
            'total': total_params,
            'trainable': trainable_params,
            'backbone': backbone_params,
            'head': head_params
        }

        if self.aux_head is not None:
            aux_params = sum(p.numel() for p in self.aux_head.parameters())
            result['aux_head'] = aux_params

        return result


class ModelBuilder:
    """模型构建器"""

    def __init__(self):
        self.backbone_registry = {
            'mobilenetv3_small': lambda **kwargs: build_mobilenet_v3('small', **kwargs),
            'mobilenetv3_large': lambda **kwargs: build_mobilenet_v3('large', **kwargs),
            'pidnet_s': lambda **kwargs: pidnet_s(**kwargs),
            'pidnet_m': lambda **kwargs: pidnet_m(**kwargs),
            'pidnet_l': lambda **kwargs: pidnet_l(**kwargs),
            'mobilevit_xs': lambda **kwargs: mobilevit_xs(**kwargs),
            'mobilevit_s': lambda **kwargs: mobilevit_s(**kwargs),
        }

        self.head_registry = {
            'deeplabv3plus': build_aspp_head,
            'deeplabv3': lambda **kwargs: build_aspp_head('deeplabv3', **kwargs),
        }

        self.mechanism_registry = {
            'rein': build_rein_module,
            'tome': build_tome_module,
        }

    def register_backbone(self, name: str, builder_func):
        """注册新的主干网络"""
        self.backbone_registry[name] = builder_func

    def register_head(self, name: str, builder_func):
        """注册新的分割头"""
        self.head_registry[name] = builder_func

    def register_mechanism(self, name: str, builder_func):
        """注册新的机制"""
        self.mechanism_registry[name] = builder_func

    def build_backbone(self, config: Dict) -> nn.Module:
        """构建主干网络"""
        backbone_name = config.get('name', 'mobilenetv3_small')
        backbone_params = config.copy()
        backbone_params.pop('name', None)

        if backbone_name not in self.backbone_registry:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        backbone = self.backbone_registry[backbone_name](**backbone_params)

        # 插入机制（如Rein）
        rein_points = config.get('rein_insertion_points', [])
        if rein_points:
            rein_config = config.get('rein_config', {})
            backbone = ReinInserter.insert_rein_to_backbone(
                backbone, rein_points, rein_config
            )

        return backbone

    def build_head(self, config: Dict, backbone_channels: int = None) -> nn.Module:
        """构建分割头"""
        head_name = config.get('name', 'deeplabv3plus')
        head_params = config.copy()
        head_params.pop('name', None)

        # 自动设置输入通道数
        if backbone_channels is not None:
            head_params.setdefault('in_channels', backbone_channels)

        # 设置低级特征通道数（用于DeepLabV3+）
        if head_name == 'deeplabv3plus':
            # 根据不同的backbone设置合适的低级特征通道数
            backbone_name = config.get('_backbone_name', '')
            if 'mobilenetv3_small' in backbone_name:
                head_params.setdefault('low_level_channels', 16)  # MobileNetV3-Small第一个倒残差块输出
            elif 'mobilenetv3_large' in backbone_name:
                head_params.setdefault('low_level_channels', 24)  # MobileNetV3-Large第一个倒残差块输出
            else:
                head_params.setdefault('low_level_channels', 24)  # 默认值

        # 移除内部参数，避免传递给head构建函数
        head_params.pop('_backbone_name', None)
        head_params.pop('_backbone_channels', None)

        if head_name not in self.head_registry:
            raise ValueError(f"Unknown head: {head_name}")

        return self.head_registry[head_name](**head_params)

    def build_model(self, config: Dict) -> SegmentationModel:
        """构建完整的分割模型"""
        # 构建主干网络
        backbone_config = config.get('backbone', {})
        backbone = self.build_backbone(backbone_config)

        # 获取主干网络输出通道数
        if hasattr(backbone, 'out_channels'):
            backbone_channels = backbone.out_channels
        else:
            # 通过前向传播推断
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = backbone(dummy_input)
                if isinstance(dummy_output, (list, tuple)):
                    backbone_channels = dummy_output[-1].shape[1]
                else:
                    backbone_channels = dummy_output.shape[1]        # 构建分割头
        head_config = config.get('head', {})
        # 传递backbone信息给head构建器
        head_config['_backbone_name'] = backbone_config.get('name', '')
        head = self.build_head(head_config, backbone_channels)

        # 构建辅助头（可选）
        aux_head = None
        if config.get('aux_head', {}).get('enabled', False):
            aux_config = config['aux_head']
            aux_config.setdefault('in_channels', backbone_channels // 2)  # 通常使用中间层特征
            aux_head = self.build_head(aux_config, aux_config['in_channels'])

        # 收集机制信息
        mechanisms = {}
        if backbone_config.get('rein_insertion_points'):
            mechanisms['rein'] = backbone_config.get('rein_config', {})

        return SegmentationModel(
            backbone=backbone,
            head=head,
            mechanisms=mechanisms,
            aux_head=aux_head
        )

    def build_from_config(self, config_path: str = None, config_dict: Dict = None) -> SegmentationModel:
        """从配置文件或配置字典构建模型"""
        if config_dict is not None:
            config = config_dict
        elif config_path is not None:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        return self.build_model(config)


# 全局模型构建器实例
model_builder = ModelBuilder()


def build_segmentation_model(config: Dict) -> SegmentationModel:
    """构建分割模型的便捷函数"""
    return model_builder.build_model(config)


def get_model_info(model: SegmentationModel) -> Dict:
    """获取模型的详细信息"""
    info = model.get_model_info()
    params = model.count_parameters()

    # 计算FLOPs（简化版）
    dummy_input = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        # 这里可以使用更精确的FLOPs计算工具
        # 暂时提供一个简单的估算
        total_params = params['total']
        # 粗略估算：每个参数大约对应2个运算（乘法+加法）
        estimated_flops = total_params * 2 * 512 * 512 / (224 * 224)  # 按输入尺寸缩放

    result = {
        'model_info': info,
        'parameters': params,
        'estimated_flops': estimated_flops,
        'parameters_mb': params['total'] * 4 / (1024 * 1024),  # 假设float32
    }

    return result


if __name__ == "__main__":
    # 测试代码
    print("Testing Model Builder...")

    # 测试配置
    test_config = {
        'backbone': {
            'name': 'mobilenetv3_small',
            'pretrained': False,
            'rein_insertion_points': [],  # 暂时不插入Rein
            'rein_config': {
                'reduction': 16,
                'activation_types': ['relu', 'sigmoid', 'tanh', 'identity']
            }
        },
        'head': {
            'name': 'deeplabv3plus',
            'num_classes': 21,
            'dropout_ratio': 0.1
        },
        'aux_head': {
            'enabled': False
        }
    }

    # 构建模型
    model = build_segmentation_model(test_config)

    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dummy_input = torch.randn(2, 3, 512, 512).to(device)

    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"Model output shape: {outputs['pred'].shape}")

    # 获取模型信息
    info = get_model_info(model)
    print(f"Model info: {info['model_info']}")
    print(f"Total parameters: {info['parameters']['total'] / 1e6:.2f}M")
    print(f"Model size: {info['parameters_mb']:.2f} MB")
    print(f"Estimated FLOPs: {info['estimated_flops'] / 1e9:.2f} GFLOPs")
