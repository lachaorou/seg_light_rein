"""
模型构建器 - 根据配置文件动态构建模型
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from configs.config_manager import ExperimentConfig


class ModelBuilder:
    """模型构建器"""

    def __init__(self):
        self.backbone_registry = {}
        self.rein_registry = {}
        self.head_registry = {}
        self._register_components()

    def _register_components(self):
        """注册所有可用组件"""
        # 注册主干网络
        self.backbone_registry = {
            'mobilenetv3_small': self._build_mobilenetv3_small,
            'mobilenetv3_large': self._build_mobilenetv3_large,
            # 'pidnet_s': self._build_pidnet_s,
            # 'mobilevit_s': self._build_mobilevit_s,
        }

        # 注册R机制
        self.rein_registry = {
            'rein': self._build_rein_mechanism,
            'token_merging': self._build_token_merging,
            'attention': self._build_attention_mechanism,
        }

        # 注册分割头
        self.head_registry = {
            'fcn_head': self._build_fcn_head,
            'aspp_head': self._build_aspp_head,
            # 'ocr_head': self._build_ocr_head,
        }

    def build_model(self, config: ExperimentConfig) -> nn.Module:
        """根据配置构建完整模型"""
        # 构建主干网络
        backbone = self._build_backbone(config.backbone)

        # 构建R机制
        rein_mechanism = None
        if config.rein.enabled:
            rein_mechanism = self._build_rein(config.rein, backbone)

        # 构建分割头
        head = self._build_head(config.head, backbone)

        # 组装完整模型
        model = SegmentationModel(
            backbone=backbone,
            rein_mechanism=rein_mechanism,
            head=head,
            config=config
        )

        return model

    def _build_backbone(self, backbone_config) -> nn.Module:
        """构建主干网络"""
        if backbone_config.name not in self.backbone_registry:
            raise ValueError(f"不支持的主干网络: {backbone_config.name}")

        builder_func = self.backbone_registry[backbone_config.name]
        backbone = builder_func(backbone_config)

        return backbone

    def _build_rein(self, rein_config, backbone) -> nn.Module:
        """构建R机制"""
        if rein_config.mechanism_type not in self.rein_registry:
            raise ValueError(f"不支持的R机制: {rein_config.mechanism_type}")

        builder_func = self.rein_registry[rein_config.mechanism_type]
        rein_mechanism = builder_func(rein_config, backbone)

        return rein_mechanism

    def _build_head(self, head_config, backbone) -> nn.Module:
        """构建分割头"""
        if head_config.name not in self.head_registry:
            raise ValueError(f"不支持的分割头: {head_config.name}")

        builder_func = self.head_registry[head_config.name]
        head = builder_func(head_config, backbone)

        return head

    # ========== 主干网络构建函数 ==========    def _build_mobilenetv3_small(self, backbone_config):
        """构建MobileNetV3-Small"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from models.backbones.mobilenetv3 import MobileNetV3Small

        backbone = MobileNetV3Small(
            num_classes=0,  # 只用于特征提取
            pretrained=backbone_config.pretrained,
            rein_insertion_points=backbone_config.rein_insertion_points
        )

        return backbone

    def _build_mobilenetv3_large(self, backbone_config):
        """构建MobileNetV3-Large"""
        from models.backbones.mobilenetv3 import MobileNetV3Large

        backbone = MobileNetV3Large(
            pretrained=backbone_config.pretrained,
            freeze_bn=backbone_config.freeze_bn
        )

        return backbone

    # ========== R机制构建函数 ==========

    def _build_rein_mechanism(self, rein_config, backbone):
        """构建Rein机制"""
        from models.rein.rein_module import ReinMechanism

        rein = ReinMechanism(
            insertion_points=rein_config.parameters.get('insertion_points', []),
            **rein_config.parameters
        )

        return rein

    def _build_token_merging(self, rein_config, backbone):
        """构建Token Merging机制"""
        from models.rein.token_merging import TokenMerging

        token_merging = TokenMerging(**rein_config.parameters)

        return token_merging

    def _build_attention_mechanism(self, rein_config, backbone):
        """构建注意力机制"""
        from models.rein.attention import AttentionMechanism

        attention = AttentionMechanism(**rein_config.parameters)

        return attention

    # ========== 分割头构建函数 ==========

    def _build_fcn_head(self, head_config, backbone):
        """构建FCN分割头"""
        from models.heads.fcn_head import FCNHead

        # 获取主干网络输出通道数
        backbone_channels = self._get_backbone_channels(backbone)

        head = FCNHead(
            in_channels=backbone_channels,
            num_classes=head_config.num_classes,
            dropout_ratio=head_config.dropout_ratio,
            feature_enhancement=head_config.feature_enhancement
        )

        return head

    def _build_aspp_head(self, head_config, backbone):
        """构建ASPP分割头"""
        from models.heads.aspp_head import ASPPHead

        backbone_channels = self._get_backbone_channels(backbone)

        head = ASPPHead(
            in_channels=backbone_channels,
            num_classes=head_config.num_classes,
            dropout_ratio=head_config.dropout_ratio
        )

        return head

    def _get_backbone_channels(self, backbone) -> int:
        """获取主干网络输出通道数"""
        # 这里需要根据具体主干网络实现来获取
        # 临时返回一个默认值
        if hasattr(backbone, 'out_channels'):
            return backbone.out_channels
        else:
            # 默认值，后续需要根据实际主干网络调整
            return 960  # MobileNetV3-Small的输出通道数


class SegmentationModel(nn.Module):
    """完整的分割模型"""

    def __init__(self, backbone: nn.Module, rein_mechanism: nn.Module,
                 head: nn.Module, config: ExperimentConfig):
        super(SegmentationModel, self).__init__()

        self.backbone = backbone
        self.rein_mechanism = rein_mechanism
        self.head = head
        self.config = config

    def forward(self, x):
        # 主干网络前向传播
        features = self.backbone(x)

        # 应用R机制（如果启用）
        if self.rein_mechanism is not None:
            features = self.rein_mechanism(features)

        # 分割头前向传播
        output = self.head(features)

        return output

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone': self.config.backbone.name,
            'rein_enabled': self.config.rein.enabled,
            'head': self.config.head.name,
        }
