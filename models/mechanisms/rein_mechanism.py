"""
Rein机制实现
基于论文：Activate or Not: Learning Customized Activation
支持可插拔式集成到各种网络结构中
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import math


class ReinModule(nn.Module):
    """Rein激活模块

    通过学习自适应激活函数，增强网络的表达能力
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation_types: List[str] = ['relu', 'sigmoid', 'tanh', 'identity'],
        learnable_weight: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()

        self.channels = channels
        self.reduction = reduction
        self.activation_types = activation_types
        self.num_activations = len(activation_types)
        self.learnable_weight = learnable_weight
        self.temperature = temperature

        # 通道注意力机制用于生成权重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, self.num_activations, 1, bias=False)
        )

        # 可学习的激活权重
        if learnable_weight:
            self.activation_weights = nn.Parameter(
                torch.ones(self.num_activations) / self.num_activations
            )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _apply_activation(self, x: torch.Tensor, activation_type: str) -> torch.Tensor:
        """应用指定类型的激活函数"""
        if activation_type == 'relu':
            return F.relu(x)
        elif activation_type == 'sigmoid':
            return torch.sigmoid(x)
        elif activation_type == 'tanh':
            return torch.tanh(x)
        elif activation_type == 'identity':
            return x
        elif activation_type == 'leaky_relu':
            return F.leaky_relu(x, 0.2)
        elif activation_type == 'elu':
            return F.elu(x)
        elif activation_type == 'swish':
            return x * torch.sigmoid(x)
        elif activation_type == 'gelu':
            return F.gelu(x)
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            增强后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 生成注意力权重 [B, num_activations, 1, 1]
        attention_weights = self.attention(x)
        attention_weights = torch.softmax(attention_weights / self.temperature, dim=1)

        # 如果有可学习权重，结合它们
        if self.learnable_weight:
            # 广播可学习权重 [1, num_activations, 1, 1]
            learnable_weights = self.activation_weights.view(1, -1, 1, 1)
            # 结合注意力权重和可学习权重
            combined_weights = attention_weights * learnable_weights
            combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
        else:
            combined_weights = attention_weights

        # 应用不同激活函数并加权融合
        activated_features = []
        for i, activation_type in enumerate(self.activation_types):
            activated = self._apply_activation(x, activation_type)
            activated_features.append(activated)

        # 堆叠所有激活结果 [B, num_activations, C, H, W]
        activated_stack = torch.stack(activated_features, dim=1)

        # 应用权重 [B, num_activations, 1, 1, 1]
        weights = combined_weights.unsqueeze(2)  # [B, num_activations, 1, 1, 1]

        # 加权求和 [B, C, H, W]
        output = (activated_stack * weights).sum(dim=1)

        return output


class ReinBlock(nn.Module):
    """Rein块，可以插入到现有网络中"""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation_types: List[str] = ['relu', 'sigmoid', 'tanh', 'identity'],
        learnable_weight: bool = True,
        temperature: float = 1.0,
        residual: bool = True
    ):
        super().__init__()

        self.rein = ReinModule(
            channels,
            reduction,
            activation_types,
            learnable_weight,
            temperature
        )
        self.residual = residual

        if residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征

        Returns:
            增强后的特征
        """
        rein_out = self.rein(x)

        if self.residual:
            # 可学习的残差连接权重
            output = self.residual_weight * x + (1 - self.residual_weight) * rein_out
        else:
            output = rein_out

        return output


class MultiScaleRein(nn.Module):
    """多尺度Rein模块"""

    def __init__(
        self,
        channels: int,
        scales: List[int] = [1, 2, 4],
        reduction: int = 16,
        activation_types: List[str] = ['relu', 'sigmoid', 'tanh', 'identity']
    ):
        super().__init__()

        self.scales = scales
        self.rein_modules = nn.ModuleList([
            ReinModule(channels, reduction, activation_types)
            for _ in scales
        ])

        # 尺度融合权重
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

        # 特征融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            多尺度增强后的特征 [B, C, H, W]
        """
        H, W = x.shape[-2:]
        scale_features = []

        for i, (scale, rein_module) in enumerate(zip(self.scales, self.rein_modules)):
            if scale == 1:
                # 原尺度
                scale_feat = rein_module(x)
            else:
                # 下采样
                scaled_size = (H // scale, W // scale)
                x_scaled = F.interpolate(x, size=scaled_size, mode='bilinear', align_corners=False)

                # 应用Rein
                rein_scaled = rein_module(x_scaled)

                # 上采样回原尺度
                scale_feat = F.interpolate(rein_scaled, size=(H, W), mode='bilinear', align_corners=False)

            scale_features.append(scale_feat)

        # 拼接所有尺度特征
        multi_scale_feat = torch.cat(scale_features, dim=1)

        # 特征融合
        output = self.fuse_conv(multi_scale_feat)

        return output


class ReinInserter:
    """Rein机制插入器，用于向现有网络中插入Rein模块"""

    @staticmethod
    def insert_rein_to_backbone(
        backbone: nn.Module,
        insertion_points: List[str],
        rein_config: Dict
    ) -> nn.Module:
        """向主干网络插入Rein模块

        Args:
            backbone: 主干网络
            insertion_points: 插入点列表（模块名称）
            rein_config: Rein配置

        Returns:
            插入Rein后的网络
        """
        def get_channels_from_module(module):
            """从模块推断通道数"""
            for layer in reversed(list(module.modules())):
                if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                    if isinstance(layer, nn.Conv2d):
                        return layer.out_channels
                    elif isinstance(layer, nn.BatchNorm2d):
                        return layer.num_features
            return None

        # 创建修改后的backbone副本
        modified_backbone = backbone

        for point in insertion_points:
            # 找到插入点
            modules = dict(modified_backbone.named_modules())
            if point in modules:
                target_module = modules[point]
                channels = get_channels_from_module(target_module)

                if channels is not None:
                    # 创建Rein模块
                    rein_module = ReinBlock(
                        channels=channels,
                        **rein_config
                    )

                    # 包装原模块
                    class WrappedModule(nn.Module):
                        def __init__(self, original_module, rein_module):
                            super().__init__()
                            self.original = original_module
                            self.rein = rein_module

                        def forward(self, x):
                            x = self.original(x)
                            x = self.rein(x)
                            return x

                    # 替换模块
                    wrapped = WrappedModule(target_module, rein_module)

                    # 更新backbone
                    parent_names = point.split('.')[:-1]
                    module_name = point.split('.')[-1]

                    parent = modified_backbone
                    for name in parent_names:
                        parent = getattr(parent, name)

                    setattr(parent, module_name, wrapped)

        return modified_backbone


def build_rein_module(
    rein_type: str = "basic",
    channels: int = 256,
    **kwargs
) -> nn.Module:
    """构建Rein模块

    Args:
        rein_type: Rein类型，'basic', 'block', 'multiscale'
        channels: 通道数
        **kwargs: 其他参数

    Returns:
        Rein模块
    """
    if rein_type == "basic":
        return ReinModule(channels, **kwargs)
    elif rein_type == "block":
        return ReinBlock(channels, **kwargs)
    elif rein_type == "multiscale":
        return MultiScaleRein(channels, **kwargs)
    else:
        raise ValueError(f"Unsupported rein type: {rein_type}")


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试基础Rein模块
    print("Testing Basic Rein Module...")
    rein_basic = build_rein_module("basic", channels=256)
    rein_basic = rein_basic.to(device)

    x = torch.randn(2, 256, 32, 32).to(device)
    out_basic = rein_basic(x)
    print(f"Basic Rein output shape: {out_basic.shape}")
    print(f"Basic Rein parameters: {sum(p.numel() for p in rein_basic.parameters())}")

    # 测试Rein块
    print("\nTesting Rein Block...")
    rein_block = build_rein_module("block", channels=256)
    rein_block = rein_block.to(device)

    out_block = rein_block(x)
    print(f"Rein Block output shape: {out_block.shape}")
    print(f"Rein Block parameters: {sum(p.numel() for p in rein_block.parameters())}")

    # 测试多尺度Rein
    print("\nTesting Multi-scale Rein...")
    rein_multiscale = build_rein_module("multiscale", channels=256, scales=[1, 2, 4])
    rein_multiscale = rein_multiscale.to(device)

    out_multiscale = rein_multiscale(x)
    print(f"Multi-scale Rein output shape: {out_multiscale.shape}")
    print(f"Multi-scale Rein parameters: {sum(p.numel() for p in rein_multiscale.parameters())}")
