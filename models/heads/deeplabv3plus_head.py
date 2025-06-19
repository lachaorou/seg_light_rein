"""
真实的DeepLabV3+ ASPP分割头实现
支持多尺度空洞卷积和图像级特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ASPPConv(nn.Sequential):
    """ASPP卷积模块"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ASPP全局平均池化模块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP)模块"""

    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256):
        super().__init__()
        modules = []

        # 1x1卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        # 不同膨胀率的3x3卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 全局平均池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 融合后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabV3PlusHead(nn.Module):
    """DeepLabV3+分割头"""

    def __init__(
        self,
        in_channels: int,
        low_level_channels: int,
        num_classes: int,
        aspp_dilate: List[int] = [12, 24, 36],
        aspp_out_channels: int = 256,
        low_level_channels_project: int = 48,
        dropout_ratio: float = 0.1
    ):
        super().__init__()

        # ASPP模块
        self.aspp = ASPP(in_channels, aspp_dilate, aspp_out_channels)

        # 低级特征投影
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_channels_project, 1, bias=False),
            nn.BatchNorm2d(low_level_channels_project),
            nn.ReLU()
        )

        # 特征融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(aspp_out_channels + low_level_channels_project, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_ratio)
        )

        # 分类器
        self.classifier = nn.Conv2d(256, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, high_level_features: torch.Tensor, low_level_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_level_features: 高级特征 (通常是backbone的最后一层特征)
            low_level_features: 低级特征 (通常是backbone的浅层特征)

        Returns:
            分割预测结果
        """
        # ASPP处理高级特征
        aspp_out = self.aspp(high_level_features)

        # 上采样ASPP输出到低级特征的尺寸
        low_level_size = low_level_features.shape[-2:]
        aspp_out = F.interpolate(aspp_out, size=low_level_size, mode='bilinear', align_corners=False)

        # 处理低级特征
        low_level_out = self.low_level_conv(low_level_features)

        # 特征融合
        fused = torch.cat([aspp_out, low_level_out], dim=1)
        fused = self.fuse_conv(fused)

        # 分类
        out = self.classifier(fused)

        return out


class DeepLabV3Head(nn.Module):
    """DeepLabV3分割头（不包含低级特征融合）"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        aspp_dilate: List[int] = [12, 24, 36],
        aspp_out_channels: int = 256,
        dropout_ratio: float = 0.1
    ):
        super().__init__()

        # ASPP模块
        self.aspp = ASPP(in_channels, aspp_dilate, aspp_out_channels)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2d(aspp_out_channels, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征

        Returns:
            分割预测结果
        """
        x = self.aspp(x)
        x = self.classifier(x)
        return x


def build_aspp_head(
    head_type: str = "deeplabv3plus",
    in_channels: int = 960,
    low_level_channels: int = 24,
    num_classes: int = 21,
    **kwargs
):
    """构建ASPP分割头

    Args:
        head_type: 头类型，'deeplabv3plus' 或 'deeplabv3'
        in_channels: 输入通道数
        low_level_channels: 低级特征通道数（仅deeplabv3plus需要）
        num_classes: 类别数
        **kwargs: 其他参数

    Returns:
        分割头模型
    """
    if head_type == "deeplabv3plus":
        return DeepLabV3PlusHead(
            in_channels=in_channels,
            low_level_channels=low_level_channels,
            num_classes=num_classes,
            **kwargs
        )
    elif head_type == "deeplabv3":
        return DeepLabV3Head(
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported head type: {head_type}")


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试DeepLabV3+头
    print("Testing DeepLabV3+ Head...")
    head_v3plus = build_aspp_head("deeplabv3plus", in_channels=960, low_level_channels=24)
    head_v3plus = head_v3plus.to(device)

    # 模拟特征
    high_level_feat = torch.randn(2, 960, 14, 14).to(device)  # 高级特征
    low_level_feat = torch.randn(2, 24, 56, 56).to(device)   # 低级特征

    out_v3plus = head_v3plus(high_level_feat, low_level_feat)
    print(f"DeepLabV3+ output shape: {out_v3plus.shape}")
    print(f"DeepLabV3+ parameters: {sum(p.numel() for p in head_v3plus.parameters()) / 1e6:.2f}M")

    # 测试DeepLabV3头
    print("\nTesting DeepLabV3 Head...")
    head_v3 = build_aspp_head("deeplabv3", in_channels=960)
    head_v3 = head_v3.to(device)

    out_v3 = head_v3(high_level_feat)
    print(f"DeepLabV3 output shape: {out_v3.shape}")
    print(f"DeepLabV3 parameters: {sum(p.numel() for p in head_v3.parameters()) / 1e6:.2f}M")
