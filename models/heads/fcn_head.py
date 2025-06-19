"""
FCN分割头实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import build_feature_enhancement


class FCNHead(nn.Module):
    """FCN分割头"""

    def __init__(self, in_channels: int, num_classes: int,
                 dropout_ratio: float = 0.1, feature_enhancement: str = 'aspp'):
        super(FCNHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_enhancement_name = feature_enhancement

        # 特征增强模块
        if feature_enhancement.lower() != 'none':
            self.feature_enhancement = build_feature_enhancement(
                feature_enhancement, in_channels, out_channels=256
            )
            classifier_in_channels = 256
        else:
            self.feature_enhancement = nn.Identity()
            classifier_in_channels = in_channels

        # 分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(classifier_in_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),
            nn.Conv2d(512, num_classes, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 特征增强
        x = self.feature_enhancement(x)

        # 分类
        x = self.classifier(x)

        # 上采样到原始尺寸
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)

        return x

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class ASPPHead(nn.Module):
    """ASPP分割头"""

    def __init__(self, in_channels: int, num_classes: int,
                 dropout_ratio: float = 0.1):
        super(ASPPHead, self).__init__()

        # ASPP模块
        from models.aspp import ASPP
        self.aspp = ASPP(in_channels, out_channels=256)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),
            nn.Conv2d(256, num_classes, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.aspp(x)
        x = self.classifier(x)

        # 上采样到原始尺寸
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)

        return x

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class DeepLabV3PlusHead(nn.Module):
    """DeepLabV3+风格的分割头"""

    def __init__(self, in_channels: int, low_level_channels: int,
                 num_classes: int, dropout_ratio: float = 0.1):
        super(DeepLabV3PlusHead, self).__init__()

        # ASPP模块
        from models.aspp import ASPP
        self.aspp = ASPP(in_channels, out_channels=256)

        # 低级特征处理
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 特征融合和分类
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),
            nn.Conv2d(256, num_classes, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, high_level_feat, low_level_feat):
        # 高级特征处理
        high_level_feat = self.aspp(high_level_feat)

        # 上采样高级特征
        high_level_feat = F.interpolate(
            high_level_feat, size=low_level_feat.shape[-2:],
            mode='bilinear', align_corners=False
        )

        # 低级特征处理
        low_level_feat = self.low_level_conv(low_level_feat)

        # 特征融合
        concat_feat = torch.cat([high_level_feat, low_level_feat], dim=1)

        # 分类
        output = self.classifier(concat_feat)

        # 上采样到原始尺寸
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)

        return output

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# 测试代码
if __name__ == "__main__":
    # 测试FCNHead
    fcn_head = FCNHead(960, 21, feature_enhancement='aspp')
    x = torch.randn(2, 960, 64, 64)
    output = fcn_head(x)
    print(f"FCNHead输出形状: {output.shape}")

    # 测试ASPPHead
    aspp_head = ASPPHead(960, 21)
    output = aspp_head(x)
    print(f"ASPPHead输出形状: {output.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in fcn_head.parameters())
    print(f"FCNHead参数量: {total_params:,}")
