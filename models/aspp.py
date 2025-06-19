"""
ASPP模块及其它特征增强模块
支持动态替换和参数化配置
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ASPP(nn.Module):
    """空洞空间金字塔池化"""

    def __init__(self, in_channels: int, out_channels: int = 256,
                 atrous_rates: List[int] = [1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.atrous_rates = atrous_rates

        # 1x1卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 空洞卷积分支
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates[1:]:  # 跳过rate=1
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate,
                         dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        h, w = x.shape[-2:]

        # 1x1卷积分支
        feat1 = self.conv1x1(x)

        # 空洞卷积分支
        atrous_feats = [feat1]
        for atrous_conv in self.atrous_convs:
            atrous_feats.append(atrous_conv(x))

        # 全局平均池化分支
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        atrous_feats.append(global_feat)

        # 特征融合
        concat_feat = torch.cat(atrous_feats, dim=1)
        output = self.project(concat_feat)

        return output


class LightweightASPP(nn.Module):
    """轻量化ASPP"""

    def __init__(self, in_channels: int, out_channels: int = 128,
                 atrous_rates: List[int] = [1, 3, 6]):
        super(LightweightASPP, self).__init__()

        # 使用深度可分离卷积减少参数量
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            if rate == 1:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                conv = nn.Sequential(
                    # 深度卷积
                    nn.Conv2d(in_channels, in_channels, 3, padding=rate,
                             dilation=rate, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    # 逐点卷积
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            self.atrous_convs.append(conv)

        # 特征融合
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(atrous_rates), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        atrous_feats = []
        for atrous_conv in self.atrous_convs:
            atrous_feats.append(atrous_conv(x))

        concat_feat = torch.cat(atrous_feats, dim=1)
        output = self.project(concat_feat)

        return output


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    """CBAM注意力模块"""

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


def build_feature_enhancement(name: str, in_channels: int, out_channels: int = 256, **kwargs):
    """构建特征增强模块工厂函数"""

    if name.lower() == 'aspp':
        return ASPP(in_channels, out_channels, **kwargs)
    elif name.lower() == 'lightweight_aspp':
        return LightweightASPP(in_channels, out_channels, **kwargs)
    elif name.lower() == 'cbam':
        return CBAM(in_channels, **kwargs)
    elif name.lower() == 'channel_attention':
        return ChannelAttention(in_channels, **kwargs)
    elif name.lower() == 'spatial_attention':
        return SpatialAttention(**kwargs)
    elif name.lower() == 'none' or name.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"不支持的特征增强模块: {name}")


# 使用示例
if __name__ == "__main__":
    # 测试ASPP
    x = torch.randn(2, 512, 32, 32)
    aspp = ASPP(512, 256)
    out = aspp(x)
    print(f"ASPP输出形状: {out.shape}")

    # 测试轻量化ASPP
    lightweight_aspp = LightweightASPP(512, 128)
    out = lightweight_aspp(x)
    print(f"LightweightASPP输出形状: {out.shape}")

    # 测试CBAM
    cbam = CBAM(512)
    out = cbam(x)
    print(f"CBAM输出形状: {out.shape}")
