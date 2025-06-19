"""
Rein机制模块实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ReinMechanism(nn.Module):
    """Rein机制实现"""

    def __init__(self, insertion_points: List[int] = None,
                 rein_dim: int = 64, num_tokens: int = 8,
                 merge_ratio: float = 0.5, **kwargs):
        super(ReinMechanism, self).__init__()

        self.insertion_points = insertion_points or []
        self.rein_dim = rein_dim
        self.num_tokens = num_tokens
        self.merge_ratio = merge_ratio

        # Token生成器
        self.token_generator = nn.Parameter(torch.randn(1, num_tokens, rein_dim))

        # Token处理模块
        self.token_processor = nn.MultiheadAttention(
            embed_dim=rein_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(rein_dim, rein_dim, 3, padding=1),
            nn.BatchNorm2d(rein_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(rein_dim, rein_dim, 1),
            nn.Sigmoid()
        )

        print(f"[Rein] 初始化完成 - tokens: {num_tokens}, dim: {rein_dim}")

    def forward(self, features):
        """应用Rein机制到特征"""
        if not self.insertion_points:
            # 如果没有指定插入点，直接返回原特征
            return features

        # 获取特征尺寸
        B, C, H, W = features.shape

        # 调整token维度到匹配特征通道
        if self.rein_dim != C:
            # 动态调整token维度
            token_proj = nn.Linear(self.rein_dim, C).to(features.device)
            tokens = token_proj(self.token_generator.expand(B, -1, -1))
        else:
            tokens = self.token_generator.expand(B, -1, -1)

        # Token自注意力处理
        processed_tokens, _ = self.token_processor(tokens, tokens, tokens)

        # 将token映射到空间维度
        token_spatial = processed_tokens.mean(dim=1, keepdim=True)  # [B, 1, C]
        token_spatial = token_spatial.unsqueeze(-1).expand(B, C, H, W)  # [B, C, H, W]

        # 特征增强
        if self.rein_dim == C:
            attention_map = self.feature_fusion(token_spatial)
            enhanced_features = features * attention_map + features
        else:
            # 简化版本：直接加权融合
            attention_weight = torch.sigmoid(token_spatial.mean(dim=1, keepdim=True))
            enhanced_features = features * (1 + attention_weight * self.merge_ratio)

        return enhanced_features

    def get_token_info(self):
        """获取token信息"""
        return {
            'num_tokens': self.num_tokens,
            'token_dim': self.rein_dim,
            'insertion_points': self.insertion_points
        }


class TokenMerging(nn.Module):
    """Token Merging机制"""

    def __init__(self, merge_ratio: float = 0.5, **kwargs):
        super(TokenMerging, self).__init__()
        self.merge_ratio = merge_ratio

        print(f"[TokenMerging] 初始化完成 - merge_ratio: {merge_ratio}")

    def forward(self, features):
        """Token合并"""
        B, C, H, W = features.shape

        # 简化的token合并：空间维度降采样
        if H > 1 and W > 1:
            # 使用平均池化进行token合并
            merged = F.adaptive_avg_pool2d(features, (H//2, W//2))
            # 上采样回原始尺寸
            merged = F.interpolate(merged, size=(H, W), mode='bilinear', align_corners=False)

            # 融合原始特征和合并特征
            output = features * (1 - self.merge_ratio) + merged * self.merge_ratio
        else:
            output = features

        return output


class AttentionMechanism(nn.Module):
    """注意力机制"""

    def __init__(self, attention_type: str = 'channel', **kwargs):
        super(AttentionMechanism, self).__init__()
        self.attention_type = attention_type

        print(f"[Attention] 初始化完成 - type: {attention_type}")

    def forward(self, features):
        """应用注意力机制"""
        if self.attention_type == 'channel':
            return self._channel_attention(features)
        elif self.attention_type == 'spatial':
            return self._spatial_attention(features)
        else:
            return features

    def _channel_attention(self, x):
        """通道注意力"""
        B, C, H, W = x.shape

        # 全局平均池化
        avg_pool = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]

        # 简化的通道注意力
        attention = torch.sigmoid(avg_pool)

        return x * attention

    def _spatial_attention(self, x):
        """空间注意力"""
        # 通道维度的平均和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并生成注意力图
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(F.conv2d(attention_input,
                                          torch.ones(1, 2, 7, 7).to(x.device) / 49,
                                          padding=3))

        return x * attention


def build_rein_mechanism(mechanism_type: str, **kwargs):
    """构建R机制"""
    if mechanism_type.lower() == 'rein':
        return ReinMechanism(**kwargs)
    elif mechanism_type.lower() == 'token_merging':
        return TokenMerging(**kwargs)
    elif mechanism_type.lower() == 'attention':
        return AttentionMechanism(**kwargs)
    else:
        raise ValueError(f"不支持的R机制类型: {mechanism_type}")


# 测试代码
if __name__ == "__main__":
    # 测试Rein机制
    rein = ReinMechanism(rein_dim=64, num_tokens=8)
    x = torch.randn(2, 256, 32, 32)
    output = rein(x)
    print(f"Rein输出形状: {output.shape}")

    # 测试Token Merging
    token_merging = TokenMerging(merge_ratio=0.3)
    output = token_merging(x)
    print(f"TokenMerging输出形状: {output.shape}")

    # 测试注意力机制
    attention = AttentionMechanism(attention_type='channel')
    output = attention(x)
    print(f"Attention输出形状: {output.shape}")
