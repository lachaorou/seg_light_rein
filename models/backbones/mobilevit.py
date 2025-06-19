"""
MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
Paper: https://arxiv.org/abs/2110.02178

This implementation provides MobileViT-S/XS variants for semantic segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def conv_3x3_bn(inp: int, oup: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    """3x3 convolution with batch normalization"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        nn.SiLU(inplace=True)
    )


def conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    """1x1 convolution with batch normalization"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    """Inverted Residual Block (MBConv)"""
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(conv_1x1_bn(inp, hidden_dim))
        layers.extend([
            # dw
            conv_3x3_bn(hidden_dim, hidden_dim, stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PreNorm(nn.Module):
    """Pre-normalization wrapper"""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


def rearrange(tensor, pattern, **axes_lengths):
    """Simple rearrange function (lightweight version of einops)"""
    # This is a simplified version - in production, use einops.rearrange
    if 'b p n (h d) -> b p h n d' in pattern:
        b, p, n, hd = tensor.shape
        h = axes_lengths['h']
        d = hd // h
        return tensor.reshape(b, p, n, h, d).transpose(-2, -1)
    elif 'b p h n d -> b p n (h d)' in pattern:
        b, p, h, n, d = tensor.shape
        return tensor.transpose(-2, -1).reshape(b, p, n, h * d)
    return tensor


class Transformer(nn.Module):
    """Transformer block"""
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileViTBlock(nn.Module):
    """MobileViT Block"""
    def __init__(self, dim: int, depth: int, channel: int, kernel_size: int, patch_size: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size

        self.conv1 = conv_1x1_bn(channel, channel)
        self.conv2 = conv_3x3_bn(channel, dim, kernel_size)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_3x3_bn(dim, channel, kernel_size)
        self.conv4 = conv_1x1_bn(2 * channel, channel)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)

        # Ensure spatial dimensions match before concatenation
        if x.shape[2:] != y.shape[2:]:
            x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


def rearrange(tensor, pattern, **axes_lengths):
    """Rearrange tensor according to pattern"""
    if 'b d (h ph) (w pw) -> b (ph pw) (h w) d' in pattern:
        b, d, h_pw, w_pw = tensor.shape
        ph, pw = axes_lengths['ph'], axes_lengths['pw']
        h, w = h_pw // ph, w_pw // pw
        x = tensor.reshape(b, d, h, ph, w, pw)
        x = x.permute(0, 3, 5, 2, 4, 1)  # b, ph, pw, h, w, d
        x = x.reshape(b, ph * pw, h * w, d)
        return x
    elif 'b (ph pw) (h w) d -> b d (h ph) (w pw)' in pattern:
        b, ph_pw, h_w, d = tensor.shape
        h, w = axes_lengths['h'], axes_lengths['w']
        ph, pw = axes_lengths['ph'], axes_lengths['pw']
        x = tensor.reshape(b, ph, pw, h, w, d)
        x = x.permute(0, 5, 3, 1, 4, 2)  # b, d, h, ph, w, pw
        x = x.reshape(b, d, h * ph, w * pw)
        return x
    return tensor


class MobileViT(nn.Module):
    """
    MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer

    Args:
        num_classes: Number of output classes (not used for segmentation backbone)
        dims: Channel dimensions for each stage
        depths: Number of transformer layers for each ViT block
        channels: Channel configurations for each stage
    """

    def __init__(self,
                 variant: str = 'xs',
                 pretrained: bool = False,
                 **kwargs):
        super(MobileViT, self).__init__()

        # Model configurations
        configs = {
            'xs': {
                'dims': [96, 120, 144],
                'depths': [2, 4, 3],
                'channels': [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
                'expansion_ratios': [1, 4, 3, 3, 3, 3, 6, 6, 6, 6],
                'strides': [1, 2, 1, 2, 1, 1, 2, 1, 1, 1]
            },
            's': {
                'dims': [144, 192, 240],
                'depths': [2, 4, 3],
                'channels': [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
                'expansion_ratios': [1, 4, 3, 3, 3, 3, 6, 6, 6, 6],
                'strides': [1, 2, 1, 2, 1, 1, 2, 1, 1, 1]
            }
        }

        assert variant in configs, f"Variant {variant} not supported"
        config = configs[variant]

        self.dims = config['dims']
        self.depths = config['depths']
        channels = config['channels']
        expansion_ratios = config['expansion_ratios']
        strides = config['strides']

        # First conv layer
        self.conv1 = conv_3x3_bn(3, channels[0], 2)

        # Build stages
        self.layers = nn.ModuleList()

        # Stage 1: MobileNet blocks
        self.layers.append(InvertedResidual(channels[0], channels[1], strides[0], expansion_ratios[0]))

        # Stage 2: MobileNet blocks
        blocks = []
        for i in range(1, 3):
            blocks.append(InvertedResidual(channels[i], channels[i+1], strides[i], expansion_ratios[i]))
        self.layers.append(nn.Sequential(*blocks))

        # Stage 3: MobileNet blocks
        blocks = []
        for i in range(3, 5):
            blocks.append(InvertedResidual(channels[i], channels[i+1], strides[i], expansion_ratios[i]))
        self.layers.append(nn.Sequential(*blocks))

        # Stage 4: MobileNet + MobileViT blocks
        blocks = []
        blocks.append(InvertedResidual(channels[5], channels[6], strides[5], expansion_ratios[5]))
        blocks.append(MobileViTBlock(
            dim=self.dims[0], depth=self.depths[0], channel=channels[6],
            kernel_size=3, patch_size=2, mlp_dim=int(self.dims[0] * 2)
        ))
        self.layers.append(nn.Sequential(*blocks))

        # Stage 5: MobileNet + MobileViT blocks
        blocks = []
        blocks.append(InvertedResidual(channels[6], channels[7], strides[6], expansion_ratios[6]))
        blocks.append(MobileViTBlock(
            dim=self.dims[1], depth=self.depths[1], channel=channels[7],
            kernel_size=3, patch_size=2, mlp_dim=int(self.dims[1] * 4)
        ))
        self.layers.append(nn.Sequential(*blocks))

        # Stage 6: MobileNet + MobileViT blocks
        blocks = []
        blocks.append(InvertedResidual(channels[7], channels[8], strides[7], expansion_ratios[7]))
        blocks.append(MobileViTBlock(
            dim=self.dims[2], depth=self.depths[2], channel=channels[8],
            kernel_size=3, patch_size=2, mlp_dim=int(self.dims[2] * 4)
        ))
        self.layers.append(nn.Sequential(*blocks))

        # Final conv
        self.conv_1x1_exp = conv_1x1_bn(channels[8], channels[9])

        # Store feature channels for segmentation
        self.feature_channels = {
            'stage1': channels[1],  # 1/2
            'stage2': channels[3],  # 1/4
            'stage3': channels[5],  # 1/8
            'stage4': channels[6],  # 1/16
            'stage5': channels[7],  # 1/32
            'stage6': channels[9],  # 1/32
        }

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load pretrained weights if available"""
        # TODO: Implement pretrained weight loading
        print("Warning: Pretrained weights not implemented for MobileViT")
        pass

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features

        Returns:
            Dictionary containing features at different scales:
            - low_level_features: 1/4 scale (for DeepLabV3+ low-level features)
            - high_level_features: 1/32 scale (for ASPP)
            - stage1: 1/2 scale
            - stage2: 1/4 scale
            - stage3: 1/8 scale
            - stage4: 1/16 scale
            - stage5: 1/32 scale
            - stage6: 1/32 scale
        """
        features = {}

        # Initial conv
        x = self.conv1(x)  # 1/2
        features['stage0'] = x

        # Stage 1
        x = self.layers[0](x)  # 1/2
        features['stage1'] = x

        # Stage 2
        x = self.layers[1](x)  # 1/4
        features['stage2'] = x
        features['low_level_features'] = x  # For DeepLabV3+

        # Stage 3
        x = self.layers[2](x)  # 1/8
        features['stage3'] = x

        # Stage 4 (with MobileViT)
        x = self.layers[3](x)  # 1/16
        features['stage4'] = x

        # Stage 5 (with MobileViT)
        x = self.layers[4](x)  # 1/32
        features['stage5'] = x

        # Stage 6 (with MobileViT)
        x = self.layers[5](x)  # 1/32
        x = self.conv_1x1_exp(x)
        features['stage6'] = x
        features['high_level_features'] = x  # For ASPP

        return features

    def get_feature_channels(self) -> Dict[str, int]:
        """Return the number of channels for each feature level"""
        return self.feature_channels.copy()


def mobilevit_xs(pretrained: bool = False, **kwargs) -> MobileViT:
    """MobileViT-XS model"""
    return MobileViT(variant='xs', pretrained=pretrained, **kwargs)


def mobilevit_s(pretrained: bool = False, **kwargs) -> MobileViT:
    """MobileViT-S model"""
    return MobileViT(variant='s', pretrained=pretrained, **kwargs)


if __name__ == "__main__":
    # Test the model
    model = mobilevit_xs()
    x = torch.randn(2, 3, 512, 512)
    features = model(x)

    print("MobileViT-XS Feature Maps:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    print(f"\nFeature channels: {model.get_feature_channels()}")

    # Test with different input size
    x = torch.randn(1, 3, 224, 224)
    features = model(x)

    print("\nMobileViT-XS Feature Maps (224x224):")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
