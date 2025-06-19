"""
真实的MobileNetV3主干网络实现
基于官方PyTorch实现，支持Small和Large两个版本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """确保通道数能被divisor整除"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    """SE注意力模块"""

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: torch.Tensor, inplace: bool) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return torch.sigmoid_(scale) if inplace else torch.sigmoid(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:
    """倒残差块配置"""

    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int = 1,
        width_mult: float = 1.0,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """MobileNetV3倒残差块"""

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: nn.Module, se_layer: nn.Module = SqueezeExcitation):
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                nn.Conv2d(cnf.input_channels, cnf.expanded_channels, 1, bias=False)
            )
            layers.append(norm_layer(cnf.expanded_channels))
            layers.append(activation_layer(inplace=True))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            nn.Conv2d(
                cnf.expanded_channels,
                cnf.expanded_channels,
                cnf.kernel,
                stride,
                (cnf.kernel - 1) // 2 * cnf.dilation,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                bias=False,
            )
        )
        layers.append(norm_layer(cnf.expanded_channels))
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))
        layers.append(activation_layer(inplace=True))

        # project
        layers.append(
            nn.Conv2d(cnf.expanded_channels, cnf.out_channels, 1, bias=False)
        )
        layers.append(norm_layer(cnf.out_channels))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    """MobileNetV3主干网络"""

    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        width_mult: float = 1.0,
        reduce_divider: int = 1,
        dilate_scale: int = 8,
        norm_layer: Optional[nn.Module] = None,
        **kwargs: Any,
    ):
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, List)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # 第一个卷积层
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            nn.Conv2d(3, firstconv_output_channels, 3, 2, 1, bias=False)
        )
        layers.append(norm_layer(firstconv_output_channels))
        layers.append(nn.Hardswish(inplace=True))

        # 倒残差块
        total_stage_blocks = len(inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage_block_id += 1
            if dilate_scale > 1 and stage_block_id > total_stage_blocks // 2:
                if stage_block_id == total_stage_blocks // 2 + 1:
                    cnf.stride = 1  # 第一个膨胀块不降采样
                cnf.dilation = dilate_scale
            layers.append(InvertedResidual(cnf, norm_layer))

        # 最后的卷积层
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel // reduce_divider
        layers.append(
            nn.Conv2d(lastconv_input_channels, lastconv_output_channels, 1, bias=False)
        )
        layers.append(norm_layer(lastconv_output_channels))
        layers.append(nn.Hardswish(inplace=True))

        self.features = nn.Sequential(*layers)
        self._out_channels = lastconv_output_channels

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    @property
    def out_channels(self) -> int:
        return self._out_channels


def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    """构建MobileNetV3配置"""
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1),
            bneck_conf(112, 5, 672, 160, True, "HS", 2),  # C4
            bneck_conf(160, 5, 960, 160, True, "HS", 1),
            bneck_conf(160, 5, 960, 160, True, "HS", 1),
        ]
        last_channel = adjust_channels(960)
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1),
            bneck_conf(48, 5, 288, 96, True, "HS", 2),  # C4
            bneck_conf(96, 5, 576, 96, True, "HS", 1),
            bneck_conf(96, 5, 576, 96, True, "HS", 1),
        ]
        last_channel = adjust_channels(576)
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return MobileNetV3(
        inverted_residual_setting,
        last_channel,
        width_mult=width_mult,
        reduce_divider=reduce_divider,
        dilate_scale=dilation,
        **kwargs,
    )


# 为了兼容性，需要导入partial
from functools import partial


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """构建MobileNetV3-Large模型"""
    model = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    if pretrained:
        # 这里可以加载预训练权重
        pass
    return model


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """构建MobileNetV3-Small模型"""
    model = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    if pretrained:
        # 这里可以加载预训练权重
        pass
    return model


# 便于导出的构建函数
def build_mobilenet_v3(variant: str = "small", **kwargs) -> MobileNetV3:
    """构建MobileNetV3模型

    Args:
        variant: 'small' 或 'large'
        **kwargs: 其他参数

    Returns:
        MobileNetV3模型
    """
    if variant == "small":
        return mobilenet_v3_small(**kwargs)
    elif variant == "large":
        return mobilenet_v3_large(**kwargs)
    else:
        raise ValueError(f"Unsupported variant: {variant}")


if __name__ == "__main__":
    # 测试代码
    model_small = build_mobilenet_v3("small")
    model_large = build_mobilenet_v3("large")

    x = torch.randn(1, 3, 224, 224)

    print("MobileNetV3-Small:")
    y_small = model_small(x)
    print(f"Output shape: {y_small.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_small.parameters()) / 1e6:.2f}M")

    print("\nMobileNetV3-Large:")
    y_large = model_large(x)
    print(f"Output shape: {y_large.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_large.parameters()) / 1e6:.2f}M")
