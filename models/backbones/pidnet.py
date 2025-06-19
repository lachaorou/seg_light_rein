"""
PIDNet主干网络 - 高效双分支架构
专为实时语义分割设计的轻量化网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class BasicBlock(nn.Module):
    """PIDNet基础残差块"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    """PIDNet Bottleneck块"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class PIDNet(nn.Module):
    """
    PIDNet主干网络
    双分支架构：Perception分支 + Instance分支 + Detail分支
    """

    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128,
                 head_planes=128, augment=True):
        super(PIDNet, self).__init__()

        self.augment = augment
        self.num_classes = num_classes

        # Perception分支 (语义感知)
        self.conv1 = nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        # Instance分支 (实例分割)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        # Detail分支 (边界细节)
        self.down2 = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )

        # 特征融合
        self.layer3_ = self._make_layer(block, planes * 2, planes * 2, 2)
        self.layer4_ = self._make_layer(block, planes * 2, planes * 2, 2)

        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1)

        # SPP模块
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        # 分类头
        if self.augment:
            self.seghead_p = SegmentHead(planes * 4, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)

        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        self.init_weights()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播"""
        # 输入分辨率：H x W
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        # Stem
        x = self.conv1(x)  # H/2 x W/2
        x = self.bn1(x)
        x = self.relu(x)

        # 第一层
        x = self.layer1(x)  # H/2 x W/2, 64

        # Detail分支
        x2 = self.layer2(x)  # H/4 x W/4, 128
        x_detail = self.down2(x)  # H/4 x W/4, 256

        # Perception分支
        x3 = self.layer3(x2)  # H/8 x W/8, 256
        x_detail = self.down3(x_detail)  # H/8 x W/8, 256

        # Instance分支开始
        x3_ = self.compression3(x3)  # H/8 x W/8, 128
        x3_ = self.layer3_(x3_)  # H/8 x W/8, 128

        x4 = self.layer4(x3)  # H/16 x W/16, 512
        x4_ = self.compression4(x4)  # H/16 x W/16, 128
        x4_ = self.layer4_(x4_)  # H/16 x W/16, 128

        # 高层特征
        x5_ = self.layer5_(x4_)  # H/16 x W/16, 512
        x5 = self.layer5(x4)  # H/16 x W/16, 2048

        # SPP处理
        x = self.spp(x5)  # H/16 x W/16, 256

        # 上采样并融合
        x = F.interpolate(x, size=(height_output, width_output),
                         mode='bilinear', align_corners=True)  # H/8 x W/8, 256

        x_ = F.interpolate(x5_, size=(height_output, width_output),
                          mode='bilinear', align_corners=True)  # H/8 x W/8, 512

        # 特征融合
        x = x + x_[:, :x.shape[1], :, :]  # H/8 x W/8, 256

        # 与Detail分支融合
        x = x + x_detail  # H/8 x W/8, 256

        if self.training and self.augment:
            # 训练时的辅助输出
            x_extra_p = self.seghead_p(x3)
            x_extra_d = self.seghead_d(x_detail)

            x_final = self.final_layer(x)

            return {
                'out': x_final,
                'aux_p': x_extra_p,
                'aux_d': x_extra_d,
                'features': [x2, x3, x4, x5]  # 多尺度特征
            }
        else:
            # 推理时的主输出
            x_final = self.final_layer(x)
            return {
                'out': x_final,
                'features': [x2, x3, x4, x5]  # 多尺度特征
            }

class DAPPM(nn.Module):
    """深度聚合金字塔池化模块"""

    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.process1 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]

        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out

class SegmentHead(nn.Module):
    """分割头"""

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(SegmentHead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=True)

        return out

def pidnet_s(num_classes=19, pretrained=False):
    """PIDNet-S (Small)"""
    model = PIDNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                   planes=32, spp_planes=128, head_planes=128)

    if pretrained:
        # 这里可以加载预训练权重
        pass

    return model

def pidnet_m(num_classes=19, pretrained=False):
    """PIDNet-M (Medium)"""
    model = PIDNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                   planes=64, spp_planes=128, head_planes=128)

    if pretrained:
        # 这里可以加载预训练权重
        pass

    return model

def pidnet_l(num_classes=19, pretrained=False):
    """PIDNet-L (Large)"""
    model = PIDNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes,
                   planes=64, spp_planes=128, head_planes=128)

    if pretrained:
        # 这里可以加载预训练权重
        pass

    return model

# 导出接口
__all__ = ['PIDNet', 'pidnet_s', 'pidnet_m', 'pidnet_l']
