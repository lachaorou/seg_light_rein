"""
简单的MobileNetV3测试
"""
import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn

# 直接在这里定义简化的MobileNetV3
class SimpleMobileNet(nn.Module):
    def __init__(self, num_classes=21):
        super(SimpleMobileNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    print("开始测试简化的MobileNet...")

    # 创建模型
    model = SimpleMobileNet(num_classes=21)

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    print("✅ 基础模型测试成功！")
