"""
简化的模型构建器 - 先确保基础功能正常工作
"""
import torch
import torch.nn as nn

class SimpleModelBuilder:
    """简化的模型构建器"""

    def __init__(self):
        pass

    def build_simple_model(self, num_classes=21):
        """构建简单的分割模型"""
        return SimpleSegModel(num_classes)

class SimpleSegModel(nn.Module):
    """简化的分割模型"""

    def __init__(self, num_classes=21):
        super(SimpleSegModel, self).__init__()

        # 编码器（模拟MobileNetV3特征提取）
        self.backbone = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Stage 2
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            # Stage 3
            nn.Conv2d(24, 40, 3, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),

            # Stage 4
            nn.Conv2d(40, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        # ASPP模块（简化版）
        self.aspp = nn.Sequential(
            nn.Conv2d(96, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 分割头
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        # 上采样层
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)  # 16x上采样

    def forward(self, x):
        # 特征提取
        features = self.backbone(x)

        # ASPP处理
        aspp_features = self.aspp(features)

        # 分割头
        output = self.head(aspp_features)

        # 上采样到原尺寸
        output = self.upsample(output)

        return output

    def get_feature_channels(self):
        """返回特征通道数"""
        return [16, 24, 40, 96, 256]

if __name__ == "__main__":
    print("测试简化的模型构建器...")

    # 创建模型构建器
    builder = SimpleModelBuilder()
    model = builder.build_simple_model(num_classes=21)

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    print("✅ 模型构建器测试成功！")
