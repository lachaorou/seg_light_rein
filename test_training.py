"""
测试训练流程 - 简化版本
"""
import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 创建简单的数据集类
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        label = torch.randint(0, 21, (224, 224))
        return image, label

# 简化的分割模型
class SimpleSegModel(nn.Module):
    def __init__(self, num_classes=21):
        super(SimpleSegModel, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, num_classes, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def test_training_pipeline():
    print("🚀 开始测试训练流程...")

    # 1. 创建模型
    model = SimpleSegModel(num_classes=21)
    print(f"✅ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 创建数据加载器
    dataset = DummyDataset(size=20)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✅ 数据加载器创建成功，数据集大小: {len(dataset)}")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print("✅ 损失函数和优化器创建成功")

    # 4. 测试一个训练步骤
    model.train()
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= 2:  # 只测试2个batch
            break

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Batch {batch_idx}: loss = {loss.item():.4f}")

    print("✅ 训练流程测试成功！")

    # 5. 测试评估
    model.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(dataloader))
        test_outputs = model(test_images)
        test_loss = criterion(test_outputs, test_labels)

        # 计算准确率（简化版）
        pred = torch.argmax(test_outputs, dim=1)
        accuracy = (pred == test_labels).float().mean()

        print(f"✅ 评估测试成功: loss = {test_loss.item():.4f}, accuracy = {accuracy.item():.4f}")

    return model

if __name__ == "__main__":
    try:
        model = test_training_pipeline()
        print("\n🎉 训练系统测试完全成功！")
        print("现在可以开始迁移更复杂的模型和配置系统。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
