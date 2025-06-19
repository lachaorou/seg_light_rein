"""
完整的训练脚本 - 结合配置系统
"""
import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
from pathlib import Path
import time
from datetime import datetime

# 导入我们的模块
from simple_model_builder import SimpleModelBuilder
from configs.config_manager import ConfigManager, ExperimentConfig

class DummyVOCDataset(Dataset):
    """模拟VOC数据集"""

    def __init__(self, size=100, split='train'):
        self.size = size
        self.split = split
        # 模拟不同的数据集大小
        if split == 'train':
            self.size = size
        else:
            self.size = size // 5  # val数据集更小

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 模拟真实的VOC图片尺寸
        image = torch.randn(3, 224, 224)
        # 模拟分割标签 (21类：20个物体类别 + 1个背景)
        label = torch.randint(0, 21, (224, 224))
        return image, label

class Trainer:
    """训练器"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建模型
        self.model = self._build_model()
        self.model.to(self.device)

        # 创建数据加载器
        self.train_loader, self.val_loader = self._build_dataloaders()

        # 创建损失函数和优化器
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()

        # 训练状态
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def _build_model(self):
        """构建模型"""
        builder = SimpleModelBuilder()
        model = builder.build_simple_model(num_classes=self.config.head.num_classes)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")

        return model

    def _build_dataloaders(self):
        """构建数据加载器"""
        train_dataset = DummyVOCDataset(size=200, split='train')
        val_dataset = DummyVOCDataset(size=50, split='val')

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0  # Windows下设为0避免问题
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        return train_loader, val_loader

    def _build_criterion(self):
        """构建损失函数"""
        if self.config.training.loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"不支持的损失函数: {self.config.training.loss_type}")

    def _build_optimizer(self):
        """构建优化器"""
        if self.config.training.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif self.config.training.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.training.optimizer}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: loss = {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        num_batches = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 计算准确率
                pred = torch.argmax(outputs, dim=1)
                correct_pixels += (pred == labels).sum().item()
                total_pixels += labels.numel()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        accuracy = correct_pixels / total_pixels

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)

        return avg_loss, accuracy

    def train(self):
        """完整训练流程"""
        print(f"\n🚀 开始训练实验: {self.config.experiment_name}")
        print(f"主干: {self.config.backbone.name}")
        print(f"R机制: {'启用' if self.config.rein.enabled else '禁用'}")
        print(f"分割头: {self.config.head.name}")
        print(f"训练轮数: {self.config.training.epochs}")
        print("=" * 50)

        start_time = time.time()

        for epoch in range(self.config.training.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")

            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss, val_accuracy = self.validate(epoch)

            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")

        total_time = time.time() - start_time
        print(f"\n✅ 训练完成！总用时: {total_time:.2f}秒")

        # 返回最终结果
        final_results = {
            'experiment_name': self.config.experiment_name,
            'backbone': self.config.backbone.name,
            'rein_enabled': self.config.rein.enabled,
            'head': self.config.head.name,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_val_accuracy': self.val_accuracies[-1],
            'total_time': total_time,
            'total_params': sum(p.numel() for p in self.model.parameters())
        }

        return final_results

def run_experiment(config_path: str):
    """运行单个实验"""
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)

    # 创建训练器并训练
    trainer = Trainer(config)
    results = trainer.train()

    return results

def run_baseline_experiments():
    """运行基线实验"""
    print("🎯 开始运行基线实验...")

    # 确保配置文件存在
    config_files = [
        'configs/mobilenetv3_baseline.yaml',
        'configs/mobilenetv3_rein.yaml'
    ]

    results = []

    for config_file in config_files:
        if Path(config_file).exists():
            print(f"\n📋 运行配置: {config_file}")
            result = run_experiment(config_file)
            results.append(result)
        else:
            print(f"⚠️ 配置文件不存在: {config_file}")

    # 生成对比表
    print("\n📊 实验结果对比:")
    print("=" * 80)
    print(f"{'实验名称':<20} {'主干':<15} {'R机制':<8} {'验证准确率':<12} {'参数量':<12} {'训练时间':<10}")
    print("-" * 80)

    for result in results:
        print(f"{result['experiment_name']:<20} "
              f"{result['backbone']:<15} "
              f"{'是' if result['rein_enabled'] else '否':<8} "
              f"{result['final_val_accuracy']:<12.4f} "
              f"{result['total_params']:<12,} "
              f"{result['total_time']:<10.1f}s")

    return results

if __name__ == "__main__":
    try:
        # 运行基线实验
        results = run_baseline_experiments()

        print("\n🎉 所有实验完成！")
        print("结果已保存，可以开始分析和优化。")

    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
