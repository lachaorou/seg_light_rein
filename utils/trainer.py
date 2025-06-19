"""
训练器 - 负责训练循环、评估、保存检查点等
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional

from configs.config_manager import ExperimentConfig
from utils.metrics import SegmentationMetrics


class Trainer:
    """训练器"""

    def __init__(self, model: nn.Module, config: ExperimentConfig,
                 device: torch.device, exp_dir: str, logger: logging.Logger):
        self.model = model
        self.config = config
        self.device = device
        self.exp_dir = exp_dir
        self.logger = logger

        # 设置损失函数
        self.criterion = self._build_criterion()

        # 设置优化器
        self.optimizer = self._build_optimizer()

        # 设置学习率调度器
        self.scheduler = self._build_scheduler()

        # 训练统计
        self.train_losses = []
        self.val_metrics = []
        self.best_miou = 0.0

    def _build_criterion(self) -> nn.Module:
        """构建损失函数"""
        loss_type = self.config.training.loss_type.lower()

        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(ignore_index=255)
        elif loss_type == 'focal':
            return FocalLoss(ignore_index=255)
        elif loss_type == 'dice':
            return DiceLoss(ignore_index=255)
        else:
            raise ValueError(f"不支持的损失函数: {loss_type}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建优化器"""
        optimizer_type = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate

        if optimizer_type == 'sgd':
            return SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_type == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """构建学习率调度器"""
        return StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train(self, train_loader, val_loader, start_epoch: int = 0,
              total_epochs: int = None):
        """训练主循环"""
        if total_epochs is None:
            total_epochs = self.config.training.epochs

        self.logger.info(f"开始训练: {start_epoch} -> {total_epochs}")

        for epoch in range(start_epoch, total_epochs):
            # 训练一个epoch
            train_loss = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # 更新学习率
            self.scheduler.step()

            # 验证
            if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
                val_metrics = self.evaluate(val_loader)
                self.val_metrics.append(val_metrics)

                # 保存最佳模型
                current_miou = val_metrics.get('mIoU', 0)
                if current_miou > self.best_miou:
                    self.best_miou = current_miou
                    self._save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"新的最佳mIoU: {current_miou:.4f}")

                # 记录指标
                self.logger.info(
                    f"Epoch {epoch+1}/{total_epochs} - "
                    f"Loss: {train_loss:.4f} - "
                    f"mIoU: {current_miou:.4f} - "
                    f"Pixel Acc: {val_metrics.get('pixel_accuracy', 0):.4f}"
                )

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

        self.logger.info(f"训练完成！最佳mIoU: {self.best_miou:.4f}")

    def _train_epoch(self, train_loader, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        return total_loss / num_batches

    def evaluate(self, val_loader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        metrics = SegmentationMetrics(self.config.head.num_classes)

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)

                metrics.update(predictions, targets)

        return metrics.get_metrics()

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config.__dict__
        }

        # 保存常规检查点
        checkpoint_path = os.path.join(self.exp_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.exp_dir, 'checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_path)

        # 保存最新模型
        latest_path = os.path.join(self.exp_dir, 'checkpoints', 'latest.pth')
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_miou = checkpoint.get('best_miou', 0.0)

        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"已加载检查点: epoch {start_epoch}, best mIoU: {self.best_miou:.4f}")

        return start_epoch


class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, alpha=1, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 简化的Dice Loss实现
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2)

        intersection = (inputs * targets_one_hot).sum()
        union = inputs.sum() + targets_one_hot.sum()

        dice = (2.0 * intersection) / (union + 1e-8)
        return 1 - dice
