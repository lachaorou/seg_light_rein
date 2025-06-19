"""
完整的训练器
整合模型、数据、损失、优化器等组件
支持多种训练策略和评估指标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm
import logging
import yaml

# 导入自定义模块
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.unified_model_builder import SegmentationModel, get_model_info
from datasets.voc_dataset import VOCDataModule
from utils.metrics import SegmentationMetrics
from utils.logger import setup_logger


class MultiLoss(nn.Module):
    """多种损失函数组合"""

    def __init__(
        self,
        loss_types: List[str] = ['cross_entropy'],
        loss_weights: List[float] = [1.0],
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.ignore_index = ignore_index
        self.class_weights = class_weights

        # 初始化损失函数
        self.loss_functions = {}
        for loss_type in loss_types:
            self.loss_functions[loss_type] = self._get_loss_function(loss_type)

    def _get_loss_function(self, loss_type: str):
        """获取损失函数"""
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=self.ignore_index
            )
        elif loss_type == 'focal':
            return FocalLoss(
                alpha=self.class_weights,
                gamma=2.0,
                ignore_index=self.ignore_index
            )
        elif loss_type == 'dice':
            return DiceLoss(ignore_index=self.ignore_index)
        elif loss_type == 'lovasz':
            return LovaszSoftmaxLoss(ignore_index=self.ignore_index)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: 预测结果 [B, C, H, W]
            targets: 真实标签 [B, H, W]

        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0

        for loss_type, weight in zip(self.loss_types, self.loss_weights):
            loss_fn = self.loss_functions[loss_type]
            loss_value = loss_fn(predictions, targets)
            losses[f'{loss_type}_loss'] = loss_value
            total_loss += weight * loss_value

        losses['total_loss'] = total_loss
        return losses


class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets,
                                 weight=self.alpha,
                                 ignore_index=self.ignore_index,
                                 reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 获取有效mask
        valid_mask = (targets != self.ignore_index)

        # 软化预测
        inputs_soft = F.softmax(inputs, dim=1)

        # 转换targets为one-hot
        targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        # 应用valid_mask
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(inputs_soft)
        inputs_soft = inputs_soft * valid_mask_expanded
        targets_onehot = targets_onehot * valid_mask_expanded

        # 计算Dice
        intersection = (inputs_soft * targets_onehot).sum()
        union = inputs_soft.sum() + targets_onehot.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax Loss"""

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 简化版Lovasz损失，实际实现较复杂
        # 这里用Focal Loss近似
        focal = FocalLoss(ignore_index=self.ignore_index)
        return focal(inputs, targets)


class SegmentationTrainer:
    """语义分割训练器"""

    def __init__(
        self,
        model: SegmentationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or setup_logger('trainer')
        self.config = config or {}

        # 将模型移到设备
        self.model = self.model.to(self.device)

        # 初始化指标计算器
        self.metrics = SegmentationMetrics(
            num_classes=self.config.get('num_classes', 21),
            ignore_index=self.config.get('ignore_index', 255)
        )

        # 训练状态
        self.current_epoch = 0
        self.best_miou = 0.0
        self.train_history = []
        self.val_history = []

        # 混合精度训练
        self.use_amp = self.config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        running_loss = 0.0
        running_losses = {}
        num_samples = 0

        # 重置指标
        self.metrics.reset()

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)

            batch_size = images.size(0)
            num_samples += batch_size

            # 前向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    losses = self._compute_loss(outputs, targets)
            else:
                outputs = self.model(images)
                losses = self._compute_loss(outputs, targets)

            total_loss = losses['total_loss']

            # 反向传播
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            # 更新统计
            running_loss += total_loss.item() * batch_size
            for key, value in losses.items():
                if key not in running_losses:
                    running_losses[key] = 0.0
                running_losses[key] += value.item() * batch_size

            # 更新指标
            predictions = outputs['pred']
            pred_labels = torch.argmax(predictions, dim=1)
            self.metrics.update(pred_labels.cpu(), targets.cpu())

            # 更新进度条
            if batch_idx % 10 == 0:
                current_miou = self.metrics.get_miou()
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'mIoU': f'{current_miou:.4f}'
                })

        # 计算epoch平均值
        epoch_loss = running_loss / num_samples
        epoch_losses = {k: v / num_samples for k, v in running_losses.items()}
        epoch_metrics = self.metrics.compute()

        # 学习率调度
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step()

        result = {
            'loss': epoch_loss,
            'miou': epoch_metrics['miou'],
            'accuracy': epoch_metrics['accuracy'],
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        result.update(epoch_losses)

        return result

    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        running_loss = 0.0
        running_losses = {}
        num_samples = 0

        # 重置指标
        self.metrics.reset()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {self.current_epoch}')

            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)

                batch_size = images.size(0)
                num_samples += batch_size

                # 前向传播
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        losses = self._compute_loss(outputs, targets)
                else:
                    outputs = self.model(images)
                    losses = self._compute_loss(outputs, targets)

                total_loss = losses['total_loss']

                # 更新统计
                running_loss += total_loss.item() * batch_size
                for key, value in losses.items():
                    if key not in running_losses:
                        running_losses[key] = 0.0
                    running_losses[key] += value.item() * batch_size

                # 更新指标
                predictions = outputs['pred']
                pred_labels = torch.argmax(predictions, dim=1)
                self.metrics.update(pred_labels.cpu(), targets.cpu())

                # 更新进度条
                current_miou = self.metrics.get_miou()
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'mIoU': f'{current_miou:.4f}'
                })

        # 计算epoch平均值
        epoch_loss = running_loss / num_samples
        epoch_losses = {k: v / num_samples for k, v in running_losses.items()}
        epoch_metrics = self.metrics.compute()

        result = {
            'loss': epoch_loss,
            'miou': epoch_metrics['miou'],
            'accuracy': epoch_metrics['accuracy']
        }
        result.update(epoch_losses)

        return result

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = self.loss_fn(outputs['pred'], targets)

        # 辅助损失
        if 'aux' in outputs:
            aux_losses = self.loss_fn(outputs['aux'], targets)
            aux_weight = self.config.get('aux_weight', 0.4)

            # 合并损失
            combined_losses = {}
            for key in losses.keys():
                if key == 'total_loss':
                    combined_losses[key] = losses[key] + aux_weight * aux_losses[key]
                else:
                    combined_losses[key] = losses[key]
                    combined_losses[f'aux_{key}'] = aux_losses[key]

            return combined_losses

        return losses

    def train(self, num_epochs: int, save_dir: str = None) -> Dict[str, List]:
        """完整训练流程"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")

        # 模型信息
        model_info = get_model_info(self.model)
        self.logger.info(f"Model parameters: {model_info['parameters']['total'] / 1e6:.2f}M")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # 验证
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            # 记录日志
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train mIoU: {train_metrics['miou']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val mIoU: {val_metrics['miou']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # 保存最佳模型
            if val_metrics['miou'] > self.best_miou:
                self.best_miou = val_metrics['miou']
                if save_dir:
                    self.save_checkpoint(
                        os.path.join(save_dir, 'best_model.pth'),
                        epoch, val_metrics['miou'], is_best=True
                    )

            # 定期保存
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'),
                    epoch, val_metrics['miou']
                )

        self.logger.info(f"Training completed. Best mIoU: {self.best_miou:.4f}")

        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }

    def save_checkpoint(self, filepath: str, epoch: int, miou: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'miou': miou,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'model_info': self.model.get_model_info()
        }

        torch.save(checkpoint, filepath)

        if is_best:
            self.logger.info(f"Saved best model with mIoU: {miou:.4f}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_miou = checkpoint.get('miou', 0.0)
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

        return checkpoint


def create_trainer(config: Dict) -> SegmentationTrainer:
    """从配置创建训练器"""
    # 导入必要模块
    from models.unified_model_builder import build_segmentation_model

    # 构建模型
    model = build_segmentation_model(config['model'])

    # 创建数据模块
    data_module = VOCDataModule(
        root_dir=config['data']['root_dir'],
        batch_size=config['training']['batch_size'],
        image_size=tuple(config['data']['image_size'])
    )
    data_module.setup()

    # 创建优化器
    optimizer_config = config['training']['optimizer']
    if optimizer_config['name'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 1e-4)
        )
    elif optimizer_config['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")

    # 创建学习率调度器
    scheduler = None
    if 'scheduler' in config['training']:
        scheduler_config = config['training']['scheduler']
        if scheduler_config['name'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['name'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs']
            )

    # 创建损失函数
    loss_config = config['training']['loss']
    loss_fn = MultiLoss(
        loss_types=loss_config.get('types', ['cross_entropy']),
        loss_weights=loss_config.get('weights', [1.0]),
        ignore_index=config['data'].get('ignore_index', 255)
    )

    # 创建训练器
    trainer = SegmentationTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        config=config
    )

    return trainer


if __name__ == "__main__":
    # 测试配置
    test_config = {
        'model': {
            'backbone': {
                'name': 'mobilenetv3_small',
                'pretrained': False
            },
            'head': {
                'name': 'deeplabv3plus',
                'num_classes': 21
            }
        },
        'data': {
            'root_dir': '/path/to/voc2012',
            'image_size': [512, 512],
            'ignore_index': 255
        },
        'training': {
            'batch_size': 4,
            'epochs': 2,
            'optimizer': {
                'name': 'adam',
                'lr': 1e-3
            },
            'loss': {
                'types': ['cross_entropy'],
                'weights': [1.0]
            }
        },
        'num_classes': 21
    }

    # 创建并测试训练器
    trainer = create_trainer(test_config)

    # 短期训练测试
    history = trainer.train(num_epochs=2)

    print("Training test completed!")
    print(f"Final train mIoU: {history['train_history'][-1]['miou']:.4f}")
    print(f"Final val mIoU: {history['val_history'][-1]['miou']:.4f}")
