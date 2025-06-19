"""
分割评估指标
"""
import numpy as np
import torch
from typing import Dict, List


class SegmentationMetrics:
    """语义分割评估指标"""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """重置指标"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """更新混淆矩阵"""
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        # 去除ignore_index
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]

        # 计算混淆矩阵
        for p, t in zip(pred.flatten(), target.flatten()):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def get_metrics(self) -> Dict[str, float]:
        """计算所有指标"""
        # IoU for each class
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                self.confusion_matrix.sum(axis=0) -
                intersection)

        # 避免除零错误
        iou = intersection / np.maximum(union, 1e-8)

        # mIoU
        valid_classes = union > 0
        miou = iou[valid_classes].mean()

        # Pixel Accuracy
        pixel_acc = intersection.sum() / np.maximum(self.confusion_matrix.sum(), 1e-8)

        # Mean Pixel Accuracy
        class_acc = intersection / np.maximum(self.confusion_matrix.sum(axis=1), 1e-8)
        mean_acc = class_acc[valid_classes].mean()

        # Frequency Weighted IoU
        freq = self.confusion_matrix.sum(axis=1) / np.maximum(self.confusion_matrix.sum(), 1e-8)
        fwiou = (freq[valid_classes] * iou[valid_classes]).sum()

        return {
            'miou': miou,  # 使用小写miou以匹配trainer期望
            'mIoU': miou,  # 保持兼容性
            'accuracy': pixel_acc,  # 添加accuracy键以匹配trainer期望
            'pixel_accuracy': pixel_acc,
            'mean_accuracy': mean_acc,
            'fwIoU': fwiou,
            'per_class_iou': iou.tolist()
        }

    def compute(self) -> Dict[str, float]:
        """计算并返回所有指标（与get_metrics相同，提供兼容性）"""
        return self.get_metrics()

    def get_miou(self) -> float:
        """获取mIoU指标"""
        metrics = self.get_metrics()
        return metrics['mIoU']

    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return self.confusion_matrix


def compute_metrics(predictions: List[torch.Tensor],
                   targets: List[torch.Tensor],
                   num_classes: int) -> Dict[str, float]:
    """计算批量预测的指标"""
    metrics = SegmentationMetrics(num_classes)

    for pred, target in zip(predictions, targets):
        # 如果是logits，转换为预测类别
        if pred.dim() > target.dim():
            pred = pred.argmax(dim=1)

        metrics.update(pred, target)

    return metrics.get_metrics()
