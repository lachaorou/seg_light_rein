"""
简化的数据集加载器 - 用于训练流程测试
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List


class SegmentationDataset(Dataset):
    """语义分割数据集"""

    def __init__(self, split: str = 'train', num_classes: int = 21,
                 input_size: Tuple[int, int] = (512, 512), debug: bool = False):
        self.split = split
        self.num_classes = num_classes
        self.input_size = input_size
        self.debug = debug

        # 临时生成假数据用于测试训练流程
        if debug:
            self.length = 100  # 调试模式少量数据
        else:
            self.length = 1000 if split == 'train' else 200

        print(f"[临时] 使用模拟数据集 - {split}: {self.length} 样本")
        print(f"[提示] 后续需要替换为真实数据集加载器")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 生成随机图像和标签用于测试
        H, W = self.input_size

        # 模拟图像 (3, H, W)
        image = torch.randn(3, H, W)

        # 模拟分割标签 (H, W)
        label = torch.randint(0, self.num_classes, (H, W), dtype=torch.long)

        # 添加一些ignore区域
        ignore_mask = torch.rand(H, W) < 0.1  # 10%的像素设为ignore
        label[ignore_mask] = 255  # ignore_index

        return image, label


class VOCSegmentationDataset(Dataset):
    """VOC分割数据集 - 真实数据集实现模板"""

    def __init__(self, split: str = 'train', num_classes: int = 21,
                 data_root: str = 'Dataset/VOC', transforms=None):
        # TODO: 实现真实的VOC数据集加载
        pass

    def __len__(self):
        # TODO: 返回实际数据量
        pass

    def __getitem__(self, idx):
        # TODO: 加载真实图像和标签
        pass


class CityscapesDataset(Dataset):
    """Cityscapes数据集 - 真实数据集实现模板"""

    def __init__(self, split: str = 'train', num_classes: int = 19,
                 data_root: str = 'Dataset/Cityscapes', transforms=None):
        # TODO: 实现Cityscapes数据集加载
        pass

    def __len__(self):
        # TODO: 返回实际数据量
        pass

    def __getitem__(self, idx):
        # TODO: 加载真实图像和标签
        pass


def get_dataset(dataset_name: str, **kwargs) -> Dataset:
    """数据集工厂函数"""
    if dataset_name.lower() == 'voc':
        return VOCSegmentationDataset(**kwargs)
    elif dataset_name.lower() == 'cityscapes':
        return CityscapesDataset(**kwargs)
    elif dataset_name.lower() == 'mock':
        return SegmentationDataset(**kwargs)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


# 使用示例
if __name__ == "__main__":
    # 测试模拟数据集
    dataset = SegmentationDataset(split='train', debug=True)
    image, label = dataset[0]
    print(f"图像形状: {image.shape}")
    print(f"标签形状: {label.shape}")
    print(f"标签范围: {label.min()} - {label.max()}")
