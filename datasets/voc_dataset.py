"""
VOC2012数据集处理模块
支持语义分割任务的数据加载、预处理和增强
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import cv2
import random


class VOC2012Dataset(Dataset):
    """VOC2012语义分割数据集"""

    # VOC2012类别（包括背景）
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]

    # 类别颜色映射（用于可视化）
    COLORS = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        transforms: Optional[Any] = None,
        augment: bool = True,
        ignore_index: int = 255
    ):
        """
        Args:
            root_dir: VOC2012数据集根目录
            split: 数据集分割 ('train', 'val', 'trainval')
            image_size: 输出图像尺寸
            transforms: 自定义变换
            augment: 是否使用数据增强
            ignore_index: 忽略的类别索引
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == 'train'
        self.ignore_index = ignore_index

        # 检查数据集路径
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.masks_dir = os.path.join(root_dir, 'SegmentationClass')
        self.splits_dir = os.path.join(root_dir, 'ImageSets', 'Segmentation')

        if not all(os.path.exists(p) for p in [self.images_dir, self.masks_dir, self.splits_dir]):
            # 如果标准路径不存在，创建一个dummy数据集用于测试
            print(f"Warning: VOC2012 dataset not found at {root_dir}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_dataset()
            return

        # 加载文件列表
        split_file = os.path.join(self.splits_dir, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_names = [line.strip() for line in f.readlines()]
        else:
            # 如果split文件不存在，使用所有可用文件
            self.file_names = [f.split('.')[0] for f in os.listdir(self.images_dir) if f.endswith('.jpg')]

        # 设置变换
        if transforms is None:
            self.transforms = self._get_default_transforms()
        else:
            self.transforms = transforms

        print(f"Loaded {len(self.file_names)} {split} samples from VOC2012 dataset")

    def _create_dummy_dataset(self):
        """创建用于测试的虚拟数据集"""
        self.file_names = [f'dummy_{i:04d}' for i in range(100)]
        self.is_dummy = True
        self.transforms = self._get_default_transforms()

    def _get_default_transforms(self):
        """获取默认的数据变换"""
        if self.augment:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if hasattr(self, 'is_dummy') and self.is_dummy:
            return self._get_dummy_item(idx)

        file_name = self.file_names[idx]

        # 加载图像
        image_path = os.path.join(self.images_dir, f'{file_name}.jpg')
        image = Image.open(image_path).convert('RGB')

        # 加载标签
        mask_path = os.path.join(self.masks_dir, f'{file_name}.png')
        mask = Image.open(mask_path)

        # 数据增强
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # 调整尺寸
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        # 转换为tensor
        if self.transforms:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)

        # 处理mask
        mask = np.array(mask, dtype=np.int64)
        # 将255（边界）设为ignore_index
        mask[mask == 255] = self.ignore_index
        mask = torch.from_numpy(mask)

        return {
            'image': image,
            'mask': mask,
            'file_name': file_name
        }

    def _get_dummy_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取虚拟数据项"""
        # 生成随机图像
        image = torch.randn(3, *self.image_size)
        # 生成随机分割标签
        mask = torch.randint(0, len(self.CLASSES), self.image_size, dtype=torch.long)

        return {
            'image': image,
            'mask': mask,
            'file_name': self.file_names[idx]
        }

    def _apply_augmentation(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """应用数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机缩放和裁剪
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.BILINEAR)
            mask = mask.resize(new_size, Image.NEAREST)

        # 随机旋转
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, Image.BILINEAR)
            mask = mask.rotate(angle, Image.NEAREST)

        # 颜色抖动（仅图像）
        if random.random() > 0.5:
            # 使用PIL的ColorJitter
            color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            image = color_jitter(image)

        return image, mask

    @staticmethod
    def decode_segmap(label_mask: np.ndarray, num_classes: int = 21) -> np.ndarray:
        """将分割标签解码为RGB图像"""
        label_colors = np.array(VOC2012Dataset.COLORS[:num_classes])
        r = np.zeros_like(label_mask).astype(np.uint8)
        g = np.zeros_like(label_mask).astype(np.uint8)
        b = np.zeros_like(label_mask).astype(np.uint8)

        for class_idx in range(num_classes):
            idx = label_mask == class_idx
            r[idx] = label_colors[class_idx, 0]
            g[idx] = label_colors[class_idx, 1]
            b[idx] = label_colors[class_idx, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb


class VOCDataModule:
    """VOC数据模块，管理训练和验证数据加载器"""

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (512, 512),
        pin_memory: bool = True
    ):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory

        # 创建数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """设置数据集"""
        self.train_dataset = VOC2012Dataset(
            root_dir=self.root_dir,
            split='train',
            image_size=self.image_size,
            augment=True
        )

        self.val_dataset = VOC2012Dataset(
            root_dir=self.root_dir,
            split='val',
            image_size=self.image_size,
            augment=False
        )

    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        if self.train_dataset is None:
            self.setup()

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        if self.val_dataset is None:
            self.setup()

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        # 这里可以根据数据集统计计算真实的类别权重
        # 暂时返回均匀权重
        return torch.ones(len(VOC2012Dataset.CLASSES))


def create_voc_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 8,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """创建VOC数据加载器的便捷函数"""
    dataset = VOC2012Dataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        augment=(split == 'train'),
        **kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


if __name__ == "__main__":
    # 测试代码
    print("Testing VOC2012 Dataset...")

    # 测试数据集（使用dummy数据）
    dataset = VOC2012Dataset(
        root_dir='/path/to/nonexistent/voc2012',  # 不存在的路径，将创建dummy数据
        split='train',
        image_size=(512, 512)
    )

    print(f"Dataset size: {len(dataset)}")

    # 测试数据加载
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"File name: {sample['file_name']}")

    # 测试数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Masks shape: {batch['mask'].shape}")
        print(f"  Unique mask values: {torch.unique(batch['mask'])}")

        if batch_idx >= 2:  # 只测试几个批次
            break

    print("VOC2012 Dataset test completed!")
