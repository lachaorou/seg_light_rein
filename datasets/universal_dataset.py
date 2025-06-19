"""
通用数据集框架 - 支持多种语义分割数据集格式
包括 Cityscapes, VOC, ADE20K, COCO-Stuff 等
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2  # 添加cv2导入
from typing import Tuple, List, Optional, Dict, Any, Union
import json
from abc import ABC, abstractmethod


class BaseSegmentationDataset(Dataset, ABC):
    """语义分割数据集基类"""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        transforms: Optional[Any] = None,
        ignore_index: int = 255
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.ignore_index = ignore_index

        # 子类需要设置这些属性
        self.classes = []
        self.colors = []
        self.num_classes = 0

        # 加载数据列表
        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self) -> List[Dict[str, str]]:
        """加载数据样本列表，返回包含image_path和mask_path的字典列表"""
        pass

    @abstractmethod
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """加载并预处理mask"""
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 处理dummy数据
        if sample['image_path'] == 'dummy':
            # 创建dummy图像和mask
            image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            mask = np.random.randint(0, self.num_classes, (256, 256), dtype=np.uint8)
        else:
            # 加载真实图像和mask
            image = Image.open(sample['image_path']).convert('RGB')
            mask = self._load_mask(sample['mask_path'])

        # 应用变换
        if self.transforms:
            image, mask = self.transforms(image, mask)

        # 转换为tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask


class CityscapesDataset(BaseSegmentationDataset):
    """Cityscapes数据集"""

    # Cityscapes 19类（训练用）
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    # 类别ID映射（从trainId到标准ID）
    TRAIN_ID_TO_ID = {
        0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22,
        10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 19

    def _load_samples(self) -> List[Dict[str, str]]:
        """加载Cityscapes样本"""
        samples = []

        # Cityscapes目录结构: leftImg8bit/{split}/* 和 gtFine/{split}/*
        img_dir = os.path.join(self.root_dir, 'leftImg8bit', self.split)
        mask_dir = os.path.join(self.root_dir, 'gtFine', self.split)

        if not os.path.exists(img_dir):
            print(f"Warning: Cityscapes dataset not found at {img_dir}")
            return self._create_dummy_samples()

        for city in os.listdir(img_dir):
            city_img_dir = os.path.join(img_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)

            if not os.path.isdir(city_img_dir):
                continue

            for img_file in os.listdir(city_img_dir):
                if img_file.endswith('_leftImg8bit.png'):
                    # 构建对应的mask文件名
                    base_name = img_file.replace('_leftImg8bit.png', '')
                    mask_file = f"{base_name}_gtFine_labelTrainIds.png"

                    img_path = os.path.join(city_img_dir, img_file)
                    mask_path = os.path.join(city_mask_dir, mask_file)

                    if os.path.exists(mask_path):
                        samples.append({
                            'image_path': img_path,
                            'mask_path': mask_path
                        })

        return samples

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """加载Cityscapes mask"""
        mask = np.array(Image.open(mask_path))

        # 调整大小
        if mask.shape[:2] != self.image_size:
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        return mask

    def _create_dummy_samples(self) -> List[Dict[str, str]]:
        """创建dummy数据用于测试"""
        print("Creating dummy Cityscapes dataset for testing...")
        return [{'image_path': 'dummy', 'mask_path': 'dummy'} for _ in range(100)]


class VOC2012Dataset(BaseSegmentationDataset):
    """VOC2012数据集"""

    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 21

    def _load_samples(self) -> List[Dict[str, str]]:
        """加载VOC2012样本"""
        samples = []

        # VOC目录结构: JPEGImages/*.jpg 和 SegmentationClass/*.png
        split_file = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', f'{self.split}.txt')
        img_dir = os.path.join(self.root_dir, 'JPEGImages')
        mask_dir = os.path.join(self.root_dir, 'SegmentationClass')

        if not os.path.exists(split_file):
            print(f"Warning: VOC2012 dataset not found at {self.root_dir}")
            return self._create_dummy_samples()

        with open(split_file, 'r') as f:
            for line in f:
                img_name = line.strip()
                img_path = os.path.join(img_dir, f'{img_name}.jpg')
                mask_path = os.path.join(mask_dir, f'{img_name}.png')

                if os.path.exists(img_path) and os.path.exists(mask_path):
                    samples.append({
                        'image_path': img_path,
                        'mask_path': mask_path
                    })

        return samples

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """加载VOC2012 mask"""
        mask = np.array(Image.open(mask_path))

        # 调整大小
        if mask.shape[:2] != self.image_size:
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        return mask

    def _create_dummy_samples(self) -> List[Dict[str, str]]:
        """创建dummy数据用于测试"""
        print("Creating dummy VOC2012 dataset for testing...")
        return [{'image_path': 'dummy', 'mask_path': 'dummy'} for _ in range(100)]


class ADE20KDataset(BaseSegmentationDataset):
    """ADE20K数据集"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 150  # ADE20K有150个类别

    def _load_samples(self) -> List[Dict[str, str]]:
        """加载ADE20K样本"""
        samples = []

        # ADE20K目录结构: images/{split}/* 和 annotations/{split}/*
        img_dir = os.path.join(self.root_dir, 'images', self.split)
        mask_dir = os.path.join(self.root_dir, 'annotations', self.split)

        if not os.path.exists(img_dir):
            print(f"Warning: ADE20K dataset not found at {img_dir}")
            return self._create_dummy_samples()

        for img_file in os.listdir(img_dir):
            if img_file.endswith('.jpg'):
                mask_file = img_file.replace('.jpg', '.png')

                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)

                if os.path.exists(mask_path):
                    samples.append({
                        'image_path': img_path,
                        'mask_path': mask_path
                    })

        return samples

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """加载ADE20K mask"""
        mask = np.array(Image.open(mask_path))

        # ADE20K的mask从1开始编号，需要减1
        mask = mask - 1
        mask[mask < 0] = self.ignore_index

        # 调整大小
        if mask.shape[:2] != self.image_size:
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        return mask

    def _create_dummy_samples(self) -> List[Dict[str, str]]:
        """创建dummy数据用于测试"""
        print("Creating dummy ADE20K dataset for testing...")
        return [{'image_path': 'dummy', 'mask_path': 'dummy'} for _ in range(100)]


def create_dataset(dataset_type: str, **kwargs) -> BaseSegmentationDataset:
    """数据集工厂函数"""
    dataset_map = {
        'cityscapes': CityscapesDataset,
        'voc2012': VOC2012Dataset,
        'ade20k': ADE20KDataset,
    }

    if dataset_type not in dataset_map:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return dataset_map[dataset_type](**kwargs)


# 为了保持向后兼容，保留原有的VOC2012Dataset导入
__all__ = ['CityscapesDataset', 'VOC2012Dataset', 'ADE20KDataset', 'create_dataset']
