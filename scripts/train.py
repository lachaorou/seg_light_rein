"""
seg_light_rein 主训练脚本
支持模块化组合和配置文件驱动
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_manager import ConfigManager, ExperimentConfig
from models.model_builder import ModelBuilder
from utils.dataset import SegmentationDataset
from utils.trainer import Trainer
from utils.logger import setup_logger
from utils.metrics import SegmentationMetrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Seg Light Rein Training Script')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (如: configs/mobilenetv3_baseline.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='实验结果输出目录')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式，使用少量数据')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto, cpu, cuda:0, cuda:1)')
    return parser.parse_args()


def setup_device(device_arg):
    """设置训练设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU信息: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    return device


def create_output_dir(config: ExperimentConfig, output_dir: str):
    """创建实验输出目录"""
    exp_dir = Path(output_dir) / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'visualizations').mkdir(exist_ok=True)

    return str(exp_dir)


def print_config_summary(config: ExperimentConfig):
    """打印配置摘要"""
    print("\n" + "="*50)
    print(f"实验名称: {config.experiment_name}")
    print("="*50)
    print(f"主干网络: {config.backbone.name}")
    print(f"预训练: {config.backbone.pretrained}")
    print(f"R机制: {'启用' if config.rein.enabled else '禁用'}")
    if config.rein.enabled:
        print(f"  - 机制类型: {config.rein.mechanism_type}")
        print(f"  - 插入策略: {config.rein.insertion_strategy}")
    print(f"分割头: {config.head.name}")
    print(f"特征增强: {config.head.feature_enhancement}")
    print(f"类别数: {config.head.num_classes}")
    print(f"批量大小: {config.training.batch_size}")
    print(f"学习率: {config.training.learning_rate}")
    print(f"训练轮数: {config.training.epochs}")
    print(f"损失函数: {config.training.loss_type}")
    print(f"优化器: {config.training.optimizer}")
    print("="*50)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 加载配置
    print(f"加载配置文件: {args.config}")
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)

    # 创建输出目录
    exp_dir = create_output_dir(config, args.output_dir)

    # 设置日志
    logger = setup_logger(exp_dir)
    logger.info(f"实验开始: {config.experiment_name}")

    # 打印配置
    print_config_summary(config)

    # 设置设备
    device = setup_device(args.device)

    # 构建模型
    print("\n构建模型...")
    model_builder = ModelBuilder()
    model = model_builder.build_model(config)
    model = model.to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # 构建数据集
    print("\n构建数据集...")
    train_dataset = SegmentationDataset(
        split='train',
        num_classes=config.head.num_classes,
        debug=args.debug
    )
    val_dataset = SegmentationDataset(
        split='val',
        num_classes=config.head.num_classes,
        debug=args.debug
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    # 构建训练器
    print("\n初始化训练器...")
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        exp_dir=exp_dir,
        logger=logger
    )

    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"恢复训练从: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)

    # 开始训练
    print(f"\n开始训练 (从第{start_epoch}轮开始)...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
        total_epochs=config.training.epochs
    )

    print(f"\n训练完成！结果保存在: {exp_dir}")

    # 最终评估
    print("\n进行最终评估...")
    metrics = trainer.evaluate(val_loader)

    # 保存最终结果
    result_summary = {
        'experiment_name': config.experiment_name,
        'final_metrics': metrics,
        'model_params': total_params,
        'config': config.__dict__
    }

    import json
    with open(os.path.join(exp_dir, 'final_results.json'), 'w') as f:
        json.dump(result_summary, f, indent=2, default=str)

    logger.info(f"实验完成: {config.experiment_name}")
    print(f"\n✅ 实验完成！最终mIoU: {metrics.get('mIoU', 0):.4f}")


if __name__ == '__main__':
    main()
