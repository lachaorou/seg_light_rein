"""
完整的配置化训练脚本
支持多种主干网络、机制、数据集和训练策略
"""
import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np
from datetime import datetime
import json

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from training.advanced_trainer import create_trainer
from utils.logger import setup_logger
# from utils.config_utils import merge_configs, validate_config


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

    return config


def save_config(config: dict, save_path: str):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif save_path.endswith('.json'):
            json.dump(config, f, indent=2, ensure_ascii=False)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 创建子目录
    subdirs = ['checkpoints', 'logs', 'configs', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    return exp_dir


def create_result_comparison_table(experiments_dir: str, output_file: str = None):
    """创建实验结果对比表"""
    if not os.path.exists(experiments_dir):
        return

    results = []

    # 遍历所有实验目录
    for exp_name in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        # 寻找最佳模型检查点
        checkpoint_dir = os.path.join(exp_path, 'checkpoints')
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location='cpu')

                # 提取实验信息
                config_path = os.path.join(exp_path, 'configs', 'config.yaml')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                else:
                    config = checkpoint.get('config', {})

                # 计算参数量
                model_info = checkpoint.get('model_info', {})
                params = checkpoint.get('train_history', [{}])[-1].get('parameters', {})

                result = {
                    'experiment': exp_name,
                    'backbone': config.get('model', {}).get('backbone', {}).get('name', 'unknown'),
                    'head': config.get('model', {}).get('head', {}).get('name', 'unknown'),
                    'mechanisms': ', '.join(model_info.get('mechanisms', [])) or 'none',
                    'best_miou': checkpoint.get('miou', 0.0),
                    'parameters_m': params.get('total', 0) / 1e6 if params else 0,
                    'final_epoch': checkpoint.get('epoch', 0),
                    'batch_size': config.get('training', {}).get('batch_size', 0),
                    'learning_rate': config.get('training', {}).get('optimizer', {}).get('lr', 0),
                    'image_size': config.get('data', {}).get('image_size', [0, 0]),
                }

                # 获取训练历史
                train_history = checkpoint.get('train_history', [])
                val_history = checkpoint.get('val_history', [])

                if train_history and val_history:
                    result.update({
                        'final_train_loss': train_history[-1].get('loss', 0),
                        'final_train_miou': train_history[-1].get('miou', 0),
                        'final_val_loss': val_history[-1].get('loss', 0),
                        'final_val_miou': val_history[-1].get('miou', 0),
                    })

                results.append(result)

            except Exception as e:
                print(f"Error loading checkpoint {best_model_path}: {e}")

    if not results:
        print("No valid experiment results found.")
        return

    # 创建对比表
    comparison_table = create_markdown_table(results)

    # 保存或打印结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(comparison_table)
        print(f"Comparison table saved to: {output_file}")
    else:
        print(comparison_table)

    return results


def create_markdown_table(results: list) -> str:
    """创建Markdown格式的对比表"""
    if not results:
        return "No results to display."

    # 表头
    headers = [
        'Experiment', 'Backbone', 'Head', 'Mechanisms', 'Best mIoU',
        'Parameters (M)', 'Final Epoch', 'Batch Size', 'Learning Rate',
        'Image Size', 'Final Train Loss', 'Final Val Loss'
    ]

    # 创建表格
    table_lines = []

    # 表头
    table_lines.append('| ' + ' | '.join(headers) + ' |')
    table_lines.append('|' + '---|' * len(headers))

    # 数据行
    for result in sorted(results, key=lambda x: x['best_miou'], reverse=True):
        row = [
            result['experiment'][:20] + '...' if len(result['experiment']) > 20 else result['experiment'],
            result['backbone'],
            result['head'],
            result['mechanisms'][:15] + '...' if len(result['mechanisms']) > 15 else result['mechanisms'],
            f"{result['best_miou']:.4f}",
            f"{result['parameters_m']:.2f}",
            str(result['final_epoch']),
            str(result['batch_size']),
            f"{result['learning_rate']:.6f}",
            f"{result['image_size'][0]}x{result['image_size'][1]}",
            f"{result.get('final_train_loss', 0):.4f}",
            f"{result.get('final_val_loss', 0):.4f}",
        ]
        table_lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(table_lines)


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--exp-name', type=str, help='Experiment name')
    parser.add_argument('--exp-dir', type=str, default='./experiments', help='Experiments directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, help='Device to use (cpu/cuda)')
    parser.add_argument('--compare', action='store_true', help='Generate comparison table')
    parser.add_argument('--compare-output', type=str, help='Output file for comparison table')

    args = parser.parse_args()

    # 生成对比表
    if args.compare:
        create_result_comparison_table(args.exp_dir, args.compare_output)
        return

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    config = load_config(args.config)

    # 设置实验名称
    if args.exp_name:
        experiment_name = args.exp_name
    else:
        experiment_name = config.get('experiment_name', 'default_exp')

    # 创建实验目录
    exp_dir = create_experiment_dir(args.exp_dir, experiment_name)

    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      # 设置日志
    logger = setup_logger('train', os.path.join(exp_dir, 'logs', 'train.log'))
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Device: {device}")

    # 保存配置
    config_save_path = os.path.join(exp_dir, 'configs', 'config.yaml')
    save_config(config, config_save_path)
    logger.info(f"Config saved to: {config_save_path}")

    try:
        # 创建训练器
        logger.info("Creating trainer...")
        trainer = create_trainer(config)
        trainer.device = device
        trainer.model = trainer.model.to(device)
        trainer.logger = logger

        # 恢复训练
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = trainer.load_checkpoint(args.resume)
            start_epoch = checkpoint['epoch'] + 1

        # 开始训练
        num_epochs = config['training']['epochs']
        remaining_epochs = num_epochs - start_epoch

        if remaining_epochs > 0:
            logger.info(f"Training for {remaining_epochs} epochs (starting from epoch {start_epoch})")

            # 训练
            history = trainer.train(
                num_epochs=remaining_epochs,
                save_dir=os.path.join(exp_dir, 'checkpoints')
            )

            # 保存训练历史
            history_path = os.path.join(exp_dir, 'results', 'training_history.yaml')
            save_config(history, history_path)
            logger.info(f"Training history saved to: {history_path}")

            # 保存最终结果摘要
            final_results = {
                'experiment_name': experiment_name,
                'config': config,
                'best_miou': trainer.best_miou,
                'final_epoch': trainer.current_epoch,
                'total_epochs': num_epochs,
                'device': str(device),
                'model_info': trainer.model.get_model_info(),
                'parameters': trainer.model.count_parameters()
            }

            summary_path = os.path.join(exp_dir, 'results', 'experiment_summary.yaml')
            save_config(final_results, summary_path)
            logger.info(f"Experiment summary saved to: {summary_path}")

            logger.info(f"Experiment completed successfully!")
            logger.info(f"Best mIoU: {trainer.best_miou:.4f}")

        else:
            logger.info("Training already completed.")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    finally:
        # 生成当前实验的对比表
        try:
            comparison_path = os.path.join(exp_dir, 'results', 'comparison_table.md')
            create_result_comparison_table(args.exp_dir, comparison_path)
            logger.info(f"Comparison table updated: {comparison_path}")
        except Exception as e:            logger.warning(f"Failed to generate comparison table: {e}")


if __name__ == '__main__':
    main()
