#!/usr/bin/env python3
"""
多数据集批量实验脚本
按阶段运行不同数据集的实验，实现渐进式验证
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from configs.config_manager import ConfigManager
from utils.logger import setup_logger

def setup_experiment_logger(experiment_name: str) -> logging.Logger:
    """设置实验专用日志"""
    log_dir = project_root / 'experiments' / experiment_name / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=f'experiment_{experiment_name}',
        log_file=str(log_dir / 'experiment.log'),
        level=logging.INFO
    )
    return logger

def run_single_experiment(config_file: str, stage_name: str) -> Dict[str, Any]:
    """运行单个实验"""
    logger = logging.getLogger('batch_experiment')

    try:
        # 加载配置
        config = ConfigManager(config_file)
        experiment_name = config.experiment.name

        logger.info(f"开始{stage_name}: {experiment_name}")
        logger.info(f"配置文件: {config_file}")

        # 设置实验专用日志
        exp_logger = setup_experiment_logger(experiment_name)

        # 记录实验开始时间
        start_time = time.time()

        # 运行训练脚本
        cmd = [
            sys.executable, 'train_complete.py',
            '--config', config_file,
            '--experiment_name', experiment_name
        ]

        logger.info(f"执行命令: {' '.join(cmd)}")

        # 执行训练
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )

        # 记录实验结束时间
        end_time = time.time()
        duration = end_time - start_time

        # 处理结果
        if result.returncode == 0:
            logger.info(f"{stage_name}成功完成，耗时: {duration:.2f}秒")
            exp_logger.info(f"实验成功完成，耗时: {duration:.2f}秒")

            # 解析输出中的指标
            metrics = parse_metrics_from_output(result.stdout)

            return {
                'success': True,
                'duration': duration,
                'metrics': metrics,
                'config_file': config_file,
                'experiment_name': experiment_name
            }
        else:
            error_msg = f"{stage_name}失败，返回码: {result.returncode}"
            logger.error(error_msg)
            logger.error(f"标准输出: {result.stdout}")
            logger.error(f"错误输出: {result.stderr}")

            exp_logger.error(error_msg)
            exp_logger.error(f"错误输出: {result.stderr}")

            return {
                'success': False,
                'duration': duration,
                'error': result.stderr,
                'config_file': config_file,
                'experiment_name': experiment_name
            }

    except subprocess.TimeoutExpired:
        error_msg = f"{stage_name}超时（2小时）"
        logger.error(error_msg)
        return {
            'success': False,
            'duration': 7200,
            'error': '实验超时',
            'config_file': config_file
        }
    except Exception as e:
        error_msg = f"{stage_name}异常: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'duration': 0,
            'error': str(e),
            'config_file': config_file
        }

def parse_metrics_from_output(output: str) -> Dict[str, float]:
    """从输出中解析指标"""
    metrics = {}

    lines = output.split('\n')
    for line in lines:
        if 'Final mIoU:' in line:
            try:
                miou = float(line.split('Final mIoU:')[1].strip().replace('%', ''))
                metrics['miou'] = miou
            except:
                pass
        elif 'Pixel Accuracy:' in line:
            try:
                acc = float(line.split('Pixel Accuracy:')[1].strip().replace('%', ''))
                metrics['pixel_accuracy'] = acc
            except:
                pass

    return metrics

def generate_summary_report(results: List[Dict[str, Any]]) -> None:
    """生成总结报告"""
    logger = logging.getLogger('batch_experiment')

    # 创建报告目录
    report_dir = project_root / 'experiments' / 'batch_experiment_report'
    report_dir.mkdir(parents=True, exist_ok=True)

    # 生成报告
    report_file = report_dir / f'summary_{int(time.time())}.md'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 多数据集批量实验报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 统计信息
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results if r['success'])
        failed_experiments = total_experiments - successful_experiments
        total_duration = sum(r['duration'] for r in results)

        f.write("## 实验统计\n\n")
        f.write(f"- 总实验数: {total_experiments}\n")
        f.write(f"- 成功实验: {successful_experiments}\n")
        f.write(f"- 失败实验: {failed_experiments}\n")
        f.write(f"- 总耗时: {total_duration:.2f}秒 ({total_duration/3600:.2f}小时)\n\n")

        # 成功实验详情
        f.write("## 成功实验结果\n\n")
        f.write("| 实验名称 | 数据集 | 配置文件 | 耗时(分钟) | mIoU(%) | Pixel Acc(%) |\n")
        f.write("|----------|--------|----------|------------|---------|-------------|\n")

        for result in results:
            if result['success']:
                duration_min = result['duration'] / 60
                miou = result['metrics'].get('miou', 'N/A')
                pixel_acc = result['metrics'].get('pixel_accuracy', 'N/A')
                dataset_type = extract_dataset_type(result['config_file'])

                f.write(f"| {result['experiment_name']} | {dataset_type} | {result['config_file']} | {duration_min:.1f} | {miou} | {pixel_acc} |\n")

        # 失败实验详情
        if failed_experiments > 0:
            f.write("\n## 失败实验详情\n\n")
            for result in results:
                if not result['success']:
                    f.write(f"### {result.get('experiment_name', 'Unknown')}\n")
                    f.write(f"- 配置文件: {result['config_file']}\n")
                    f.write(f"- 错误信息: {result['error']}\n\n")

        # 性能对比
        f.write("\n## 性能对比分析\n\n")
        f.write("### 不同数据集性能对比\n\n")

        dataset_performance = {}
        for result in results:
            if result['success'] and 'miou' in result['metrics']:
                dataset = extract_dataset_type(result['config_file'])
                if dataset not in dataset_performance:
                    dataset_performance[dataset] = []
                dataset_performance[dataset].append(result['metrics']['miou'])

        for dataset, mious in dataset_performance.items():
            avg_miou = sum(mious) / len(mious)
            f.write(f"- **{dataset}**: 平均mIoU = {avg_miou:.2f}%\n")

        # 建议和总结
        f.write("\n## 建议和总结\n\n")

        if successful_experiments == total_experiments:
            f.write("✅ 所有实验成功完成！\n\n")
            f.write("**建议下一步:**\n")
            f.write("1. 分析各数据集的性能差异\n")
            f.write("2. 尝试更复杂的模型架构\n")
            f.write("3. 进行超参数优化\n")
            f.write("4. 实施模型集成策略\n")
        elif successful_experiments > 0:
            f.write(f"⚠️ 部分实验失败 ({failed_experiments}/{total_experiments})\n\n")
            f.write("**建议:**\n")
            f.write("1. 检查失败实验的错误信息\n")
            f.write("2. 调整配置参数\n")
            f.write("3. 检查数据集路径和格式\n")
            f.write("4. 确保计算资源充足\n")
        else:
            f.write("❌ 所有实验失败！\n\n")
            f.write("**紧急建议:**\n")
            f.write("1. 检查环境配置\n")
            f.write("2. 验证数据集完整性\n")
            f.write("3. 简化配置重新测试\n")
            f.write("4. 查看详细错误日志\n")

    logger.info(f"总结报告已生成: {report_file}")

def extract_dataset_type(config_file: str) -> str:
    """从配置文件名中提取数据集类型"""
    if 'voc2012' in config_file.lower():
        return 'VOC2012'
    elif 'ade20k' in config_file.lower():
        return 'ADE20K'
    elif 'cityscapes' in config_file.lower():
        return 'Cityscapes'
    else:
        return 'Unknown'

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        name='batch_experiment',
        log_file=str(project_root / 'experiments' / 'batch_experiment.log'),
        level=logging.INFO
    )

    logger.info("开始多数据集批量实验")

    # 定义实验阶段
    experiment_stages = [
        {
            'name': '阶段1: VOC2012快速验证',
            'config': 'configs/voc2012_quick_test.yaml',
            'description': '验证模型架构和训练流程'
        },
        {
            'name': '阶段2: ADE20K全面评估',
            'config': 'configs/ade20k_full_eval.yaml',
            'description': '验证模型泛化能力和多类别性能'
        },
        {
            'name': '阶段3: Cityscapes高精度测试',
            'config': 'configs/cityscapes_precision_test.yaml',
            'description': '追求最高精度和细节优化'
        }
    ]

    results = []
    total_start_time = time.time()

    for i, stage in enumerate(experiment_stages, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"开始 {stage['name']} ({i}/{len(experiment_stages)})")
        logger.info(f"描述: {stage['description']}")
        logger.info(f"配置: {stage['config']}")
        logger.info(f"{'='*60}")

        # 检查配置文件是否存在
        config_path = project_root / stage['config']
        if not config_path.exists():
            error_msg = f"配置文件不存在: {config_path}"
            logger.error(error_msg)
            results.append({
                'success': False,
                'duration': 0,
                'error': error_msg,
                'config_file': stage['config']
            })
            continue

        # 运行实验
        result = run_single_experiment(stage['config'], stage['name'])
        results.append(result)

        # 如果实验失败且是关键阶段，询问是否继续
        if not result['success'] and i == 1:  # VOC2012是基础验证
            logger.warning("基础验证实验失败，建议检查配置后重试")
            logger.warning("继续后续实验可能会失败...")

    # 记录总体完成时间
    total_duration = time.time() - total_start_time
    logger.info(f"\n所有实验完成，总耗时: {total_duration:.2f}秒 ({total_duration/3600:.2f}小时)")

    # 生成总结报告
    generate_summary_report(results)

    # 输出简要结果
    successful_count = sum(1 for r in results if r['success'])
    logger.info(f"实验结果: {successful_count}/{len(results)} 成功")

    if successful_count == len(results):
        logger.info("🎉 所有实验成功完成！")
        return 0
    elif successful_count > 0:
        logger.warning(f"⚠️ 部分实验失败 ({len(results) - successful_count} 个)")
        return 1
    else:
        logger.error("❌ 所有实验失败！")
        return 2

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
