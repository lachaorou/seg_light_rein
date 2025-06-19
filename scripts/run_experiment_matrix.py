"""
实验矩阵自动化运行脚本
支持批量运行多个实验配置，自动记录结果和生成对比报告
"""
import os
import sys
import yaml
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentRunner:
    """实验自动化运行器"""

    def __init__(self,
                 base_dir: str = ".",
                 results_dir: str = "results/experiment_matrix",
                 gpu_ids: List[int] = [0]):
        self.base_dir = Path(base_dir)
        self.results_dir = Path(results_dir)
        self.gpu_ids = gpu_ids
        self.current_gpu = 0

        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 实验记录
        self.experiment_log = []

    def get_next_gpu(self) -> int:
        """获取下一个可用GPU"""
        gpu_id = self.gpu_ids[self.current_gpu % len(self.gpu_ids)]
        self.current_gpu += 1
        return gpu_id

    def load_experiment_configs(self, config_pattern: str) -> List[Dict]:
        """加载实验配置文件"""
        configs = []
        config_files = list(self.base_dir.glob(config_pattern))

        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    config['config_file'] = str(config_file)
                    config['experiment_id'] = config_file.stem
                    configs.append(config)
            except Exception as e:
                print(f"Warning: Failed to load {config_file}: {e}")

        return configs

    def run_single_experiment(self, config: Dict, timeout: int = 7200) -> Dict:
        """运行单个实验"""
        experiment_id = config['experiment_id']
        config_file = config['config_file']
        gpu_id = self.get_next_gpu()

        print(f"\\n{'='*50}")
        print(f"Running Experiment: {experiment_id}")
        print(f"Config: {config_file}")
        print(f"GPU: {gpu_id}")
        print(f"{'='*50}")

        # 准备实验目录
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)

        # 记录开始时间
        start_time = time.time()

        # 构建运行命令
        cmd = [
            "python", "train_complete.py",
            "--config", config_file,
            "--gpu_id", str(gpu_id),
            "--output_dir", str(exp_dir)
        ]

        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            # 运行实验
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # 记录结束时间
            end_time = time.time()
            duration = end_time - start_time

            # 解析结果
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr

            # 尝试从输出中提取性能指标
            metrics = self.extract_metrics_from_output(stdout)

            # 记录实验结果
            experiment_result = {
                'experiment_id': experiment_id,
                'config_file': config_file,
                'gpu_id': gpu_id,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration': duration,
                'success': success,
                'metrics': metrics,
                'stdout': stdout[-2000:] if stdout else "",  # 保留最后2000字符
                'stderr': stderr[-1000:] if stderr else "",   # 保留最后1000字符
                'return_code': result.returncode
            }

            # 保存实验结果
            with open(exp_dir / "experiment_result.json", 'w') as f:
                json.dump(experiment_result, f, indent=2)

            print(f"✅ Experiment {experiment_id} completed successfully!")
            if metrics:
                print(f"   Best mIoU: {metrics.get('best_miou', 'N/A'):.4f}")
                print(f"   Final Loss: {metrics.get('final_loss', 'N/A'):.4f}")

            return experiment_result

        except subprocess.TimeoutExpired:
            print(f"❌ Experiment {experiment_id} timed out after {timeout}s")
            return {
                'experiment_id': experiment_id,
                'config_file': config_file,
                'success': False,
                'error': 'timeout',
                'duration': timeout
            }

        except Exception as e:
            print(f"❌ Experiment {experiment_id} failed: {e}")
            return {
                'experiment_id': experiment_id,
                'config_file': config_file,
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }

    def extract_metrics_from_output(self, stdout: str) -> Dict:
        """从输出中提取性能指标"""
        metrics = {}

        if not stdout:
            return metrics

        lines = stdout.split('\\n')
        for line in lines:
            # 提取最佳mIoU
            if 'Best mIoU:' in line:
                try:
                    metrics['best_miou'] = float(line.split('Best mIoU:')[1].split()[0])
                except:
                    pass

            # 提取最终loss
            if 'Final Loss:' in line:
                try:
                    metrics['final_loss'] = float(line.split('Final Loss:')[1].split()[0])
                except:
                    pass

            # 提取训练时间
            if 'Training completed in' in line:
                try:
                    time_str = line.split('Training completed in')[1].split()[0]
                    metrics['training_time'] = float(time_str)
                except:
                    pass

        return metrics

    def run_experiment_matrix(self,
                              config_pattern: str = "configs/*.yaml",
                              max_parallel: int = 1,
                              timeout: int = 7200) -> List[Dict]:
        """运行实验矩阵"""
        # 加载所有配置
        configs = self.load_experiment_configs(config_pattern)
        print(f"Found {len(configs)} experiment configurations")

        if not configs:
            print("No valid configurations found!")
            return []

        # 运行实验
        results = []
        for i, config in enumerate(configs):
            print(f"\\nProgress: {i+1}/{len(configs)}")
            result = self.run_single_experiment(config, timeout)
            results.append(result)
            self.experiment_log.append(result)

            # 保存中间结果
            self.save_experiment_summary(results)

        print(f"\\n🎉 All experiments completed!")
        print(f"Results saved to: {self.results_dir}")

        return results

    def save_experiment_summary(self, results: List[Dict]):
        """保存实验总结"""
        summary_file = self.results_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        # 生成CSV报告
        self.generate_csv_report(results)

        # 生成可视化报告
        self.generate_visualization_report(results)

    def generate_csv_report(self, results: List[Dict]):
        """生成CSV格式的实验报告"""
        rows = []
        for result in results:
            if result['success']:
                metrics = result.get('metrics', {})
                row = {
                    'experiment_id': result['experiment_id'],
                    'best_miou': metrics.get('best_miou', 0.0),
                    'final_loss': metrics.get('final_loss', 0.0),
                    'duration_hours': result.get('duration', 0) / 3600,
                    'success': result['success']
                }
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_file = self.results_dir / "experiment_results.csv"
            df.to_csv(csv_file, index=False)

            # 排序并显示top结果
            df_sorted = df.sort_values('best_miou', ascending=False)
            print(f"\\n📊 Top 5 Results:")
            print(df_sorted.head().to_string(index=False))

    def generate_visualization_report(self, results: List[Dict]):
        """生成可视化报告"""
        successful_results = [r for r in results if r['success']]
        if len(successful_results) < 2:
            return

        # 准备数据
        exp_ids = [r['experiment_id'] for r in successful_results]
        mious = [r.get('metrics', {}).get('best_miou', 0) for r in successful_results]
        durations = [r.get('duration', 0) / 3600 for r in successful_results]  # 转换为小时

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # mIoU对比
        ax1.bar(range(len(exp_ids)), mious)
        ax1.set_xlabel('Experiments')
        ax1.set_ylabel('Best mIoU')
        ax1.set_title('Experiment Performance Comparison')
        ax1.set_xticks(range(len(exp_ids)))
        ax1.set_xticklabels(exp_ids, rotation=45, ha='right')

        # 训练时间对比
        ax2.bar(range(len(exp_ids)), durations)
        ax2.set_xlabel('Experiments')
        ax2.set_ylabel('Duration (hours)')
        ax2.set_title('Training Time Comparison')
        ax2.set_xticks(range(len(exp_ids)))
        ax2.set_xticklabels(exp_ids, rotation=45, ha='right')

        plt.tight_layout()

        # 保存图表
        plot_file = self.results_dir / "experiment_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Visualization saved to: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='Run experiment matrix')
    parser.add_argument('--config_pattern', default='configs/*.yaml',
                        help='Pattern to match config files')
    parser.add_argument('--results_dir', default='results/experiment_matrix',
                        help='Directory to save results')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0],
                        help='GPU IDs to use')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Timeout per experiment in seconds')
    parser.add_argument('--max_parallel', type=int, default=1,
                        help='Maximum parallel experiments')

    args = parser.parse_args()

    # 创建实验运行器
    runner = ExperimentRunner(
        results_dir=args.results_dir,
        gpu_ids=args.gpu_ids
    )

    # 运行实验矩阵
    results = runner.run_experiment_matrix(
        config_pattern=args.config_pattern,
        max_parallel=args.max_parallel,
        timeout=args.timeout
    )

    # 输出总结
    successful = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"\\n📋 Experiment Summary:")
    print(f"   Total experiments: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {total - successful}")
    print(f"   Success rate: {successful/total*100:.1f}%")


if __name__ == "__main__":
    main()
