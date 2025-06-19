"""
配置管理系统 - 支持模块化组合和参数化控制
"""
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class BackboneConfig:
    """主干网络配置"""
    name: str = "mobilenetv3_small"
    pretrained: bool = True
    freeze_bn: bool = False
    rein_insertion_points: list = None  # R机制插入点

    def __post_init__(self):
        if self.rein_insertion_points is None:
            self.rein_insertion_points = []

@dataclass
class ReinConfig:
    """R机制配置"""
    enabled: bool = False
    mechanism_type: str = "rein"  # rein, token_merging, attention
    insertion_strategy: str = "multi_point"  # single_point, multi_point
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class HeadConfig:
    """分割头配置"""
    name: str = "fcn_head"  # fcn_head, aspp_head, ocr_head
    num_classes: int = 21
    dropout_ratio: float = 0.1
    feature_enhancement: str = "aspp"  # aspp, attention, transformer

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 8
    learning_rate: float = 0.01
    epochs: int = 100
    loss_type: str = "cross_entropy"  # cross_entropy, dice, lovasz
    optimizer: str = "sgd"  # sgd, adam, ranger

@dataclass
class ExperimentConfig:
    """完整实验配置"""
    experiment_name: str = "exp_baseline"
    backbone: BackboneConfig = None
    rein: ReinConfig = None
    head: HeadConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.backbone is None:
            self.backbone = BackboneConfig()
        if self.rein is None:
            self.rein = ReinConfig()
        if self.head is None:
            self.head = HeadConfig()
        if self.training is None:
            self.training = TrainingConfig()

class ConfigManager:
    """配置管理器"""

    @staticmethod
    def save_config(config: ExperimentConfig, config_path: str) -> None:
        """保存配置到文件"""
        config_dict = asdict(config)

        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif config_path.endswith('.json'):
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError("配置文件格式仅支持 .yaml, .yml, .json")

    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        """从文件加载配置"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("配置文件格式仅支持 .yaml, .yml, .json")

        # 递归构建配置对象
        backbone_config = BackboneConfig(**config_dict.get('backbone', {}))
        rein_config = ReinConfig(**config_dict.get('rein', {}))
        head_config = HeadConfig(**config_dict.get('head', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))

        return ExperimentConfig(
            experiment_name=config_dict.get('experiment_name', 'exp_default'),
            backbone=backbone_config,
            rein=rein_config,
            head=head_config,
            training=training_config
        )

    @staticmethod
    def create_baseline_configs() -> Dict[str, ExperimentConfig]:
        """创建基线配置集合"""
        configs = {}

        # MobileNetV3 baseline
        configs['mobilenetv3_baseline'] = ExperimentConfig(
            experiment_name='mobilenetv3_baseline',
            backbone=BackboneConfig(name='mobilenetv3_small', pretrained=True),
            rein=ReinConfig(enabled=False),
            head=HeadConfig(name='fcn_head', feature_enhancement='aspp'),
            training=TrainingConfig(batch_size=8, learning_rate=0.01)
        )

        # MobileNetV3 + Rein
        configs['mobilenetv3_rein'] = ExperimentConfig(
            experiment_name='mobilenetv3_rein',
            backbone=BackboneConfig(name='mobilenetv3_small', pretrained=True),
            rein=ReinConfig(enabled=True, mechanism_type='rein'),
            head=HeadConfig(name='fcn_head', feature_enhancement='aspp'),
            training=TrainingConfig(batch_size=8, learning_rate=0.01)
        )

        # PIDNet baseline
        configs['pidnet_baseline'] = ExperimentConfig(
            experiment_name='pidnet_baseline',
            backbone=BackboneConfig(name='pidnet_s', pretrained=True),
            rein=ReinConfig(enabled=False),
            head=HeadConfig(name='fcn_head', feature_enhancement='none'),
            training=TrainingConfig(batch_size=8, learning_rate=0.01)
        )

        return configs

# 使用示例
if __name__ == "__main__":
    # 创建配置管理器
    config_manager = ConfigManager()

    # 生成基线配置
    baseline_configs = config_manager.create_baseline_configs()

    # 保存配置文件
    for name, config in baseline_configs.items():
        config_manager.save_config(config, f"configs/{name}.yaml")
        print(f"已生成配置: configs/{name}.yaml")
