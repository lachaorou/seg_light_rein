"""
日志记录工具
"""
import logging
import os
from datetime import datetime


def setup_logger(exp_dir: str, name: str = 'seg_light_rein') -> logging.Logger:
    """设置日志记录器"""

    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    log_file = os.path.join(exp_dir, 'logs', f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
