"""
config/logging_config.py
========================
Cấu hình logging toàn hệ thống sử dụng loguru.
"""

import sys
from loguru import logger


def setup_logging(level: str = "INFO"):
    """
    Cấu hình loguru với output ra console và file xoay vòng.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    logger.remove()  # Xoá handler mặc định

    # Console handler – màu sắc, dễ đọc
    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    )

    # File handler – lưu log xoay vòng 10MB, giữ 7 ngày
    logger.add(
        "logs/fahasa_ai_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        enqueue=True  # Thread-safe
    )

    logger.info(f"📝 Logging khởi tạo với level={level}")
