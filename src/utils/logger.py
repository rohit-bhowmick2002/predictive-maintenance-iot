"""
Structured logging for the Predictive Maintenance pipeline.
Uses loguru for developer-friendly logging with file rotation.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/pdm.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure structured logging with console and file sinks.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        rotation: When to rotate log files
        retention: How long to keep old logs
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler with structured JSON
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        compression="gz",
        serialize=False,
        backtrace=True,
        diagnose=False,
    )

    return logger


# Initialize logger
setup_logger()

__all__ = ["logger"]
