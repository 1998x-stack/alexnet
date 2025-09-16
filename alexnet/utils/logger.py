# -*- coding: utf-8 -*-
"""Loguru + TensorBoard logger setup."""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from loguru import logger


def setup_logger(cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["OUTPUT"]["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(out_dir / "train.log", rotation="5 MB", encoding="utf-8", enqueue=True, level="INFO")
    logger.add(lambda msg: print(msg, end=""))  # 同步到控制台


def get_loguru_logger():
    return logger
