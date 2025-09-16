# -*- coding: utf-8 -*-
"""Minimal DDP helpers (single-process torchrun support)."""
from __future__ import annotations
import os
import torch

def ddp_init_if_needed() -> None:
    """若使用 torchrun, 则初始化进程组."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def world_info() -> str:
    return f"rank={os.environ.get('RANK','0')}, world_size={os.environ.get('WORLD_SIZE','1')}"
