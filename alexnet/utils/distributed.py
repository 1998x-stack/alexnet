# -*- coding: utf-8 -*-
"""Minimal DDP helpers with CPU-safe backend fallback."""
from __future__ import annotations
import os
import torch

def ddp_init_if_needed() -> None:
    """若使用 torchrun, 则初始化进程组；CPU-only 环境自动改用 gloo 避免 NCCL 报错."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def world_info() -> str:
    return f"rank={os.environ.get('RANK','0')}, world_size={os.environ.get('WORLD_SIZE','1')}"
