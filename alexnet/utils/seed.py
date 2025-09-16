# -*- coding: utf-8 -*-
"""Seeding utils."""
from __future__ import annotations
import random
import numpy as np
import torch
from typing import Any

def set_global_seed(seed: int = 42) -> None:
    """设置全局随机种子，提升复现性."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为公平，默认不开 cudnn.deterministic（会降低速度），按需在 cfg 控制
    torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int) -> None:
    """DataLoader worker 的种子设置（如需）."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
