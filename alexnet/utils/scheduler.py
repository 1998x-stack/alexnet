# -*- coding: utf-8 -*-
"""Build optimizer/scheduler from config (pluggable)."""
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

def build_optimizer(cfg: Dict[str, Any], params: Iterable) -> torch.optim.Optimizer:
    opt_cfg = cfg["TRAIN"]["optimizer"]
    name = opt_cfg["name"].lower()
    lr = cfg["TRAIN"]["lr"]
    wd = cfg["TRAIN"]["weight_decay"]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=False)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer {name}")

def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    s_cfg = cfg["TRAIN"]["scheduler"]
    name = s_cfg["name"].lower()
    if name == "multistep":
        milestones = s_cfg.get("milestones", [30, 60, 80])
        gamma = s_cfg.get("gamma", 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif name == "cosine":
        T = cfg["TRAIN"]["epochs"]
        return CosineAnnealingLR(optimizer, T_max=T)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    else:
        return None
