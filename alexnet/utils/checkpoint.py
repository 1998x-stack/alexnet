# -*- coding: utf-8 -*-
"""Checkpoint save/load helpers."""
from __future__ import annotations
from typing import Tuple
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(path, epoch: int, model: nn.Module, optimizer, scheduler, scaler, best_metric: float):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "best_metric": best_metric,
    }
    torch.save(state, str(path))

def maybe_resume(path: str, model: nn.Module, optimizer, scheduler, scaler) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_metric", 0.0))

def load_model_weights(path: str, model: nn.Module) -> None:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
