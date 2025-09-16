# -*- coding: utf-8 -*-
"""Optional higher-level DataModule wrapper."""
from __future__ import annotations
from typing import Any, Dict, Tuple
from .datasets import build_dataloaders

class DataModule:
    """封装数据构建，方便扩展多数据源."""
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.num_classes = None
        self.tencrop_transform = None

    def setup(self) -> None:
        tl, vl, nc, ten = build_dataloaders(self.cfg)
        self.train_loader, self.val_loader, self.num_classes, self.tencrop_transform = tl, vl, nc, ten
