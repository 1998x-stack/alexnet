# -*- coding: utf-8 -*-
"""Eval / Export entry."""
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from loguru import logger as log

from alexnet.models.registry import MODEL_REGISTRY
from alexnet.data import DataModule
from alexnet.engine.metrics import accuracy_topk
from alexnet.utils.checkpoint import load_model_weights
from alexnet.utils.distributed import is_main_process


def _build_model(cfg: Dict[str, Any]) -> nn.Module:
    ctor = MODEL_REGISTRY.get(cfg["MODEL"]["name"], None)
    if ctor is None:
        raise ValueError(f"Unknown model: {cfg['MODEL']['name']}")
    model = ctor(**cfg["MODEL"]["kwargs"])
    return model


@torch.no_grad()
def run_eval(cfg: Dict[str, Any], ckpt_path: str = "", export_onnx: str = "") -> None:
    """评估或导出 ONNX."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DataModule(cfg)
    dm.setup()

    model = _build_model(cfg).to(device)
    if ckpt_path:
        load_model_weights(ckpt_path, model)
        log.info(f"已加载权重: {ckpt_path}")

    if export_onnx:
        model.eval()
        dummy = torch.randn(1, 3, cfg["DATA"]["img_size"], cfg["DATA"]["img_size"]).to(device)
        onnx_path = Path(export_onnx)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=13, input_names=["input"], output_names=["logits"])
        log.info(f"导出 ONNX 到: {onnx_path}")
        return

    # 简单单 crop 评估
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()

    losses = 0.0
    top1_meter = 0.0
    top5_meter = 0.0
    for step, (x, y) in enumerate(dm.val_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        a1, a5 = accuracy_topk(logits, y, topk=(1, 5))
        losses += loss.item()
        top1_meter += a1.item()
        top5_meter += a5.item()

    n = len(dm.val_loader)
    log.info(f"[Eval] loss={losses/n:.4f} top1={top1_meter/n:.2f} top5={top5_meter/n:.2f}")
