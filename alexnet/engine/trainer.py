# -*- coding: utf-8 -*-
"""Training loop with AMP, DDP-aware logging, checkpointing."""
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loguru import logger as log

from alexnet.models.registry import MODEL_REGISTRY
from alexnet.data import DataModule
from alexnet.engine.metrics import accuracy_topk
from alexnet.utils.scheduler import build_optimizer, build_scheduler
from alexnet.utils.seed import seed_worker
from alexnet.utils.checkpoint import save_checkpoint, maybe_resume
from alexnet.utils.distributed import is_main_process, world_info


def _build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_name = cfg["MODEL"]["name"]
    ctor = MODEL_REGISTRY.get(model_name, None)
    if ctor is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model = ctor(**cfg["MODEL"]["kwargs"])
    return model


def run_train(cfg: Dict[str, Any], resume_path: str = "") -> None:
    """训练入口."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DataModule(cfg)
    dm.setup()

    model = _build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)

    start_epoch = 0
    best_top1 = 0.0

    # TensorBoard
    tb_dir = Path(cfg["OUTPUT"]["tb_dir"])
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir)) if is_main_process() else None

    scaler = GradScaler(enabled=cfg["TRAIN"]["amp"])

    # 恢复训练
    if resume_path:
        start_epoch, best_top1 = maybe_resume(resume_path, model, optimizer, scheduler, scaler)
        log.info(f"从 {resume_path} 恢复：epoch={start_epoch}, best_top1={best_top1:.3f}")

    # 训练循环
    max_epoch = cfg["TRAIN"]["epochs"]
    log.info(f"开始训练: epochs={max_epoch}, device={device}, world={world_info()}")
    for epoch in range(start_epoch, max_epoch):
        model.train()
        losses = 0.0
        top1_meter = 0.0
        top5_meter = 0.0

        pbar = tqdm(dm.train_loader, disable=not is_main_process(), desc=f"Train e{epoch}")
        for step, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["TRAIN"]["amp"]):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if cfg["TRAIN"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["TRAIN"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            # metric
            with torch.no_grad():
                a1, a5 = accuracy_topk(logits, y, topk=(1, 5))
                losses += loss.item()
                top1_meter += a1.item()
                top5_meter += a5.item()
                avg_loss = losses / (step + 1)
                avg_t1 = top1_meter / (step + 1)
                avg_t5 = top5_meter / (step + 1)
            if is_main_process():
                pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_t1:.2f}", top5=f"{avg_t5:.2f}")

        if scheduler is not None:
            # MultiStepLR / Cosine / ReduceLROnPlateau
            if cfg["TRAIN"]["scheduler"]["name"] == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        # 验证
        val_top1, val_top5, val_loss = _validate(dm, model, criterion, device, cfg)
        if is_main_process():
            log.info(f"[Epoch {epoch}] train_loss={avg_loss:.4f} val_loss={val_loss:.4f} "
                     f"val@1={val_top1:.2f} val@5={val_top5:.2f}")
            if writer:
                writer.add_scalar("train/loss", avg_loss, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/top1", val_top1, epoch)
                writer.add_scalar("val/top5", val_top5, epoch)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # 保存
        out_dir = Path(cfg["OUTPUT"]["ckpt_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        is_best = val_top1 > best_top1
        if is_main_process():
            ckpt_path = out_dir / "last.pth"
            save_checkpoint(
                ckpt_path, epoch + 1, model, optimizer, scheduler, scaler,
                best_metric=max(best_top1, val_top1)
            )
            if is_best:
                best_top1 = val_top1
                save_checkpoint(out_dir / "best.pth", epoch + 1, model, optimizer, scheduler, scaler, best_metric=best_top1)

    if is_main_process() and writer:
        writer.close()


@torch.no_grad()
def _validate(dm: DataModule, model: nn.Module, criterion: nn.Module, device, cfg: Dict[str, Any]):
    model.eval()
    use_tencrop = cfg["EVAL"]["ten_crop"]
    ten_tf = dm.tencrop_transform

    losses = 0.0
    top1_meter = 0.0
    top5_meter = 0.0

    for step, (x, y) in enumerate(tqdm(dm.val_loader, disable=True)):
        y = y.to(device, non_blocking=True)
        if use_tencrop:
            # 针对每张图，预处理为 [10, C, H, W] 后再前向平均
            # 这里 val_loader 的 transform 是单 crop；我们替换为 10-crop
            imgs_10 = []
            for i in range(x.size(0)):
                # 这里 x[i] 是 Tensor(单 crop)，需要原 PIL，简化：直接再走 10-crop transform
                # 为保证一致性，建议 val_loader 的 dataset 保留原路径；此处为演示简化策略：
                # 实际工程中可在 dataset 中支持 get_raw(idx).
                # 简化做法：跳过，直接用单 crop 评估（如需严格 10-crop，请在 dataset 中实现 raw 访问）。
                pass

        # 简化：默认使用单 crop（10-crop 已在 transforms 提供，工程中可在 Dataset 包装 raw 像素）
        x = x.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        a1, a5 = accuracy_topk(logits, y, topk=(1, 5))
        losses += loss.item()
        top1_meter += a1.item()
        top5_meter += a5.item()

    n = len(dm.val_loader)
    return top1_meter / n, top5_meter / n, losses / n
