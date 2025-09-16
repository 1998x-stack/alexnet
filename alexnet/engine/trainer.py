# -*- coding: utf-8 -*-
"""Training loop with AMP, DDP-aware logging, checkpointing + robust DataLoader fallback."""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger as log

from alexnet.models.registry import MODEL_REGISTRY
from alexnet.data import DataModule
from alexnet.engine.metrics import accuracy_topk
from alexnet.utils.scheduler import build_optimizer, build_scheduler
from alexnet.utils.checkpoint import save_checkpoint, maybe_resume
from alexnet.utils.distributed import is_main_process, world_info


def _build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_name = cfg["MODEL"]["name"]
    ctor = MODEL_REGISTRY.get(model_name, None)
    if ctor is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model = ctor(**cfg["MODEL"]["kwargs"])
    return model


def _one_epoch_train(dm: DataModule, model: nn.Module, criterion: nn.Module, optimizer, scaler, device, cfg, epoch: int):
    """单个 epoch 的训练，实现为可重试（处理 Bus error -> workers=0 降级）."""
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

    return avg_loss, avg_t1, avg_t5


@torch.no_grad()
def _validate(dm: DataModule, model: nn.Module, criterion: nn.Module, device, cfg: Dict[str, Any]):
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
    return top1_meter / n, top5_meter / n, losses / n


def run_train(cfg: Dict[str, Any], resume_path: str = "") -> None:
    """训练入口（含 DataLoader bus error 自动降级到 workers=0）."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DataModule(cfg)
    dm.setup()

    model = _build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)

    start_epoch = 0
    best_top1 = 0.0

    tb_dir = Path(cfg["OUTPUT"]["tb_dir"]); tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir)) if is_main_process() else None
    scaler = GradScaler(enabled=cfg["TRAIN"]["amp"])

    if resume_path:
        start_epoch, best_top1 = maybe_resume(resume_path, model, optimizer, scheduler, scaler)
        log.info(f"从 {resume_path} 恢复：epoch={start_epoch}, best_top1={best_top1:.3f}")

    max_epoch = cfg["TRAIN"]["epochs"]
    log.info(f"开始训练: epochs={max_epoch}, device={device}, world={world_info()}")

    for epoch in range(start_epoch, max_epoch):
        # —— 训练（带一次性重试策略）——
        tried_fallback = False
        while True:
            try:
                avg_loss, avg_t1, avg_t5 = _one_epoch_train(dm, model, criterion, optimizer, scaler, device, cfg, epoch)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if (("dataloader worker" in msg or "bus error" in msg or "shared memory" in msg) and not tried_fallback):
                    tried_fallback = True
                    old_workers = int(cfg["DATA"]["workers"])
                    cfg["DATA"]["workers"] = 0
                    log.warning(
                        f"检测到 DataLoader 共享内存/Bus error，自动降级: workers {old_workers} -> 0，pin_memory 将由构建逻辑关闭。重建 DataLoader 并重试本 epoch。"
                    )
                    dm = DataModule(cfg); dm.setup()
                    continue
                raise  # 非可恢复错误或已重试过，直接抛出

        # —— 调度器 —— 
        if scheduler is not None:
            if cfg["TRAIN"]["scheduler"]["name"].lower() == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        # —— 验证 —— 
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

        # —— 保存 —— 
        out_dir = Path(cfg["OUTPUT"]["ckpt_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
        is_best = val_top1 > best_top1
        if is_main_process():
            save_checkpoint(out_dir / "last.pth", epoch + 1, model, optimizer, scheduler, scaler,
                            best_metric=max(best_top1, val_top1))
            if is_best:
                best_top1 = val_top1
                save_checkpoint(out_dir / "best.pth", epoch + 1, model, optimizer, scheduler, scaler,
                                best_metric=best_top1)

    if is_main_process() and writer:
        writer.close()
