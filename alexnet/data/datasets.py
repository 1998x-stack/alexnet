# -*- coding: utf-8 -*-
"""Datasets & dataloaders: local ImageFolder and HF imagefolder (robust splits)."""
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, List, Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets as tvd
import torchvision.transforms as T

from .transforms import (
    build_train_transform,
    build_eval_transform,
    build_tencrop_eval_transform,
)

try:
    from datasets import load_dataset  # HuggingFace datasets
    _HAS_HF = True
except Exception:
    _HAS_HF = False


def _candidate_splits(requested: str) -> List[str]:
    r = requested.lower()
    # 更鲁棒的 split 候选；常见数据集使用 validation 而非 val
    if r in ("val", "validation", "valid"):
        return [requested, "validation", "val", "valid", "test", "dev"]
    if r in ("train", "training"):
        return [requested, "train", "training"]
    if r in ("test", "testing"):
        return [requested, "test", "validation", "val"]
    # 兜底：就按原值尝试
    return [requested]


def _wrap_hf_to_torch(hf_ds, tfm):
    """把 HuggingFace dataset 包一层，返回 (tensor, label)."""
    from PIL import Image

    class HFWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_ds, tfm):
            self.ds = hf_ds
            self.tfm = tfm
            self.classes = None
            if "label" in hf_ds.features and hasattr(hf_ds.features["label"], "names"):
                self.classes = list(hf_ds.features["label"].names)

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            y = int(item.get("label", 0))
            return self.tfm(img), y

    return HFWrapper(hf_ds, tfm)


def _build_imagefolder(
    data_dir: str, split: str, transform
):
    """使用 torchvision ImageFolder 加载本地目录，目录结构需含 split 子目录。"""
    root = Path(data_dir) / split
    ds = tvd.ImageFolder(root=str(root), transform=transform)
    return ds


def _build_hf_imagefolder(hf_name_or_dir: str, split: str, transform):
    """更健壮的 HF 加载：
    - 若传的是本地目录：用 imagefolder 读取
    - 若传的是远程 repo（如 frgfm/imagenette）：自动尝试多种 split 名
    """
    if not _HAS_HF:
        raise RuntimeError("需要安装 datasets 才能使用 HF imagefolder")

    p = Path(hf_name_or_dir)
    if p.exists():
        # 本地目录：按照 imagefolder 读取
        hf_ds = load_dataset("imagefolder", data_dir=str(p), split=split)
        return _wrap_hf_to_torch(hf_ds, transform)

    # 远程：优先直接按 repo id 读取，并尝试 split 候选
    last_err = None
    for cand in _candidate_splits(split):
        try:
            hf_ds = load_dataset(hf_name_or_dir, split=cand)
            return _wrap_hf_to_torch(hf_ds, transform)
        except Exception as e:
            last_err = e

    # 仍失败则再尝试以 imagefolder 协议远程（某些公开桶以路径方式提供）
    try:
        hf_ds = load_dataset("imagefolder", data_dir=hf_name_or_dir, split=split)
        return _wrap_hf_to_torch(hf_ds, transform)
    except Exception:
        # 抛出更清晰的报错，提示可改 engine 或 split
        raise RuntimeError(
            f"无法从 HF 加载数据集 `{hf_name_or_dir}`，尝试的 split={_candidate_splits(split)}。"
            "请确认 split 名称，或将 DATA.engine 设为 `imagenette`/`imagefolder` 并提供正确目录。"
        ) from last_err


def build_dataloaders(cfg: Dict[str, Any]):
    """根据配置构建 train/val dataloader（含 CPU-only 环境优化）."""
    img_size = cfg["DATA"]["img_size"]
    use_lighting = cfg["AUG"]["lighting"]
    lighting_std = cfg["AUG"]["lighting_std"]
    hflip_p = cfg["AUG"]["hflip_p"]
    normalize = (tuple(cfg["AUG"]["mean"]), tuple(cfg["AUG"]["std"]))
    warp_square = cfg["AUG"].get("warp_square_256", False)

    train_tf = build_train_transform(img_size, use_lighting, lighting_std, hflip_p, normalize, warp_square)
    eval_tf = build_eval_transform(img_size, normalize)
    ten_tf = build_tencrop_eval_transform(img_size, normalize)

    engine = cfg["DATA"]["engine"]  # ["imagefolder", "hf_imagefolder", "cifar10", "imagenette"]
    bs = cfg["TRAIN"]["batch_size"]
    workers = int(cfg["DATA"]["workers"])
    pin_mem = torch.cuda.is_available()  # CPU-only 环境禁用 pin_memory 更稳
    persistent = workers > 0

    if engine == "imagefolder":
        train_ds = _build_imagefolder(cfg["DATA"]["train_dir"], "train", train_tf)
        # 常规约定是 "val" 子目录，如无可自定义
        val_split = cfg["DATA"].get("val_split", "val")
        val_ds = _build_imagefolder(cfg["DATA"]["train_dir"], val_split, eval_tf)

    elif engine == "hf_imagefolder":
        train_ds = _build_hf_imagefolder(cfg["DATA"]["hf_name_or_dir"], "train", train_tf)
        val_split = cfg["DATA"].get("val_split", "val")
        val_ds = _build_hf_imagefolder(cfg["DATA"]["hf_name_or_dir"], val_split, eval_tf)

    elif engine == "cifar10":
        train_ds = tvd.CIFAR10(root=cfg["DATA"]["train_dir"], train=True, download=True, transform=train_tf)
        val_ds = tvd.CIFAR10(root=cfg["DATA"]["train_dir"], train=False, download=True, transform=eval_tf)

    elif engine == "imagenette":
        # 明确使用 HuggingFace 的 imagenette 切分：train / validation
        name = cfg["DATA"].get("hf_name_or_dir", "frgfm/imagenette")
        train_ds = _build_hf_imagefolder(name, "train", train_tf)
        val_ds = _build_hf_imagefolder(name, "validation", eval_tf)
    else:
        raise ValueError(f"Unknown DATA.engine: {engine}")

    # DataLoader 构造，带 CPU-only 适配
    common = dict(num_workers=workers, pin_memory=pin_mem, persistent_workers=persistent)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **common)

    num_classes = cfg["MODEL"]["num_classes"]
    tencrop_transform = ten_tf
    return train_loader, val_loader, num_classes, tencrop_transform
