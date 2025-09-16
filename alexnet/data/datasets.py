# -*- coding: utf-8 -*-
"""Datasets & dataloaders: local ImageFolder and HF imagefolder."""
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets as tvd

try:
    from datasets import load_dataset  # HuggingFace datasets
    _HAS_HF = True
except Exception:
    _HAS_HF = False

from .transforms import build_train_transform, build_eval_transform, build_tencrop_eval_transform


def _build_imagefolder(
    data_dir: str, split: str, transform, class_map: Optional[Dict[str, int]] = None
):
    """使用 torchvision ImageFolder 加载本地目录."""
    root = Path(data_dir) / split
    ds = tvd.ImageFolder(root=str(root), transform=transform)
    if class_map:
        ds.class_to_idx = class_map
    return ds


def _build_hf_imagefolder(hf_name_or_dir: str, split: str, transform):
    """加载 HF imagefolder: 
    - 远程: load_dataset('imagefolder', data_files=...) 或 repo_id
    - 本地: load_dataset('imagefolder', data_dir='path')
    """
    if not _HAS_HF:
        raise RuntimeError("需要安装 datasets 才能使用 HF imagefolder")

    if Path(hf_name_or_dir).exists():
        ds = load_dataset("imagefolder", data_dir=hf_name_or_dir, split=split)
    else:
        # 远程 hub: 若为标准名字（如 imagenette/自制数据集）
        try:
            ds = load_dataset(hf_name_or_dir, split=split)
        except Exception:
            # 尝试以 imagefolder 远程结构拉取
            ds = load_dataset("imagefolder", data_dir=hf_name_or_dir, split=split)

    # HuggingFace -> TorchVision 风格（返回 (image, label)）
    from PIL import Image

    class HFWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_ds, tfm):
            self.ds = hf_ds
            self.tfm = tfm
            # 尝试从 features 中解析 label 名称
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
            y = int(item["label"]) if "label" in item else 0
            return self.tfm(img), y

    return HFWrapper(ds, transform)


def build_dataloaders(cfg: Dict[str, Any]):
    """根据配置构建 train/val dataloader."""
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
    workers = cfg["DATA"]["workers"]
    pin_mem = True

    if engine == "imagefolder":
        train_ds = _build_imagefolder(cfg["DATA"]["train_dir"], "train", train_tf)
        val_ds = _build_imagefolder(cfg["DATA"]["train_dir"], "val", eval_tf)
    elif engine == "hf_imagefolder":
        train_ds = _build_hf_imagefolder(cfg["DATA"]["hf_name_or_dir"], "train", train_tf)
        val_ds = _build_hf_imagefolder(cfg["DATA"]["hf_name_or_dir"], "val", eval_tf)
    elif engine == "cifar10":
        # 小规模 demo
        train_ds = tvd.CIFAR10(root=cfg["DATA"]["train_dir"], train=True, download=True, transform=train_tf)
        val_ds = tvd.CIFAR10(root=cfg["DATA"]["train_dir"], train=False, download=True, transform=eval_tf)
    elif engine == "imagenette":
        # torchvision 没有直接内置，改用 HF imagefolder 常见镜像（需网络）
        train_ds = _build_hf_imagefolder(cfg["DATA"]["hf_name_or_dir"], "train", train_tf)
        val_ds = _build_hf_imagefolder(cfg["DATA"]["hf_name_or_dir"], "validation", eval_tf)
    else:
        raise ValueError(f"Unknown DATA.engine: {engine}")

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin_mem, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin_mem
    )

    # 10-crop 评估时用到的 transform（按需在 evaluator 中调用）
    tencrop_transform = ten_tf

    num_classes = cfg["MODEL"]["num_classes"]
    return train_loader, val_loader, num_classes, tencrop_transform
