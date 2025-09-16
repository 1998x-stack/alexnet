# -*- coding: utf-8 -*-
"""Data transforms, incl. AlexNet Lighting (PCA color jitter) and TenCrop eval."""
from __future__ import annotations
from typing import Tuple, Optional, Callable
import math
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class Lighting(object):
    """AlexNet 的 PCA 颜色扰动（Lighting）.

    论文中对 ImageNet 统计得到的 RGB 特征值/向量，这里给出常用近似常数。
    """
    # 常用预计算（TorchVision 实现思路）
    eigval = torch.tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])

    def __init__(self, alphastd: float = 0.1) -> None:
        self.alphastd = alphastd

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.alphastd <= 0:
            return img
        if not isinstance(img, Image.Image):
            # 假定已是 Tensor[C,H,W]，则直接在 Tensor 上操作
            x = img
        else:
            x = TF.to_tensor(img)
        # 采样 alpha ~ N(0, alphastd)
        alpha = x.new_empty(3).normal_(0, self.alphastd)
        rgb = (Lighting.eigvec * alpha * Lighting.eigval).sum(dim=1)
        x = x + rgb.view(3, 1, 1)
        x = torch.clamp(x, 0.0, 1.0)
        if isinstance(img, Image.Image):
            return TF.to_pil_image(x)
        return x


def _resize_to_256_shorter() -> T.Compose:
    # 将短边缩放到 256，保持比例，接近论文常见复现
    return T.Compose([
        T.Resize(256, interpolation=TF.InterpolationMode.BILINEAR),
    ])


def build_train_transform(
    img_size: int = 224,
    use_lighting: bool = True,
    lighting_std: float = 0.1,
    hflip_p: float = 0.5,
    normalize: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.485,0.456,0.406),(0.229,0.224,0.225)),
    warp_square_256: bool = False,
) -> T.Compose:
    """训练增强：随机裁 224、翻转、Lighting、Normalize."""
    pre = T.Resize((256, 256)) if warp_square_256 else _resize_to_256_shorter()
    aug = [
        pre,
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=hflip_p),
    ]
    if use_lighting:
        aug.append(Lighting(alphastd=lighting_std))
    aug.extend([
        T.ToTensor(),
        T.Normalize(mean=normalize[0], std=normalize[1]),
    ])
    return T.Compose(aug)


def build_eval_transform(
    img_size: int = 224,
    normalize: Tuple[Tuple[float,float,float], Tuple[float,float,float]] = ((0.485,0.456,0.406),(0.229,0.224,0.225)),
    center_256: bool = True,
    warp_square_256: bool = False,
) -> T.Compose:
    """评估增强：resize 到 256，中心裁 224（或直接 224），Normalize."""
    pre = T.Resize((256, 256)) if warp_square_256 else (
        T.Resize(256) if center_256 else T.Resize(img_size)
    )
    return T.Compose([
        pre,
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=normalize[0], std=normalize[1]),
    ])


def build_tencrop_eval_transform(
    img_size: int = 224,
    normalize: Tuple[Tuple[float,float,float], Tuple[float,float,float]] = ((0.485,0.456,0.406),(0.229,0.224,0.225)),
    warp_square_256: bool = False,
) -> Callable:
    """10-crop 评估：四角+中心及其水平翻转，共 10 个裁剪."""
    pre = T.Resize((256, 256)) if warp_square_256 else T.Resize(256)
    base = T.Compose([
        pre,
        T.TenCrop(img_size),  # -> tuple of PIL
    ])

    post = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=normalize[0], std=normalize[1]),
    ])

    def fn(img):
        crops = base(img)
        crops = [post(c) for c in crops]
        return torch.stack(crops, dim=0)  # [10, C, H, W]
    return fn
