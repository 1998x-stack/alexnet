# -*- coding: utf-8 -*-
"""Classification metrics: Top-1 / Top-5 accuracy."""
from __future__ import annotations
from typing import Tuple
import torch

def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1,5)) -> Tuple[torch.Tensor, ...]:
    """计算 Top-k 准确率.
    Args:
        logits: [N, C]
        targets: [N]
        topk: k 列表
    Returns:
        每个 k 的准确率（百分比，0~100）
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [N, maxk]
    pred = pred.t()  # [maxk, N]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return tuple(res)
