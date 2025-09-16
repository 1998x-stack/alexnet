# -*- coding: utf-8 -*-
"""Simple registries for pluggable models."""
from __future__ import annotations
from typing import Callable, Dict

MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str) -> Callable:
    """装饰器：注册模型构造函数."""
    def deco(fn: Callable) -> Callable:
        MODEL_REGISTRY[name] = fn
        return fn
    return deco
