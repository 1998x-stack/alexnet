# -*- coding: utf-8 -*-
"""YAML config loader + CLI overrides."""
from __future__ import annotations
from typing import Dict, Any, List
import copy
import yaml
from pathlib import Path


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_overrides(cfg: Dict[str, Any], opts: List[str]) -> Dict[str, Any]:
    """将命令行 --opts 形如 A.B=1 C=xx 合并进字典."""
    out = copy.deepcopy(cfg)
    for kv in opts:
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        keys = key.split(".")
        d = out
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        # 尝试类型转换
        v = _auto_type(val)
        d[keys[-1]] = v
    return out


def _auto_type(s: str):
    if s.lower() in ["true", "false"]:
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s
