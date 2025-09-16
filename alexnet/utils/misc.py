# -*- coding: utf-8 -*-
"""Misc helpers."""
from __future__ import annotations
from typing import Dict, Any
import json

def pretty_dict(d: Dict[str, Any]) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2)
