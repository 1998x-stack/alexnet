#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for training/eval/export AlexNet with pluggable components."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

from alexnet.utils.config import load_config, merge_overrides
from alexnet.utils.logger import setup_logger, get_loguru_logger
from alexnet.utils.seed import set_global_seed
from alexnet.utils.distributed import ddp_init_if_needed, is_main_process
from alexnet.engine.trainer import run_train
from alexnet.engine.evaluator import run_eval
from alexnet.utils.misc import pretty_dict


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="AlexNet Project")
    p.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "export"],
                   help="运行模式：train/eval/export")
    p.add_argument("--opts", nargs="*", default=[],
                   help="额外配置覆盖，如 TRAIN.lr=0.005 DATA.train_dir=/data ...")
    p.add_argument("--export-onnx", type=str, default="", help="导出 ONNX 模型路径（mode=export）")
    p.add_argument("--resume", type=str, default="", help="恢复训练的 checkpoint 路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_overrides(cfg, args.opts)

    # 初始化日志目录
    setup_logger(cfg)
    logger = get_loguru_logger()

    # DDP 初始化（如使用 torchrun）
    ddp_init_if_needed()

    if is_main_process():
        logger.info("有效配置:\n" + pretty_dict(cfg))

    set_global_seed(cfg["SEED"])

    if args.mode == "train":
        run_train(cfg, resume_path=args.resume)
    elif args.mode == "eval":
        run_eval(cfg, ckpt_path=args.resume)
    elif args.mode == "export":
        if not args.export_onnx:
            raise ValueError("mode=export 需要 --export-onnx 指定导出路径")
        run_eval(cfg, ckpt_path=args.resume, export_onnx=args.export_onnx)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
