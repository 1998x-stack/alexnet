#!/usr/bin/env bash
set -e
source .venv/bin/activate
# 单 crop 评估
python -u main.py --config configs/alexnet_imagenette.yaml --mode eval --resume runs/imagenette/ckpts/best.pth

# 导出 ONNX
python -u main.py --config configs/alexnet_imagenette.yaml --mode export --resume runs/imagenette/ckpts/best.pth --export-onnx exports/alexnet_imagenette.onnx
