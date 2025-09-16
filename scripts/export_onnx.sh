#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -u main.py --config configs/alexnet_imagenet.yaml --mode export --resume runs/imagenet/ckpts/best.pth --export-onnx exports/alexnet_imagenet.onnx
