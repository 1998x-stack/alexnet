#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -u main.py --config configs/alexnet_cifar10.yaml --mode train
