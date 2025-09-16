# AlexNet Pluggable Project

- Paper-faithful AlexNet with LRN, overlapping pooling, dropout, lighting(PCA) jitter.
- Pluggable registries for model/dataset/optimizer/scheduler.
- AMP + DDP ready, TensorBoard + Loguru logging.
- 10-crop evaluation and ablation configs.

```
alexnet_proj/
├── README.md
├── requirements.txt
├── main.py
├── configs/
│   ├── alexnet_imagenet.yaml
│   ├── alexnet_imagenette.yaml
│   ├── alexnet_cifar10.yaml
│   └── ablations/
│       ├── no_lrn.yaml
│       ├── no_overlap_pool.yaml
│       ├── no_dropout.yaml
│       ├── no_lighting.yaml
│       └── center_crop_eval.yaml
├── alexnet/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── alexnet.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── transforms.py
│   │   └── datamodule.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       ├── seed.py
│       ├── checkpoint.py
│       ├── distributed.py
│       ├── scheduler.py
│       └── misc.py
└── scripts/
    ├── install.sh
    ├── tensorboard.sh
    ├── train_imagenette.sh
    ├── train_cifar10.sh
    ├── train_imagenet.sh
    ├── eval.sh
    └── export_onnx.sh

```

## Quickstart
```bash
bash scripts/install.sh
# Small-scale sanity training on Imagenette (from HF imagefolder or local)
bash scripts/train_imagenette.sh
# CIFAR-10 demo
bash scripts/train_cifar10.sh
```