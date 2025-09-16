#!/usr/bin/env bash
set -e
source .venv/bin/activate

CFG=configs/alexnet_imagenet.yaml

# DDP 示例（按主机 GPU 数调整 nproc_per_node）
torchrun --nproc_per_node=8 main.py --config $CFG --mode train

# 或单卡
# python -u main.py --config $CFG --mode train

# 消融合集（建议分机器跑）
# 1) no LRN
torchrun --nproc_per_node=8 main.py --config $CFG --mode train --opts MODEL.kwargs.use_lrn=false OUTPUT.ckpt_dir=runs/imagenet/nolrn/ckpts OUTPUT.tb_dir=runs/imagenet/nolrn/tb OUTPUT.log_dir=runs/imagenet/nolrn/logs
# 2) no overlap pooling
torchrun --nproc_per_node=8 main.py --config $CFG --mode train --opts MODEL.kwargs.overlap_pool=false OUTPUT.ckpt_dir=runs/imagenet/nooverlap/ckpts OUTPUT.tb_dir=runs/imagenet/nooverlap/tb OUTPUT.log_dir=runs/imagenet/nooverlap/logs
# 3) no dropout
torchrun --nproc_per_node=8 main.py --config $CFG --mode train --opts MODEL.kwargs.dropout_p=0.0 OUTPUT.ckpt_dir=runs/imagenet/nodrop/ckpts OUTPUT.tb_dir=runs/imagenet/nodrop/tb OUTPUT.log_dir=runs/imagenet/nodrop/logs
# 4) no lighting
torchrun --nproc_per_node=8 main.py --config $CFG --mode train --opts AUG.lighting=false OUTPUT.ckpt_dir=runs/imagenet/nolight/ckpts OUTPUT.tb_dir=runs/imagenet/nolight/tb OUTPUT.log_dir=runs/imagenet/nolight/logs
