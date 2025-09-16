#!/usr/bin/env bash
set -e
source .venv/bin/activate

CFG=configs/alexnet_imagenette.yaml

# 单机单卡
python -u main.py --config $CFG --mode train

# 消融示例（无 LRN）
python -u main.py --config $CFG --mode train --opts MODEL.kwargs.use_lrn=false OUTPUT.ckpt_dir=runs/imagenette/ckpts_nolrn OUTPUT.tb_dir=runs/imagenette/tb_nolrn OUTPUT.log_dir=runs/imagenette/logs_nolrn

# 消融示例（无 Overlap Pool）
python -u main.py --config $CFG --mode train --opts MODEL.kwargs.overlap_pool=false OUTPUT.ckpt_dir=runs/imagenette/ckpts_nooverlap OUTPUT.tb_dir=runs/imagenette/tb_nooverlap OUTPUT.log_dir=runs/imagenette/logs_nooverlap

# 消融示例（无 Dropout）
python -u main.py --config $CFG --mode train --opts MODEL.kwargs.dropout_p=0.0 OUTPUT.ckpt_dir=runs/imagenette/ckpts_nodrop OUTPUT.tb_dir=runs/imagenette/tb_nodrop OUTPUT.log_dir=runs/imagenette/logs_nodrop
