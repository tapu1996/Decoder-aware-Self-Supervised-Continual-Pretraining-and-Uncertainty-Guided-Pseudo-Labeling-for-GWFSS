#!/usr/bin/env bash

set -x
CONFIG=$1
PRETRAIN=$2  # pretrained model
GPUS=$3
PY_ARGS=${@:4}
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun --gres=gpu:${GPUS} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train.py ${CONFIG} --launcher="none" --cfg-options model.backbone.init_cfg.type=Pretrained model.backbone.init_cfg.checkpoint=$PRETRAIN ${PY_ARGS}
