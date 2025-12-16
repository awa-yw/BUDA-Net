#!/bin/bash

EXP=no_m

python train.py \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --batch_size 4 \
  --lr 2e-4 \
  --ablation no_m \
  --save_dir checkpoints/${EXP} \
  --log_dir logs/${EXP}
