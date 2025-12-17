#!/bin/bash

EXP=full

python train.py \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --batch_size 4 \
  --lr 2e-3 \
  --ablation full \
  --save_dir checkpoints/${EXP} \
  --log_dir logs/${EXP}
