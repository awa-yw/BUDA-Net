#!/bin/bash

EXP=deblur_only

python train.py \
  --stage1_epochs 30 \
  --stage2_epochs 0 \
  --batch_size 4 \
  --lr 2e-3 \
  --ablation deblur_only \
  --save_dir checkpoints/${EXP} \
  --log_dir logs/${EXP}
