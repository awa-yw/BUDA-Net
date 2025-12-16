#!/bin/bash

EXP=full

python test.py \
  --dataset gopro \
  --data_root datasets/gopro \
  --ckpt checkpoints/${EXP}/best_gopro.pth \
  --batch_size 1
