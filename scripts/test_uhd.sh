#!/bin/bash

EXP=full

python test.py \
  --dataset uhd \
  --data_root datasets/uhd \
  --ckpt checkpoints/${EXP}/best_gopro.pth \
  --batch_size 1
