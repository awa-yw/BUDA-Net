#!/usr/bin/env bash
set -e

echo "=============================="
echo " BUDA-Net Ablation Experiments"
echo "=============================="

############################
# 4. Full Model (BUDA-Net)
############################
EXP=full
echo "Running EXP = ${EXP}"

python train.py \
  --stage1_epochs 100 \
  --stage2_epochs 0 \
  --batch_size 4 \
  --lr 2e-4 \
  --repair_weight 0.5 \
  --ablation full \
  --save_dir checkpoints/${EXP} \
  --log_dir logs/${EXP}

############################
# 1. Deblur Only
############################
EXP=deblur_only
echo "Running EXP = ${EXP}"

python train.py \
  --stage1_epochs 100 \
  --stage2_epochs 0 \
  --batch_size 4 \
  --lr 2e-4 \
  --ablation deblur_only \
  --save_dir checkpoints/${EXP} \
  --log_dir logs/${EXP}

echo "Finished ${EXP}"
echo "------------------------------"


############################
# 2. Fixed Fusion
############################
EXP=fixed_fusion
echo "Running EXP = ${EXP}"

python train.py \
  --stage1_epochs 100 \
  --stage2_epochs 0 \
  --batch_size 4 \
  --lr 2e-4 \
  --ablation fixed_fusion \
  --save_dir checkpoints/${EXP} \
  --log_dir logs/${EXP}

echo "Finished ${EXP}"
echo "------------------------------"


############################
# 3. No Mask
############################
EXP=no_m
echo "Running EXP = ${EXP}"

python train.py \
  --stage1_epochs 100 \
  --stage2_epochs 0 \
  --batch_size 4 \
  --lr 2e-4 \
  --ablation no_m \
  --save_dir checkpoints/${EXP} \
  --log_dir logs/${EXP}

echo "Finished ${EXP}"
echo "------------------------------"



echo "Finished ${EXP}"
echo "=============================="
echo " All experiments completed 🎉"
echo "=============================="
