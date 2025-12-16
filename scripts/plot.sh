#!/bin/bash
python scripts/plot_train_curve.py \
  --log logs/full/train_log.csv \
  --out figures/train_full_curve.png

python scripts/plot_ablation_compare.py \
  --inputs \
    "BUDA-Net:logs/full/train_log.csv" \
    "No-M:logs/no_m/train_log.csv" \
    "Fixed-Fusion:logs/fixed_fusion/train_log.csv" \
  --out figures/ablation_loss_compare.png

python scripts/plot_metric_curve.py \
  --log logs/full/train_log.csv \
  --out figures/psnr_ssim_curve.png

python scripts/visualize_results.py \
  --ckpt checkpoints/full/final.pth \
  --data_root datasets/GoPro \
  --indices 3 12 45 \
  --out_dir figures/visual
