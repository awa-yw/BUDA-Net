# test.py
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import DualBranchDeblurRepairNet
from datasets.gopro_dataset import GoProDataset
from datasets.uhd_dip_dataset import UHDDIPDataset

device = "cuda:3" if torch.cuda.is_available() else "cpu"


# =====================
# Metrics
# =====================
def psnr(pred, gt, max_val=1.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(pred, gt, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean()
    mu_y = gt.mean()
    sigma_x = pred.var()
    sigma_y = gt.var()
    sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean()

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return num / den


# =====================
# Evaluation
# =====================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    psnr_list, ssim_list = [], []

    for blur, sharp in loader:
        blur = blur.to(device)
        sharp = sharp.to(device)

        out = model(blur, ablation=args.ablation)

        pred = out["I_out"].clamp(0, 1)

        for i in range(pred.size(0)):
            psnr_list.append(psnr(pred[i], sharp[i]).item())
            ssim_list.append(ssim(pred[i], sharp[i]).item())

    return np.mean(psnr_list), np.mean(ssim_list)


# =====================
# Main
# =====================
def main(args):

    # ---- model ----
    model = DualBranchDeblurRepairNet(base=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # ---- dataset ----
    if args.dataset.lower() == "gopro":
        test_set = GoProDataset(
            root=args.data_root,
            split="test",
            crop_size=args.crop_size,
            training=False
        )
    elif args.dataset.lower() == "uhd":
        test_set = UHDDIPDataset(
            root=args.data_root,
            split="test",
            crop_size=args.crop_size,
            training=False
        )
    else:
        raise ValueError("dataset must be 'gopro' or 'uhd'")

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # ---- eval ----
    avg_psnr, avg_ssim = evaluate(model, test_loader)

    print("=" * 40)
    print(f" Dataset : {args.dataset.upper()}")
    print(f" Checkpt : {args.ckpt}")
    print(f" PSNR    : {avg_psnr:.2f} dB")
    print(f" SSIM    : {avg_ssim:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BUDA-Net Testing Script")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gopro", "uhd"],
                        help="Dataset to evaluate on")

    parser.add_argument("--data_root", type=str, required=True,
                        help="Root path to dataset (e.g. datasets/GoPro)")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Crop size (not used in test, kept for compatibility)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=48,
                        help="Base channels of BUDA-Net")
    parser.add_argument(
    "--ablation",
        type=str,
        default="deblur_only",
        choices=["full", "deblur_only", "no_m", "fixed_fusion"]
    )

    args = parser.parse_args()
    main(args)
