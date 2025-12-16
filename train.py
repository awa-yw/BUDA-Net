import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time

from model import DualBranchDeblurRepairNet
from datasets.gopro_dataset import GoProDataset
from datasets.uhd_dip_dataset import UHDDIPDataset
from loss import l1_loss, edge_loss
from eval_utils import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# CSV Logger
# =========================
class CSVLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w")
        self.f.write(
            "epoch,stage,loss_deblur,loss_repair,loss_total,"
            "psnr_gopro,ssim_gopro,psnr_uhd,ssim_uhd\n"
        )
        self.f.flush()

    def log(self, epoch, stage, ld, lr, lt,
            psnr_gopro, ssim_gopro, psnr_uhd, ssim_uhd):
        self.f.write(
            f"{epoch},{stage},"
            f"{ld:.6f},{lr:.6f},{lt:.6f},"
            f"{psnr_gopro:.4f},{ssim_gopro:.4f},"
            f"{psnr_uhd:.4f},{ssim_uhd:.4f}\n"
        )
        self.f.flush()

    def close(self):
        self.f.close()


# =========================
# Setup logger (console + file)
# =========================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    fh = logging.FileHandler(os.path.join(log_dir, "train_log.txt"))
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# =========================
# Main
# =========================
def main(args):
    logger = setup_logger(args.log_dir)

    # ---- model ----
    net = DualBranchDeblurRepairNet(base=args.base_channels).to(device)

    # ---- data loaders ----
    gopro_loader = DataLoader(
        GoProDataset("datasets/gopro", "train", args.crop_size, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    test_gopro_loader = DataLoader(
        GoProDataset("datasets/gopro", "test", args.crop_size, False),
        batch_size=1, shuffle=False, num_workers=2
    )

    uhd_loader = DataLoader(
        UHDDIPDataset("datasets/uhd", "train", args.crop_size, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    test_uhd_loader = DataLoader(
        UHDDIPDataset("datasets/uhd", "test", args.crop_size, False),
        batch_size=1, shuffle=False, num_workers=2
    )

    # ---- optimizer ----
    optimizer = optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=1e-4
    )

    # ---- training state ----
    start_stage = 1
    start_epoch_s1 = 0
    start_epoch_s2 = 0
    best_psnr_gopro = -1.0
    best_psnr_uhd = -1.0

    # ---- resume ----
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_stage = ckpt["stage"]
        start_epoch_s1 = ckpt.get("epoch_stage1", 0)
        start_epoch_s2 = ckpt.get("epoch_stage2", 0)
        best_psnr_gopro = ckpt.get("best_psnr_gopro", -1.0)
        best_psnr_uhd = ckpt.get("best_psnr_uhd", -1.0)
        logger.info(f"Resumed from {args.resume}")

    # ---- logger ----
    csv_logger = CSVLogger(os.path.join(args.log_dir, "train_log.csv"))
    os.makedirs(args.save_dir, exist_ok=True)

    def save_latest(stage, e1, e2):
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage": stage,
            "epoch_stage1": e1,
            "epoch_stage2": e2,
            "best_psnr_gopro": best_psnr_gopro,
            "best_psnr_uhd": best_psnr_uhd,
        }, os.path.join(args.save_dir, "latest.pth"))

    # ==================================================
    # Stage-1: GoPro
    # ==================================================
    if start_stage <= 1:
        logger.info("========== Stage-1: GoPro Pre-training ==========")
        for p in net.repair.parameters():
            p.requires_grad = False

        for epoch in range(start_epoch_s1, args.stage1_epochs):
            t0 = time.time()
            net.train()
            loss_sum = 0

            for blur, sharp in gopro_loader:
                blur, sharp = blur.to(device), sharp.to(device)
                out = net(blur)
                I_d = out["I_d"]
                I_out = I_d if args.ablation == "deblur_only" else out["I_out"]

                loss_deblur = l1_loss(I_d, sharp) + 0.1 * edge_loss(I_d, sharp)
                loss = loss_deblur + l1_loss(I_out, sharp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            avg_loss = loss_sum / len(gopro_loader)
            psnr_g, ssim_g = evaluate(net, test_gopro_loader, device)

            csv_logger.log(epoch, "stage1", loss_deblur.item(), 0.0,
                           avg_loss, psnr_g, ssim_g, 0.0, 0.0)

            if psnr_g > best_psnr_gopro:
                best_psnr_gopro = psnr_g
                torch.save(net.state_dict(),
                           os.path.join(args.save_dir, "best_gopro.pth"))

            save_latest(1, epoch + 1, 0)

            logger.info(
                f"[Stage1][{epoch+1}/{args.stage1_epochs}] "
                f"Loss {avg_loss:.4f} | PSNR {psnr_g:.2f} | "
                f"Time {time.time()-t0:.1f}s"
            )

    # ==================================================
    # Stage-2: UHD-DIP
    # ==================================================
    logger.info("========== Stage-2: UHD-DIP Fine-tuning ==========")
    for p in net.parameters():
        p.requires_grad = True

    for epoch in range(start_epoch_s2, args.stage2_epochs):
        t0 = time.time()
        net.train()
        loss_sum = 0

        for blur, sharp in uhd_loader:
            blur, sharp = blur.to(device), sharp.to(device)
            out = net(blur)
            I_d, I_r, M = out["I_d"], out["I_r"], out["M"]

            if args.ablation == "no_m":
                I_out = 0.5 * (I_d + I_r)
                loss_r = (I_r - sharp).abs().mean()
            elif args.ablation == "fixed_fusion":
                I_out = 0.7 * I_d + 0.3 * I_r
                loss_r = (M * (I_r - sharp).abs()).mean()
            else:
                I_out = (1 - M) * I_d + M * I_r
                loss_r = (M * (I_r - sharp).abs()).mean()

            loss_d = l1_loss(I_d, sharp)
            loss = loss_d + l1_loss(I_out, sharp) + args.repair_weight * loss_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(uhd_loader)
        psnr_g, ssim_g = evaluate(net, test_gopro_loader, device)
        psnr_u, ssim_u = evaluate(net, test_uhd_loader, device)

        csv_logger.log(epoch, "stage2", loss_d.item(), loss_r.item(),
                       avg_loss, psnr_g, ssim_g, psnr_u, ssim_u)

        if psnr_g > best_psnr_gopro:
            best_psnr_gopro = psnr_g
            torch.save(net.state_dict(),
                       os.path.join(args.save_dir, "best_gopro.pth"))

        if psnr_u > best_psnr_uhd:
            best_psnr_uhd = psnr_u
            torch.save(net.state_dict(),
                       os.path.join(args.save_dir, "best_uhd.pth"))

        save_latest(2, args.stage1_epochs, epoch + 1)

        logger.info(
            f"[Stage2][{epoch+1}/{args.stage2_epochs}] "
            f"Loss {avg_loss:.4f} | "
            f"G {psnr_g:.2f}/{ssim_g:.3f} | "
            f"U {psnr_u:.2f}/{ssim_u:.3f} | "
            f"Time {time.time()-t0:.1f}s"
        )

    torch.save(net.state_dict(), os.path.join(args.save_dir, "final.pth"))
    csv_logger.close()
    logger.info("Training finished.")


# =========================
# Cmd
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("BUDA-Net Training")

    parser.add_argument("--stage1_epochs", type=int, default=30)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--repair_weight", type=float, default=0.5)
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "deblur_only", "no_m", "fixed_fusion"])
    parser.add_argument("--save_dir", type=str, default="checkpoints/full")
    parser.add_argument("--log_dir", type=str, default="logs/full")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)
