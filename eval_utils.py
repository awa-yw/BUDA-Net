# eval_utils.py
import torch
import numpy as np

@torch.no_grad()
def psnr(pred, gt, max_val=1.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

@torch.no_grad()
def ssim(pred, gt, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean()
    mu_y = gt.mean()
    sigma_x = pred.var()
    sigma_y = gt.var()
    sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return num / den

@torch.no_grad()
def evaluate(model, dataloader, device, ablation="full"):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for blur, sharp in dataloader:
            blur, sharp = blur.to(device), sharp.to(device)
            out = model(blur, ablation=ablation)  # 传给 model.forward()
            pred = out["I_out"]

            psnr_val = psnr(pred, sharp)
            ssim_val = ssim(pred, sharp)

            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1

    return total_psnr / count, total_ssim / count
