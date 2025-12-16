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
def evaluate(model, loader, device):
    model.eval()
    psnr_list, ssim_list = [], []

    for blur, sharp in loader:
        blur = blur.to(device)
        sharp = sharp.to(device)

        out = model(blur)
        pred = out["I_out"].clamp(0, 1)

        for i in range(pred.size(0)):
            psnr_list.append(psnr(pred[i], sharp[i]).item())
            ssim_list.append(ssim(pred[i], sharp[i]).item())

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))
