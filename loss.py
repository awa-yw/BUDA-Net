import torch.nn.functional as F

def l1_loss(pred, gt):
    return F.l1_loss(pred, gt)

def edge_loss(pred, gt):
    def sobel(x):
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx.abs().mean() + gy.abs().mean()
    return sobel(pred) - sobel(gt)
