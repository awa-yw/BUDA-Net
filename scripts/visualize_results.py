import os
import argparse
import torch
import torchvision.utils as vutils

from model import DualBranchDeblurRepairNet
from datasets.gopro_dataset import GoProDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def main(args):

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- model ----
    model = DualBranchDeblurRepairNet(base=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # ---- dataset ----
    dataset = GoProDataset(
        root=args.data_root,
        split="test",
        crop_size=args.crop_size,
        training=False
    )

    for idx in args.indices:
        blur, sharp = dataset[idx]
        blur = blur.unsqueeze(0).to(device)

        out = model(blur)
        I_d = out["I_d"].clamp(0,1)
        I_r = out["I_r"].clamp(0,1)
        I_out = out["I_out"].clamp(0,1)

        grid = torch.cat([
            blur.cpu(),
            I_d.cpu(),
            I_r.cpu(),
            I_out.cpu(),
            sharp.unsqueeze(0)
        ], dim=0)

        save_path = os.path.join(args.out_dir, f"sample_{idx}.png")
        vutils.save_image(
            grid, save_path,
            nrow=5,
            padding=10,
            normalize=False
        )

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BUDA-Net Visualization")

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="datasets/GoPro")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--out_dir", default="figures/visual")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 5, 10],
        help="Indices of test samples"
    )

    args = parser.parse_args()
    main(args)
