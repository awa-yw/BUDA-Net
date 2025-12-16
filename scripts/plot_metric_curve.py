import csv
import argparse
import matplotlib.pyplot as plt

def read_metrics(csv_path):
    psnr, ssim = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "psnr" in row:
                psnr.append(float(row["psnr"]))
                ssim.append(float(row["ssim"]))
    return psnr, ssim

def main(args):
    psnr, ssim = read_metrics(args.log)

    plt.figure(figsize=(8,4))
    plt.plot(psnr, label="PSNR (dB)", linewidth=2)
    plt.plot(ssim, label="SSIM", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("PSNR / SSIM over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default="psnr_ssim_curve.png")
    args = parser.parse_args()
    main(args)
