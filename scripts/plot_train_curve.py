import csv
import argparse
import matplotlib.pyplot as plt

def read_log(csv_path):
    epochs = []
    loss_total = []
    loss_deblur = []
    loss_repair = []
    stage = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            loss_deblur.append(float(row["loss_deblur"]))
            loss_repair.append(float(row["loss_repair"]))
            loss_total.append(float(row["loss_total"]))
            stage.append(row["stage"])

    return epochs, loss_deblur, loss_repair, loss_total, stage


def main(args):
    epochs, ld, lr, lt, stage = read_log(args.log)

    plt.figure(figsize=(8, 5))

    plt.plot(lt, label="Total Loss", linewidth=2)
    plt.plot(ld, label="Deblur Loss", linestyle="--")

    # Repair loss only meaningful in stage2
    if any(s == "stage2" for s in stage):
        plt.plot(lr, label="Repair Loss", linestyle=":")

    # stage split
    if "stage2" in stage:
        split = stage.index("stage2")
        plt.axvline(split, color="gray", linestyle="--", label="Stage-2 Start")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot training curve")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to train_log.csv")
    parser.add_argument("--out", type=str, default="train_curve.png",
                        help="Output image file")
    args = parser.parse_args()
    main(args)
