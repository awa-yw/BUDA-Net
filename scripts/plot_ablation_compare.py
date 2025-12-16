import csv
import argparse
import matplotlib.pyplot as plt

def read_total_loss(csv_path):
    losses = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            losses.append(float(row["loss_total"]))
    return losses


def main(args):
    plt.figure(figsize=(8, 5))

    for item in args.inputs:
        name, path = item.split(":")
        loss = read_total_loss(path)
        plt.plot(loss, label=name, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot ablation comparison")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Format: name:path_to_csv"
    )
    parser.add_argument("--out", type=str, default="ablation_compare.png")
    args = parser.parse_args()
    main(args)
