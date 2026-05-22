import argparse
import csv
import json
import os
import torch
import torch.nn as nn

from utils import set_seed
from data import get_mnist, get_loader
from model import MiniCNN
from train import train_epoch, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continual Learning on MNIST")
    parser.add_argument("--n", type=int, default=100, help="Number of continual iterations")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


CHECKPOINT_PATH = "logs/checkpoint.pt"
JSONL_PATH = "logs/results.jsonl"
CSV_PATH = "logs/results.csv"
LOG_FIELDS = ["iteration", "train_size", "true_label", "predicted_label", "confidence", "loss"]


def write_entry(entry: dict, csv_writer) -> None:
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    csv_writer.writerow(entry)


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu to override.")

    device = torch.device(args.device)
    set_seed(args.seed)

    dataset = get_mnist(seed=args.seed)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("logs", exist_ok=True)
    open(JSONL_PATH, "w").close()  # truncate at start of run
    csv_file = open(CSV_PATH, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
    csv_writer.writeheader()

    try:
        # --- Bootstrap: train on first 100 samples, save checkpoint ---
        model = MiniCNN().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        loader = get_loader(dataset, start=0, end=100, batch_size=args.batch_size)
        train_epoch(model, loader, optimizer, criterion, device)
        torch.save(model.state_dict(), CHECKPOINT_PATH)

        # --- Iterative re-instantiation loop ---
        for i in range(100, 100 + args.n):
            # Re-instantiate model from last checkpoint
            model = MiniCNN().to(device)
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

            # Predict image i BEFORE training on it
            img_i, true_label_i = dataset[i]
            pred_label, conf, loss = predict(
                model, img_i, torch.tensor(true_label_i), criterion, device
            )

            # Train on single new image i only
            loader = get_loader(dataset, start=i, end=i + 1, batch_size=args.batch_size)
            train_epoch(model, loader, optimizer, criterion, device)

            # Save updated checkpoint
            torch.save(model.state_dict(), CHECKPOINT_PATH)

            entry = {
                "iteration": i,
                "train_size": i + 1,
                "true_label": int(true_label_i),
                "predicted_label": int(pred_label),
                "confidence": round(conf, 6),
                "loss": round(loss, 6),
            }
            write_entry(entry, csv_writer)
            print(
                f"iter {i:4d} | train_size={i+1:4d} | true={true_label_i} "
                f"pred={pred_label} conf={conf:.4f} loss={loss:.4f}"
            )

        print(f"\nDone. {args.n} entries written to {JSONL_PATH} and {CSV_PATH}.")
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
