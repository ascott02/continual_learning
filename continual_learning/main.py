import argparse
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


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu to override.")

    device = torch.device(args.device)
    set_seed(args.seed)

    dataset = get_mnist(seed=args.seed)

    # Milestone 3: train on first 100 samples, predict sample 101
    model = MiniCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    loader = get_loader(dataset, start=0, end=100, batch_size=args.batch_size)
    train_epoch(model, loader, optimizer, criterion, device)

    img_101, true_label_101 = dataset[100]
    pred, conf, loss = predict(model, img_101, torch.tensor(true_label_101), criterion, device)

    print(f"Sample 101 — true: {true_label_101}, predicted: {pred}, confidence: {conf:.4f}, loss: {loss:.4f}")


if __name__ == "__main__":
    main()
