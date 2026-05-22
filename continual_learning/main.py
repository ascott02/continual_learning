import argparse
import torch

from utils import set_seed
from data import get_mnist


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

    print("First 5 labels:")
    for i in range(5):
        img, label = dataset[i]
        print(f"  [{i}] label={label}")


if __name__ == "__main__":
    main()
