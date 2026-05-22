import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist(seed: int) -> datasets.MNIST:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()

    return Subset(dataset, indices)


def get_loader(dataset, start: int, end: int, batch_size: int) -> DataLoader:
    subset = Subset(dataset, list(range(start, end)))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)
