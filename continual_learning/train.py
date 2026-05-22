import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> None:
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


def predict(model: nn.Module, image: torch.Tensor, label: torch.Tensor,
            criterion: nn.Module, device: torch.device) -> tuple[int, float, float]:
    model.eval()
    with torch.no_grad():
        image, label = image.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
        logits = model(image)
        loss = criterion(logits, label).item()
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)
    return predicted.item(), confidence.item(), loss
