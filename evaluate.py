import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from submission.model import get_model, get_transform


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=Path("submission"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model = get_model()
    try:
        state = torch.load(args.submission / "model.pth", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.submission / "model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dataset = ImageFolder(args.data, transform=get_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    targets, predictions = [], []
    for images, labels in loader:
        predictions.extend(model(images).argmax(dim=1).tolist())
        targets.extend(labels.tolist())

    labels = list(range(len(dataset.classes)))
    print(f"classes: {dataset.classes}")
    print(f"accuracy: {accuracy_score(targets, predictions):.4f}")
    print(f"macro_f1: {f1_score(targets, predictions, labels=labels, average='macro', zero_division=0):.4f}")
    print("f1_per_class:", f1_score(targets, predictions, labels=labels, average=None, zero_division=0))
    print("confusion_matrix:\n", confusion_matrix(targets, predictions, labels=labels))


if __name__ == "__main__":
    main()
