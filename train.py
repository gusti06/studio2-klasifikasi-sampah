import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from submission.model import IMAGE_SIZE, NUM_CLASSES, build_model


CLASS_NAMES = [
    "botol_plastik",
    "gelas_plastik",
    "kertas_kardus",
    "saset_kemasan",
    "styrofoam",
]


def split_indices(targets, validation_fraction, seed):
    rng = random.Random(seed)
    by_class = {label: [] for label in range(NUM_CLASSES)}
    for index, label in enumerate(targets):
        by_class[label].append(index)
    train_indices, validation_indices = [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        count = max(1, round(len(indices) * validation_fraction))
        validation_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    return train_indices, validation_indices


def evaluate(model, loader, device):
    model.eval()
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return f1_score(targets, predictions, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("submission/model.pth"))
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--model",
        choices=["densenet121", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny"],
        default="efficientnet_b0",
    )
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"Dataset tidak ditemukan: {args.data}. Isi folder train/ terlebih dahulu.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.72, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(12)], p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    base_dataset = ImageFolder(args.data)
    if base_dataset.classes != CLASS_NAMES:
        raise SystemExit(f"Urutan kelas harus {CLASS_NAMES}, ditemukan {base_dataset.classes}")
    train_indices, validation_indices = split_indices(base_dataset.targets, args.val_fraction, args.seed)
    train_dataset = Subset(ImageFolder(args.data, transform=train_transform), train_indices)
    validation_dataset = Subset(ImageFolder(args.data, transform=validation_transform), validation_indices)
    train_labels = [base_dataset.targets[index] for index in train_indices]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    sample_weights = torch.as_tensor([class_weights[label] for label in train_labels], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, pretrained=True)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    epochs_without_improvement = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        score = evaluate(model, validation_loader, device)
        print(f"epoch {epoch:02d}/{args.epochs} macro_f1={score:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        if score > best_f1:
            best_f1 = score
            epochs_without_improvement = 0
            torch.save({key: value.cpu() for key, value in model.state_dict().items()}, args.output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break
    print(f"best_macro_f1={best_f1:.4f}; saved={args.output}")


if __name__ == "__main__":
    main()
