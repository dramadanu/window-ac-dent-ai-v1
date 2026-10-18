import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    if set(train_ds.classes) != {"dent", "good"}:
        raise SystemExit("Expected data/train/dent and data/train/good folders.")
    if train_ds.classes != val_ds.classes:
        raise SystemExit(f"Class mismatch. train={train_ds.classes}, val={val_ds.classes}")
    if len(train_ds) < 4 or len(val_ds) < 2:
        raise SystemExit("Dataset too small. Add labeled images to data/train and data/val.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as exc:
        raise SystemExit(
            "Could not load pretrained MobileNetV3 weights. Check internet access or install "
            f"weights through an approved process. Original error: {exc}"
        )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(train_ds.classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)
        val_acc = v_correct / max(v_total, 1)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={train_loss:.4f} "
              f"| train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": train_ds.classes,
                "image_size": 224,
                "architecture": "mobilenet_v3_small",
            }, out_dir / "dent_model.pth")
            (out_dir / "class_names.json").write_text(
                json.dumps(train_ds.classes, indent=2), encoding="utf-8"
            )
    print(f"Best validation accuracy: {best_acc:.3f}")
    print(f"Saved model: {out_dir / 'dent_model.pth'}")
    print("Note: evaluate once on a separate held-out test set before production use.")

if __name__ == "__main__":
    main()
