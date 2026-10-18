"""Fine-tune the condenser detector (Faster R-CNN MobileNetV3) on your labeled boxes.

Positives : images with a box in dataset_raw/boxes.json
Negatives : entries marked null in boxes.json (key x in label_boxes.py) and every image in dataset_raw/background/
Run:  python train_detector.py --epochs 30
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from detector import build_detector_model, HeuristicDetector

RAW = Path("dataset_raw")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def collect_items(use_auto):
    boxes_file = RAW / "boxes.json"
    boxes = json.loads(boxes_file.read_text()) if boxes_file.exists() else {}
    items = []
    for rel, box in boxes.items():
        if (RAW / rel).exists():
            items.append((RAW / rel, box))
    bg = RAW / "background"
    if bg.exists():
        items += [(p, None) for p in sorted(bg.iterdir()) if p.suffix.lower() in EXT]
    if use_auto:   # include unlabeled images using heuristic proposals (less accurate than your own boxes)
        heur = HeuristicDetector()
        for label in ("good", "dent"):
            for p in sorted((RAW / label).glob("*")):
                rel = p.relative_to(RAW).as_posix()
                if p.suffix.lower() in EXT and rel not in boxes:
                    d = heur.detect(cv2.imread(str(p)))
                    if d:
                        items.append((p, list(d.box)))
    return items


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, box = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = min(1.0, 960.0 / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        b = None if box is None else np.array(box, dtype=np.float32) * s
        if self.train:
            if random.random() < 0.5:                                  # horizontal flip
                img = np.ascontiguousarray(img[:, ::-1])
                if b is not None:
                    b = np.array([w - b[2], b[1], w - b[0], b[3]], dtype=np.float32)
            if random.random() < 0.6:                                  # zoom-out: condenser smaller in frame
                z = random.uniform(1.0, 1.8)
                W, H = int(w * z), int(h * z)
                canvas = np.full((H, W, 3), img.reshape(-1, 3).mean(0), dtype=np.uint8)
                ox, oy = random.randint(0, W - w), random.randint(0, H - h)
                canvas[oy:oy + h, ox:ox + w] = img
                img = canvas
                if b is not None:
                    b = b + np.array([ox, oy, ox, oy], dtype=np.float32)
            alpha, beta = random.uniform(0.8, 1.2), random.uniform(-20, 20)  # brightness / contrast
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        if b is None:
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}
        else:
            target = {"boxes": torch.tensor(b, dtype=torch.float32).unsqueeze(0),
                      "labels": torch.tensor([1], dtype=torch.int64)}
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def evaluate(model, items, device, score_thr=0.6):
    """Mean IoU on positives, hit-rate (IoU>=0.5), and false alarms on negatives."""
    model.eval()
    ious, fp, n_neg = [], 0, 0
    ds = BoxDataset(items, train=False)
    with torch.no_grad():
        for k in range(len(ds)):
            img, tgt = ds[k]
            out = model([img.to(device)])[0]
            keep = out["scores"] >= score_thr
            pred = out["boxes"][keep][:1].cpu().numpy()
            if len(tgt["boxes"]) == 0:
                n_neg += 1
                fp += int(len(pred) > 0)
            else:
                ious.append(iou(pred[0], tgt["boxes"][0].numpy()) if len(pred) else 0.0)
    return (float(np.mean(ious)) if ious else 0.0,
            float(np.mean([v >= 0.5 for v in ious])) if ious else 0.0, fp, n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--use-auto", action="store_true", help="also train on heuristic boxes for unlabeled images")
    ap.add_argument("--out", default="models/condenser_detector.pth")
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    items = collect_items(args.use_auto)
    n_pos = sum(1 for _, b in items if b)
    if n_pos < 8:
        raise SystemExit(f"Only {n_pos} labeled boxes found. Run `python label_boxes.py` first (aim for 30+, more is better).")
    random.shuffle(items)
    n_val = max(3, int(0.2 * len(items))) if len(items) >= 15 else 0
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"Train: {len(train_items)} images | Val: {len(val_items)} images | positives total: {n_pos}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector_model(pretrained=True).to(device)
    loader = torch.utils.data.DataLoader(BoxDataset(train_items, True), batch_size=args.batch_size,
                                         shuffle=True, collate_fn=collate, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(args.epochs * 0.6)), gamma=0.1)

    best = float("-inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        msg = f"Epoch {epoch + 1:02d}/{args.epochs} | loss={total / max(len(loader), 1):.4f}"
        score = -total
        if val_items:
            miou, hit, fp, n_neg = evaluate(model, val_items, device)
            msg += f" | val mIoU={miou:.3f} hit@0.5={hit:.2f} false_alarms={fp}/{n_neg}"
            score = miou - 0.1 * fp
        print(msg)
        if score > best or epoch == 0:
            best = score
            torch.save({"state_dict": model.state_dict(), "min_size": 480, "max_size": 800,
                        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn"}, args.out)
    print(f"Saved detector: {args.out}")
    print("Check it visually on live video before relying on it; with few images, add more varied boxes + background shots.")


if __name__ == "__main__":
    main()
