"""Build the CLASSIFIER dataset from raw photos: detect the condenser, crop it, split train/val/test.

Box source priority per image:
  1. your manual box in dataset_raw/boxes.json   (most reliable)
  2. the detector (learned model if available, else heuristic)
Images where no condenser is found are skipped (listed at the end) unless --fallback-full is given.
"""
import argparse
import json
import random
from pathlib import Path

import cv2

from detector import load_detector, crop_box

random.seed(42)

RAW = Path("dataset_raw")
OUT = Path("data")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="auto", choices=["auto", "heuristic", "learned"])
    ap.add_argument("--fallback-full", action="store_true", help="use the full image when no condenser is detected")
    args = ap.parse_args()

    classes = ["good", "dent"]
    boxes_file = RAW / "boxes.json"
    manual = json.loads(boxes_file.read_text()) if boxes_file.exists() else {}
    detector = load_detector(args.detector)
    print(f"Detector for unlabeled images: {detector.name} | manual boxes: {sum(1 for v in manual.values() if v)}")

    datasets, skipped = {}, []
    for label in classes:
        folder = RAW / label
        if not folder.exists():
            raise SystemExit(f"Folder tidak ditemukan: {folder}")
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            raise SystemExit(f"Tidak ada gambar di {folder}. Tambahkan foto GOOD dan DENT terlebih dahulu.")
        crops = []
        for p in images:
            frame = cv2.imread(str(p))
            if frame is None:
                skipped.append(f"{p} (unreadable)")
                continue
            rel = p.relative_to(RAW).as_posix()
            box, src = manual.get(rel), "manual"
            if not box:
                det = detector.detect(frame)
                box, src = (list(det.box), detector.name) if det else (None, None)
            if box:
                crops.append((p, crop_box(frame, tuple(box))))
            elif args.fallback_full:
                crops.append((p, frame))
            else:
                skipped.append(f"{p} (no condenser found)")
        random.shuffle(crops)
        datasets[label] = crops

    for split in ["train", "val", "test"]:
        for label in classes:
            (OUT / split / label).mkdir(parents=True, exist_ok=True)

    for label, crops in datasets.items():
        n = len(crops)
        if n < 3:
            raise SystemExit(f"Kelas {label} hanya memiliki {n} crop valid. Minimal 3 diperlukan untuk split awal.")
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        splits = {"train": crops[:n_train], "val": crops[n_train:n_train + n_val], "test": crops[n_train + n_val:]}
        for split, selected in splits.items():
            target = OUT / split / label
            for old in target.iterdir():                      # remove stale files from a previous run
                if old.is_file() and old.suffix.lower() in EXTENSIONS:
                    old.unlink()
            for p, crop in selected:
                cv2.imwrite(str(target / f"{p.stem}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"{label.upper()}: {n} crops | train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    if skipped:
        print("\nSkipped:")
        for s in skipped:
            print("  -", s)
    print("\nDataset preparation selesai (gambar sudah di-crop ke area kondensor).")
    print("Cek beberapa file di data/train/* secara visual sebelum training.")


if __name__ == "__main__":
    main()
