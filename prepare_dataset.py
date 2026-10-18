
from pathlib import Path
import random
import shutil

random.seed(42)

RAW = Path("dataset_raw")
OUT = Path("data")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    classes = ["good", "dent"]
    datasets = {}

    # Read images from the raw dataset folders
    for label in classes:
        folder = RAW / label

        if not folder.exists():
            raise SystemExit(f"Folder tidak ditemukan: {folder}")

        images = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        if not images:
            raise SystemExit(
                f"Tidak ada gambar di {folder}. "
                "Tambahkan foto GOOD dan DENT terlebih dahulu."
            )

        random.shuffle(images)
        datasets[label] = images

    # Prepare output folders
    for split in ["train", "val", "test"]:
        for label in classes:
            (OUT / split / label).mkdir(
                parents=True, exist_ok=True
            )

    # Split each class: 70% train, 15% validation, remainder test
    for label, images in datasets.items():
        n = len(images)

        if n < 3:
            raise SystemExit(
                f"Kelas {label} hanya memiliki {n} foto. "
                "Minimal 3 foto diperlukan untuk membuat split awal."
            )

        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, selected in splits.items():
            target = OUT / split / label

            # Remove old generated images to prevent stale data
            for old_file in target.iterdir():
                if old_file.is_file() and old_file.suffix.lower() in EXTENSIONS:
                    old_file.unlink()

            for image in selected:
                shutil.copy2(image, target / image.name)

        print(
            f"{label.upper()}: {n} images | "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )

    print("\nDataset preparation selesai.")
    print("Periksa hasil split sebelum training.")


if __name__ == "__main__":
    main()
