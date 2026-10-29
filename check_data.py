from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "val", "test"]:
    print(f"\n===== {split.upper()} =====")
    total = 0

    for cls in ["0", "1", "2", "3", "4"]:
        folder = Path(split) / cls

        count = sum(
            1 for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        )

        print(f"Class {cls}: {count}")
        total += count

    print(f"Total: {total}")