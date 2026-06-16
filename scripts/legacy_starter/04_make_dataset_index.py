"""Create a simple dataset index for MVTec AD categories.

This is useful before experiments: it records train/test counts and defect folders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_images(path: Path) -> int:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./datasets/MVTecAD")
    parser.add_argument("--out", default="./results/dataset_index_mvtec.json")
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for category_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        test_root = category_dir / "test"
        defect_types = sorted([p.name for p in test_root.iterdir() if p.is_dir()]) if test_root.exists() else []
        records.append({
            "category": category_dir.name,
            "train_good_images": count_images(category_dir / "train" / "good"),
            "test_images": count_images(test_root),
            "ground_truth_masks": count_images(category_dir / "ground_truth"),
            "test_defect_folders": defect_types,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved dataset index to {out}")


if __name__ == "__main__":
    main()
