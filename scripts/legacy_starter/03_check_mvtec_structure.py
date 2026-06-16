"""Check MVTec AD folder structure before running baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./datasets/MVTecAD")
    args = parser.parse_args()
    root = Path(args.root)

    print(f"Checking MVTec AD root: {root.resolve()}")
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    ok = True
    for category in EXPECTED:
        croot = root / category
        train_good = croot / "train" / "good"
        test_root = croot / "test"
        gt_root = croot / "ground_truth"
        status = croot.exists() and train_good.exists() and test_root.exists()
        print(f"{category:12s} exists={croot.exists()} train/good={train_good.exists()} test={test_root.exists()} ground_truth={gt_root.exists()}")
        ok = ok and status

    if not ok:
        raise SystemExit("MVTec AD structure is incomplete. Download/extract the official dataset first.")
    print("MVTec AD structure looks usable.")


if __name__ == "__main__":
    main()
