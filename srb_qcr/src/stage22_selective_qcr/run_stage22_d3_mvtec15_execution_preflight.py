from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

OUT = (
    ROOT
    / "docs/stage22_selective_qcr"
    / "stage22_d3_mvtec15_execution_preflight.txt"
)

MVTEC15 = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

CODE_PATHS = [
    "scripts/legacy_starter/run_mvtec_baselines.py",
    "experiments/analysis/full_test_patchcore_candidate_regions.py",
    "experiments/analysis/patchcore_candidate_regions.py",
    "experiments/stage7_generalization/run_visa_patchcore_baseline_and_candidates.py",
    "experiments/stage7_generalization/visa_binary_prompt_reasoning.py",
    "experiments/analysis/real_anomaly_crop_visual_prompt_reasoning.py",
    "experiments/stage10_dataset_expansion/run_stage10_d_patchcore_candidate_crops.py",
    "experiments/stage10_dataset_expansion/run_stage10_e_vlm_full_vs_crop.py",
]

PATTERN = re.compile(
    r"argparse|add_argument|CATEGORY|CATEGORIES|"
    r"data_root|dataset_root|output|save_path|"
    r"patchcore|candidate_regions|"
    r"patchcore_image_predictions|"
    r"create_model_and_transforms|open_clip|"
    r"def main|def run|def evaluate|"
    r"checkpoint|load_state_dict",
    re.IGNORECASE,
)

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        1
        for item in path.rglob("*")
        if (
            item.is_file()
            and item.suffix.lower() in IMAGE_SUFFIXES
        )
    )


def code_context(path: Path) -> list[str]:
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    selected = []
    shown = set()

    for index, line in enumerate(lines):
        if not PATTERN.search(line):
            continue

        start = max(0, index - 2)
        end = min(len(lines), index + 4)

        for i in range(start, end):
            if i in shown:
                continue

            selected.append(
                f"{i + 1:4d}: {lines[i]}"
            )
            shown.add(i)

        selected.append("----")

        if len(selected) >= 160:
            selected.append(
                "... output truncated ..."
            )
            break

    return selected


def package_status(name: str) -> str:
    return (
        "installed"
        if importlib.util.find_spec(name)
        else "missing"
    )


def checkpoint_inventory() -> list[str]:
    candidates = []

    search_roots = [
        ROOT / "results",
        ROOT / "runs",
        ROOT / "checkpoints",
    ]

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for path in search_root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".ckpt",
                ".pth",
                ".pt",
            }:
                continue

            text = str(path).lower()

            if (
                "patchcore" in text
                or "mvtec" in text
            ):
                candidates.append(
                    str(path.relative_to(ROOT))
                )

    return sorted(candidates)


def baseline_summary() -> list[str]:
    path = (
        ROOT
        / "results/baselines"
        / "patchcore_mvtec_summary.csv"
    )

    if not path.exists():
        return ["not found"]

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    output = [
        f"path: {path.relative_to(ROOT)}",
        f"rows: {len(rows)}",
        f"columns: {reader.fieldnames}",
        "categories: "
        + ", ".join(
            sorted(
                {
                    str(
                        row.get(
                            "category",
                            row.get("class", ""),
                        )
                    )
                    for row in rows
                }
            )
        ),
    ]

    return output


def main() -> None:
    lines = [
        "===== STAGE 22-D3 MVTec15 EXECUTION PREFLIGHT =====",
        "",
        "1. ENVIRONMENT",
        "",
    ]

    for package in [
        "torch",
        "torchvision",
        "anomalib",
        "open_clip",
        "sklearn",
        "cv2",
        "PIL",
        "pandas",
        "numpy",
    ]:
        lines.append(
            f"{package}: {package_status(package)}"
        )

    lines += [
        "",
        "2. DATASET INVENTORY",
        "",
        (
            "category | train/good | test/good | "
            "test/anomaly | ground_truth"
        ),
    ]

    dataset_root = ROOT / "datasets/MVTecAD"

    total_test = 0

    for category in MVTEC15:
        root = dataset_root / category

        train_good = count_images(
            root / "train/good"
        )

        test_good = count_images(
            root / "test/good"
        )

        all_test = count_images(
            root / "test"
        )

        test_anomaly = max(
            all_test - test_good,
            0,
        )

        ground_truth = count_images(
            root / "ground_truth"
        )

        total_test += all_test

        lines.append(
            f"{category} | "
            f"{train_good} | "
            f"{test_good} | "
            f"{test_anomaly} | "
            f"{ground_truth}"
        )

    lines += [
        "",
        f"total test images: {total_test}",
        "",
        "3. EXISTING PATCHCORE SUMMARY",
        "",
        *baseline_summary(),
        "",
        "4. CHECKPOINT INVENTORY",
        "",
    ]

    checkpoints = checkpoint_inventory()

    if checkpoints:
        lines.extend(checkpoints)
    else:
        lines.append(
            "no reusable MVTec/PatchCore checkpoints found"
        )

    lines += [
        "",
        "5. CODE PATHS",
        "",
    ]

    for relative in CODE_PATHS:
        path = ROOT / relative

        lines.append("=" * 90)
        lines.append(relative)
        lines.append("=" * 90)

        if not path.exists():
            lines.append("NOT FOUND")
            lines.append("")
            continue

        lines.append(
            f"size_bytes: {path.stat().st_size}"
        )

        lines.extend(code_context(path))
        lines.append("")

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print("[DONE]", OUT)
    print("total test images:", total_test)
    print("checkpoints:", len(checkpoints))
    print()
    print("查看精简结论：")
    print(
        "grep -E "
        "'total test images:|checkpoints:|"
        "installed|missing|NOT FOUND|"
        "no reusable' "
        + str(OUT.relative_to(ROOT))
    )


if __name__ == "__main__":
    main()
