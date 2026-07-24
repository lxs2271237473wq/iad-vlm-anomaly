from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path


PROJECT = Path("/root/private_data/iad-vlm-anomaly").resolve()
SOURCE_ROOT = PROJECT / "datasets/MVTec_AD_2_anomalib_all"

VIEW_ROOT = Path("/root/private_data/anomalyclip_data/ad2four").resolve()
ANOMALYCLIP_ROOT = Path("/root/private_data/third_party/AnomalyCLIP").resolve()

DATASET_PY = ANOMALYCLIP_ROOT / "dataset.py"
DATASET_BACKUP = ANOMALYCLIP_ROOT / "dataset.py.stage20_original_backup"

OUT_REPORT = (
    PROJECT
    / "docs/stage20_anomalyclip_baseline"
    / "stage20_c2_ad2four_meta_report.md"
)

CATEGORIES = {
    "fruit_jelly": SOURCE_ROOT / "fruit_jelly_folder",
    "sheet_metal": SOURCE_ROOT / "sheet_metal_folder",
    "vial": SOURCE_ROOT / "vial_folder",
    "walnuts": SOURCE_ROOT / "walnuts_folder",
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalized_stem(path: Path) -> str:
    stem = path.stem.lower()

    for suffix in (
        "_mask",
        "-mask",
        "_gt",
        "-gt",
        "_ground_truth",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def build_mask_index(mask_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}

    for path in image_files(mask_dir):
        index.setdefault(normalized_stem(path), []).append(path)

    return index


def find_mask(
    anomaly_image: Path,
    mask_dir: Path,
    mask_index: dict[str, list[Path]],
) -> Path:
    key = normalized_stem(anomaly_image)
    matches = mask_index.get(key, [])

    if len(matches) == 1:
        return matches[0]

    direct_candidates = [
        mask_dir / anomaly_image.name,
        mask_dir / f"{anomaly_image.stem}_mask{anomaly_image.suffix}",
        mask_dir / f"{anomaly_image.stem}.png",
        mask_dir / f"{anomaly_image.stem}_mask.png",
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        f"Could not uniquely match mask for {anomaly_image}. "
        f"normalized key={key!r}, matches={matches}"
    )


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        current = link.resolve()
        if current == target.resolve():
            return
        link.unlink()

    elif link.exists():
        raise RuntimeError(
            f"Refusing to replace non-symlink path: {link}"
        )

    link.symlink_to(target.resolve(), target_is_directory=True)


def relative_to_view(path: Path) -> str:
    return path.relative_to(VIEW_ROOT).as_posix()


def build_meta() -> tuple[dict, list[dict]]:
    VIEW_ROOT.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict[str, list[dict]]] = {
        "train": {},
        "test": {},
    }

    summary = []

    for category, source in CATEGORIES.items():
        if not source.exists():
            raise FileNotFoundError(source)

        category_view = VIEW_ROOT / category
        ensure_symlink(category_view, source)

        train_good = image_files(category_view / "train/good")
        test_good = image_files(category_view / "test/good")
        test_bad = image_files(category_view / "test/bad")

        mask_dir = category_view / "ground_truth/bad"
        masks = image_files(mask_dir)
        mask_index = build_mask_index(mask_dir)

        train_rows = []
        test_rows = []

        for image_path in train_good:
            train_rows.append(
                {
                    "img_path": relative_to_view(image_path),
                    "mask_path": "",
                    "cls_name": category,
                    "specie_name": "good",
                    "anomaly": 0,
                }
            )

        for image_path in test_good:
            test_rows.append(
                {
                    "img_path": relative_to_view(image_path),
                    "mask_path": "",
                    "cls_name": category,
                    "specie_name": "good",
                    "anomaly": 0,
                }
            )

        matched_masks = set()

        for image_path in test_bad:
            mask_path = find_mask(
                anomaly_image=image_path,
                mask_dir=mask_dir,
                mask_index=mask_index,
            )

            matched_masks.add(mask_path.resolve())

            test_rows.append(
                {
                    "img_path": relative_to_view(image_path),
                    "mask_path": relative_to_view(mask_path),
                    "cls_name": category,
                    "specie_name": "bad",
                    "anomaly": 1,
                }
            )

        if len(test_bad) != len(masks):
            raise RuntimeError(
                f"{category}: test/bad={len(test_bad)} "
                f"but masks={len(masks)}"
            )

        if len(matched_masks) != len(test_bad):
            raise RuntimeError(
                f"{category}: masks are not one-to-one. "
                f"bad={len(test_bad)}, unique masks={len(matched_masks)}"
            )

        meta["train"][category] = train_rows
        meta["test"][category] = test_rows

        summary.append(
            {
                "category": category,
                "train_good": len(train_good),
                "test_good": len(test_good),
                "test_bad": len(test_bad),
                "masks": len(masks),
                "test_total": len(test_rows),
            }
        )

    meta_path = VIEW_ROOT / "meta.json"
    meta_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return meta, summary


def patch_dataset_py() -> str:
    if not DATASET_PY.exists():
        raise FileNotFoundError(DATASET_PY)

    text = DATASET_PY.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if "dataset_name == 'ad2four'" in text or 'dataset_name == "ad2four"' in text:
        return "already_patched"

    if not DATASET_BACKUP.exists():
        shutil.copy2(DATASET_PY, DATASET_BACKUP)

    insertion = (
        "    elif dataset_name == 'ad2four':\n"
        "        obj_list = [\n"
        "            'fruit_jelly',\n"
        "            'sheet_metal',\n"
        "            'vial',\n"
        "            'walnuts',\n"
        "        ]\n"
    )

    # Insert immediately before the loop that enumerates obj_list.
    pattern = re.compile(
        r"(?P<indent>^[ \t]*)for\s+class_id\s*,\s*class_name\s+in\s+"
        r"enumerate\s*\(\s*obj_list\s*\)\s*:",
        flags=re.MULTILINE,
    )

    match = pattern.search(text)

    if not match:
        raise RuntimeError(
            "Could not find the obj_list enumeration in dataset.py. "
            "The official source layout may have changed."
        )

    indent = match.group("indent")

    if indent != "    ":
        raise RuntimeError(
            f"Unexpected indentation before obj_list loop: {indent!r}"
        )

    patched = text[: match.start()] + insertion + text[match.start():]

    DATASET_PY.write_text(
        patched,
        encoding="utf-8",
        newline="\n",
    )

    return "patched"


def validate_meta(meta: dict) -> dict:
    expected_test_total = 20 + 60 + 24 + 90 + 35 + 105 + 60 + 90
    expected_anomaly_total = 60 + 90 + 105 + 90

    test_total = 0
    anomaly_total = 0
    normal_total = 0

    for category in CATEGORIES:
        rows = meta["test"][category]

        for row in rows:
            image = VIEW_ROOT / row["img_path"]

            if not image.exists():
                raise FileNotFoundError(image)

            if row["anomaly"] == 1:
                mask = VIEW_ROOT / row["mask_path"]
                if not mask.exists():
                    raise FileNotFoundError(mask)
                anomaly_total += 1
            else:
                if row["mask_path"] != "":
                    raise RuntimeError(
                        f"Normal sample has mask: {row}"
                    )
                normal_total += 1

            test_total += 1

    if test_total != expected_test_total:
        raise RuntimeError(
            f"Expected {expected_test_total} test images, found {test_total}"
        )

    if anomaly_total != expected_anomaly_total:
        raise RuntimeError(
            f"Expected {expected_anomaly_total} anomalies, "
            f"found {anomaly_total}"
        )

    return {
        "test_total": test_total,
        "normal_total": normal_total,
        "anomaly_total": anomaly_total,
    }


def write_report(
    summary: list[dict],
    totals: dict,
    patch_status: str,
) -> None:
    lines = [
        "# Stage 20-C2: AD2-four AnomalyCLIP Metadata",
        "",
        "## Data view",
        "",
        f"- source: `{SOURCE_ROOT}`",
        f"- non-destructive view: `{VIEW_ROOT}`",
        f"- metadata: `{VIEW_ROOT / 'meta.json'}`",
        "- view method: symbolic links",
        "",
        "## Dataset mapping",
        "",
        "- dataset name: `ad2four`",
        "- categories: `fruit_jelly`, `sheet_metal`, `vial`, `walnuts`",
        f"- dataset.py patch status: `{patch_status}`",
        f"- original backup: `{DATASET_BACKUP}`",
        "",
        "## Counts",
        "",
        "| category | train good | test good | test bad | masks | test total |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in summary:
        lines.append(
            f"| {row['category']} | "
            f"{row['train_good']} | "
            f"{row['test_good']} | "
            f"{row['test_bad']} | "
            f"{row['masks']} | "
            f"{row['test_total']} |"
        )

    lines += [
        "",
        "## Validation",
        "",
        f"- test images: `{totals['test_total']}`",
        f"- normal test images: `{totals['normal_total']}`",
        f"- anomalous test images: `{totals['anomaly_total']}`",
        "- anomaly image-mask matching: one-to-one",
        "",
        "## Next step",
        "",
        "Load the official Dataset class and run a one-image GPU smoke test.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    meta, summary = build_meta()
    totals = validate_meta(meta)
    patch_status = patch_dataset_py()
    write_report(summary, totals, patch_status)

    print("[DONE]", VIEW_ROOT / "meta.json")
    print("[DONE]", OUT_REPORT)
    print("[DATASET_PATCH]", patch_status)
    print()
    print("===== totals =====")
    print(totals)
    print()
    print("===== per category =====")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
