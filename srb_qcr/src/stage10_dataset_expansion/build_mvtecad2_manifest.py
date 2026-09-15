from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ROOT = Path(".").resolve()
DATASET_ROOT = ROOT / "datasets" / "MVTec_AD_2"

OUT_DIR = ROOT / "results" / "stage10_dataset_expansion"
DOC_DIR = ROOT / "docs" / "stage10_dataset_expansion"

OUT_MANIFEST = OUT_DIR / "stage10_b1_mvtecad2_manifest.csv"
OUT_SUMMARY = OUT_DIR / "stage10_b1_mvtecad2_manifest_summary.csv"
OUT_REPORT = DOC_DIR / "stage10_b1_mvtecad2_manifest_report.md"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

VALID_SPLITS = {
    "train",
    "validation",
    "test_public",
    "test_private",
    "test_private_mixed",
}

MASK_KEYWORDS = {
    "mask",
    "masks",
    "ground_truth",
    "gt",
    "annotation",
    "annotations",
    "label",
    "labels",
}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def has_mask_keyword(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    stem = path.stem.lower()

    if any(part in MASK_KEYWORDS for part in parts):
        return True

    if "mask" in name or "ground_truth" in name or stem.endswith("_gt"):
        return True

    return False


def find_split(path: Path) -> Optional[str]:
    for part in path.parts:
        if part in VALID_SPLITS:
            return part
    return None


def find_category(path: Path) -> Optional[str]:
    try:
        rel = path.relative_to(DATASET_ROOT)
    except ValueError:
        return None

    if len(rel.parts) < 2:
        return None

    return rel.parts[0]


def split_relative_parts(path: Path, split: str) -> List[str]:
    rel = path.relative_to(DATASET_ROOT)
    parts = list(rel.parts)

    if split not in parts:
        return []

    idx = parts.index(split)
    return parts[idx + 1:]


def infer_label(split: str, rel_after_split: List[str]) -> Dict[str, object]:
    lower_parts = [p.lower() for p in rel_after_split]

    if split in {"train", "validation"}:
        return {
            "label_available": True,
            "is_anomaly": 0,
            "anomaly_type": "good",
            "evaluation_scope": "normal_only",
        }

    if split == "test_public":
        if any(p in {"good", "normal", "ok"} for p in lower_parts):
            return {
                "label_available": True,
                "is_anomaly": 0,
                "anomaly_type": "good",
                "evaluation_scope": "local_labeled_test",
            }

        # If the image is directly under test_public without a defect folder,
        # keep the type as unknown_public_anomaly. Otherwise use the first folder.
        anomaly_type = "unknown_public_anomaly"
        if len(rel_after_split) >= 2:
            anomaly_type = rel_after_split[0]
        elif len(rel_after_split) == 1:
            anomaly_type = "unknown_public_anomaly"

        return {
            "label_available": True,
            "is_anomaly": 1,
            "anomaly_type": anomaly_type,
            "evaluation_scope": "local_labeled_test",
        }

    if split in {"test_private", "test_private_mixed"}:
        return {
            "label_available": False,
            "is_anomaly": -1,
            "anomaly_type": "unknown_private",
            "evaluation_scope": "official_server_only",
        }

    return {
        "label_available": False,
        "is_anomaly": -1,
        "anomaly_type": "unknown",
        "evaluation_scope": "unknown",
    }


def collect_mask_candidates() -> Dict[str, List[Path]]:
    mask_candidates: Dict[str, List[Path]] = {}

    for path in DATASET_ROOT.rglob("*"):
        if not is_image_file(path):
            continue
        if not has_mask_keyword(path):
            continue

        category = find_category(path)
        split = find_split(path)

        if category is None or split is None:
            continue

        key = f"{category}::{split}"
        mask_candidates.setdefault(key, []).append(path)

    return mask_candidates


def find_mask_for_image(image_path: Path, mask_candidates: Dict[str, List[Path]]) -> str:
    category = find_category(image_path)
    split = find_split(image_path)

    if category is None or split is None:
        return ""

    key = f"{category}::{split}"
    candidates = mask_candidates.get(key, [])

    if not candidates:
        return ""

    stem = image_path.stem.lower()

    exact = []
    partial = []

    for m in candidates:
        mstem = m.stem.lower()
        if mstem == stem:
            exact.append(m)
        elif stem in mstem or mstem in stem:
            partial.append(m)

    chosen = exact[0] if exact else (partial[0] if partial else None)

    if chosen is None:
        return ""

    return str(chosen.relative_to(ROOT))


def build_manifest() -> pd.DataFrame:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing dataset root: {DATASET_ROOT}")

    mask_candidates = collect_mask_candidates()
    rows = []

    for path in sorted(DATASET_ROOT.rglob("*")):
        if not is_image_file(path):
            continue

        if has_mask_keyword(path):
            continue

        category = find_category(path)
        split = find_split(path)

        if category is None or split is None:
            continue

        rel_after_split = split_relative_parts(path, split)
        label_info = infer_label(split, rel_after_split)

        mask_path = ""
        if split == "test_public" and int(label_info["is_anomaly"]) == 1:
            mask_path = find_mask_for_image(path, mask_candidates)

        rows.append({
            "dataset": "MVTec AD 2",
            "category": category,
            "split": split,
            "image_path": str(path.relative_to(ROOT)),
            "mask_path": mask_path,
            "label_available": bool(label_info["label_available"]),
            "is_anomaly": int(label_info["is_anomaly"]),
            "anomaly_type": str(label_info["anomaly_type"]),
            "evaluation_scope": str(label_info["evaluation_scope"]),
            "filename": path.name,
            "relative_after_split": "/".join(rel_after_split),
            "has_mask": bool(mask_path),
        })

    if not rows:
        raise RuntimeError(f"No image rows found under {DATASET_ROOT}")

    return pd.DataFrame(rows)


def build_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    summary = (
        manifest
        .groupby(
            [
                "dataset",
                "category",
                "split",
                "label_available",
                "is_anomaly",
                "anomaly_type",
                "evaluation_scope",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            num_images=("image_path", "count"),
            num_masks=("has_mask", "sum"),
        )
        .sort_values(["category", "split", "is_anomaly", "anomaly_type"])
    )

    return summary


def write_report(manifest: pd.DataFrame, summary: pd.DataFrame) -> None:
    total = len(manifest)
    labeled = int(manifest["label_available"].sum())
    unlabeled = total - labeled
    public_rows = manifest[manifest["split"] == "test_public"]
    public_anomaly = public_rows[public_rows["is_anomaly"] == 1]
    public_masked = int(public_anomaly["has_mask"].sum()) if len(public_anomaly) else 0

    lines = []
    lines.append("# Stage 10-B1 MVTec AD 2 Manifest Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage builds a unified manifest for MVTec AD 2.")
    lines.append("It does not train models, run anomaly detectors, run VLM reasoning, or modify previous results.")
    lines.append("")
    lines.append("## 2. Dataset Root")
    lines.append("")
    lines.append(f"`{DATASET_ROOT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 3. Output Files")
    lines.append("")
    lines.append(f"- `{OUT_MANIFEST.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 4. Overall Statistics")
    lines.append("")
    lines.append(f"- Total manifest images: `{total}`")
    lines.append(f"- Labeled images: `{labeled}`")
    lines.append(f"- Unlabeled private images: `{unlabeled}`")
    lines.append(f"- Public anomalous images with masks detected: `{public_masked}` / `{len(public_anomaly)}`")
    lines.append("")
    lines.append("## 5. Split Summary")
    lines.append("")
    lines.append("| Category | Split | Label available | is_anomaly | Anomaly type | Images | Masks | Evaluation scope |")
    lines.append("|---|---|---:|---:|---|---:|---:|---|")

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['category']} | {r['split']} | {r['label_available']} | {int(r['is_anomaly'])} | "
            f"{r['anomaly_type']} | {int(r['num_images'])} | {int(r['num_masks'])} | {r['evaluation_scope']} |"
        )

    lines.append("")
    lines.append("## 6. Important Notes")
    lines.append("")
    lines.append("- `train` and `validation` are treated as normal-only splits.")
    lines.append("- `test_public` is treated as the local labeled test split.")
    lines.append("- `test_private` and `test_private_mixed` are marked as label-unavailable and should not be used for local AUROC/AP/F1.")
    lines.append("- If mask coverage is zero, Stage 10 should first run image-level detector/crop/VLM reasoning; pixel-level evaluation should wait until mask pairing is confirmed.")
    lines.append("")
    lines.append("## 7. Next Step")
    lines.append("")
    lines.append("Stage 10-B2 should adapt this manifest into the existing PatchCore/FastFlow and crop-reasoning pipeline.")
    lines.append("For the first pilot, use only:")
    lines.append("")
    lines.append("```text")
    lines.append("category = vial")
    lines.append("train + validation for normal reference")
    lines.append("test_public for local evaluation")
    lines.append("test_private/test_private_mixed excluded from local metric computation")
    lines.append("```")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    summary = build_summary(manifest)

    manifest.to_csv(OUT_MANIFEST, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(manifest, summary)

    print("[DONE]", OUT_MANIFEST)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print("manifest_rows:", len(manifest))
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
