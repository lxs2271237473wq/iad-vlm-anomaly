from __future__ import annotations

from pathlib import Path
from io import StringIO
import json
import pandas as pd


ROOT = Path(".").resolve()

AD2_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

SCAN_ROOTS = [
    ROOT / "results",
    ROOT / "runs",
    ROOT / "outputs",
    ROOT / "docs",
]

OUT_DIR = ROOT / "results/stage18_ad2_qcr_ablation"
DOC_DIR = ROOT / "docs/stage18_ad2_qcr_ablation"

OUT_INVENTORY = OUT_DIR / "stage18_b0_ad2_qcr_source_inventory.csv"
OUT_CANDIDATES = OUT_DIR / "stage18_b0_ad2_qcr_source_candidates.csv"
OUT_REPORT = DOC_DIR / "stage18_b0_ad2_qcr_source_inventory_report.md"


CATEGORY_COLS = [
    "category",
    "class",
    "object",
    "object_name",
    "class_name",
    "category_name",
]

IMAGE_ID_COLS = [
    "image_key",
    "image_path",
    "img_path",
    "path",
    "filename",
    "file_name",
    "sample_id",
]

LABEL_COLS = [
    "label",
    "y",
    "y_true",
    "gt",
    "gt_label",
    "is_anomaly",
    "is_anomaly_final",
    "target",
]

DETECTOR_SCORE_COLS = [
    "detector_score",
    "detector_score_norm",
    "patchcore_score",
    "fastflow_score",
    "efficientad_score",
    "anomaly_score",
    "image_score",
    "score",
]

VLM_SCORE_COLS = [
    "vlm_score",
    "vlm_score_norm",
    "crop_vlm_score",
    "context_vlm_score",
    "clip_score",
    "winclip_score",
    "anomaly_vlm_score",
]

QUALITY_COLS = [
    "candidate_quality",
    "candidate_quality_norm",
    "quality",
    "quality_score",
    "crop_quality",
]

CONSISTENCY_COLS = [
    "high_high_consistency",
    "consistency",
    "K",
    "agreement",
]


def read_table_sample(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=5000)
        if path.suffix.lower() in [".tsv", ".txt"]:
            return pd.read_csv(path, sep="\t", nrows=5000)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= 5000:
                        break
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return pd.DataFrame(rows)
    except Exception:
        return None

    return None


def safe_unique_values(df: pd.DataFrame, col: str, max_values: int = 50) -> list[str]:
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).unique().tolist()
    return vals[:max_values]


def find_first_existing(cols: set[str], candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return ""


def detect_categories(df: pd.DataFrame, path: Path) -> tuple[str, list[str]]:
    cols = set(df.columns)

    for c in CATEGORY_COLS:
        real_col = find_first_existing(cols, [c])
        if real_col:
            vals = safe_unique_values(df, real_col, max_values=200)
            found = [cat for cat in AD2_CATEGORIES if cat in vals]
            return real_col, found

    # Fallback: search in path strings or image_key-like columns.
    text_cols = []
    for c in IMAGE_ID_COLS:
        real_col = find_first_existing(cols, [c])
        if real_col:
            text_cols.append(real_col)

    found = set()

    path_str = str(path).replace("\\", "/")
    for cat in AD2_CATEGORIES:
        if cat in path_str:
            found.add(cat)

    for c in text_cols:
        sample = df[c].dropna().astype(str).head(5000).tolist()
        joined = "\n".join(sample)
        for cat in AD2_CATEGORIES:
            if cat in joined:
                found.add(cat)

    return "", sorted(found)


def scan_file(path: Path) -> dict:
    df = read_table_sample(path)

    base = {
        "file": str(path.relative_to(ROOT)),
        "suffix": path.suffix.lower(),
        "readable": False,
        "num_sample_rows": None,
        "num_cols": None,
        "category_col": "",
        "ad2_categories_found": "",
        "ad2_coverage_count": 0,
        "image_id_col": "",
        "label_col": "",
        "detector_score_col": "",
        "vlm_score_col": "",
        "quality_col": "",
        "consistency_col": "",
        "has_image_id": False,
        "has_label": False,
        "has_detector_score": False,
        "has_vlm_score": False,
        "has_quality": False,
        "has_consistency": False,
        "source_role": "unreadable_or_irrelevant",
        "qcr_assembly_value": "none",
        "notes": "",
    }

    if df is None or df.empty or len(df.columns) <= 1:
        return base

    cols = set(df.columns)

    category_col, ad2_found = detect_categories(df, path)

    image_id_col = find_first_existing(cols, IMAGE_ID_COLS)
    label_col = find_first_existing(cols, LABEL_COLS)
    detector_col = find_first_existing(cols, DETECTOR_SCORE_COLS)
    vlm_col = find_first_existing(cols, VLM_SCORE_COLS)
    quality_col = find_first_existing(cols, QUALITY_COLS)
    consistency_col = find_first_existing(cols, CONSISTENCY_COLS)

    has_image = bool(image_id_col)
    has_label = bool(label_col)
    has_detector = bool(detector_col)
    has_vlm = bool(vlm_col)
    has_quality = bool(quality_col)
    has_consistency = bool(consistency_col)

    role_parts = []
    if has_detector:
        role_parts.append("detector")
    if has_vlm:
        role_parts.append("vlm")
    if has_quality:
        role_parts.append("quality")
    if has_consistency:
        role_parts.append("consistency")
    if has_label:
        role_parts.append("label")

    if len(ad2_found) == 0:
        source_role = "non_ad2_or_summary"
        qcr_value = "none"
    elif has_image and has_label and has_detector and has_vlm and has_quality:
        source_role = "ad2_qcr_ready_or_near_ready"
        qcr_value = "high"
    elif has_image and has_label and (has_detector or has_vlm or has_quality):
        source_role = "ad2_partial_per_image_source"
        qcr_value = "medium"
    elif len(ad2_found) > 0:
        source_role = "ad2_summary_or_category_level_source"
        qcr_value = "low"
    else:
        source_role = "unclassified"

    return {
        **base,
        "readable": True,
        "num_sample_rows": len(df),
        "num_cols": len(df.columns),
        "category_col": category_col,
        "ad2_categories_found": ";".join(ad2_found),
        "ad2_coverage_count": len(ad2_found),
        "image_id_col": image_id_col,
        "label_col": label_col,
        "detector_score_col": detector_col,
        "vlm_score_col": vlm_col,
        "quality_col": quality_col,
        "consistency_col": consistency_col,
        "has_image_id": has_image,
        "has_label": has_label,
        "has_detector_score": has_detector,
        "has_vlm_score": has_vlm,
        "has_quality": has_quality,
        "has_consistency": has_consistency,
        "source_role": source_role,
        "qcr_assembly_value": qcr_value,
        "notes": ";".join(role_parts),
    }


def iter_candidate_files() -> list[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".jsonl", ".parquet"}
    files = []

    for root in SCAN_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in suffixes:
                continue

            # Skip obvious huge raw logs or unrelated files.
            name = p.name.lower()
            if "events.out.tfevents" in name:
                continue
            if p.stat().st_size > 500 * 1024 * 1024:
                continue

            files.append(p)

    return sorted(files)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    files = iter_candidate_files()

    rows = []
    for i, p in enumerate(files, start=1):
        if i % 100 == 0:
            print(f"[SCAN] {i}/{len(files)} {p}")
        rows.append(scan_file(p))

    inv = pd.DataFrame(rows)
    inv.to_csv(OUT_INVENTORY, index=False, lineterminator="\n")

    if inv.empty:
        candidates = inv
    else:
        candidates = inv[
            (inv["ad2_coverage_count"] > 0)
            & (inv["qcr_assembly_value"].isin(["high", "medium", "low"]))
        ].copy()

        order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        candidates["_order"] = candidates["qcr_assembly_value"].map(order).fillna(9)
        candidates = candidates.sort_values(
            ["_order", "ad2_coverage_count", "file"],
            ascending=[True, False, True],
        ).drop(columns=["_order"])

    candidates.to_csv(OUT_CANDIDATES, index=False, lineterminator="\n")

    high = int((candidates["qcr_assembly_value"] == "high").sum()) if not candidates.empty else 0
    medium = int((candidates["qcr_assembly_value"] == "medium").sum()) if not candidates.empty else 0
    low = int((candidates["qcr_assembly_value"] == "low").sum()) if not candidates.empty else 0

    lines = [
        "# Stage 18-B0 AD2 QCR Source Inventory",
        "",
        "## Purpose",
        "",
        "Scan existing result/run files to determine whether AD2 four-category QCR predictions can be assembled from existing per-image sources.",
        "",
        "## Summary",
        "",
        f"- scanned files: `{len(files)}`",
        f"- AD2 high-value QCR-ready/near-ready files: `{high}`",
        f"- AD2 medium-value partial per-image files: `{medium}`",
        f"- AD2 low-value summary/category-level files: `{low}`",
        "",
        "## Candidate files",
        "",
        "| File | Coverage | Role | Value | Image ID | Label | Detector | VLM | Quality | Notes |",
        "|---|---:|---|---|---|---|---|---|---|---|",
    ]

    if candidates.empty:
        lines.append("| none | 0/4 | none | none |  |  |  |  |  | No AD2 source candidates found. |")
    else:
        for _, r in candidates.head(80).iterrows():
            lines.append(
                f"| `{r['file']}` | {r['ad2_coverage_count']}/4 | "
                f"{r['source_role']} | {r['qcr_assembly_value']} | "
                f"{r['image_id_col']} | {r['label_col']} | "
                f"{r['detector_score_col']} | {r['vlm_score_col']} | "
                f"{r['quality_col']} | {r['notes']} |"
            )

    lines += [
        "",
        "## Decision rule",
        "",
        "- If high-value files exist, proceed to Stage 18-B1: assemble AD2 QCR predictions from existing sources.",
        "- If only medium-value files exist, inspect whether detector/VLM/quality sources can be joined by image key.",
        "- If no high/medium-value files exist, proceed to Stage 18-C: generate AD2 QCR predictions from scratch.",
        "",
    ]

    if high > 0:
        decision = "proceed_to_stage18_b1_assemble_existing_sources"
    elif medium > 0:
        decision = "inspect_medium_sources_then_assemble_or_generate_missing_parts"
    else:
        decision = "proceed_to_stage18_c_generate_ad2_qcr_predictions"

    lines += [
        "## Recommended next action",
        "",
        f"`{decision}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_INVENTORY)
    print("[DONE]", OUT_CANDIDATES)
    print("[DONE]", OUT_REPORT)
    print()
    print("high:", high, "medium:", medium, "low:", low)
    print()
    if not candidates.empty:
        print(candidates.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
