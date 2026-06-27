from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

STAGE10_REGIONS = ROOT / "results/stage10_dataset_expansion/stage10_d_patchcore_candidate_regions.csv"
STAGE10_SUMMARY = ROOT / "results/stage10_dataset_expansion/stage10_d_patchcore_candidate_summary.csv"
STAGE11_REGIONS = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv"
STAGE11_SUMMARY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv"
STAGE11A_MAPPING = ROOT / "results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_mapping.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_MATCHED = OUT_DIR / "stage11_f1_vial_path_aligned_drift_matched.csv"
OUT_SUMMARY = OUT_DIR / "stage11_f1_vial_path_aligned_drift_summary.csv"
OUT_REPORT = DOC_DIR / "stage11_f1_vial_path_aligned_drift_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def strip_generated_prefix(stem: str) -> str:
    # Stage 11 adapter file names look like 000123_original_name.png.
    # Keep original AD2 stems like 000_shift_1; only remove six-digit generated prefix.
    return re.sub(r"^\d{6}_", "", stem)


def path_label(path_text: object, gt_label: object = None) -> str:
    text = "" if pd.isna(path_text) else str(path_text).replace("\\", "/").lower()

    if "/test/good/" in text or "/train/good/" in text or text.endswith("/good"):
        return "good"
    if "/test/bad/" in text or "/ground_truth/bad/" in text or text.endswith("/bad"):
        return "bad"

    if gt_label is not None and not pd.isna(gt_label):
        s = str(gt_label).strip().lower()
        if s in {"1", "true", "bad", "anomaly", "abnormal", "tensor(1)"}:
            return "bad"
        if s in {"0", "false", "good", "normal", "tensor(0)"}:
            return "good"
        try:
            return "bad" if float(s) > 0 else "good"
        except Exception:
            pass

    return "unknown"


def image_stem(path_text: object) -> str:
    text = "" if pd.isna(path_text) else str(path_text)
    stem = Path(text).stem
    return strip_generated_prefix(stem)


def make_keys(path_text: object, gt_label: object = None) -> dict:
    stem = image_stem(path_text)
    label = path_label(path_text, gt_label)

    return {
        "key_stem": stem,
        "key_label_stem": f"{label}:{stem}",
    }


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_regions(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    out = df.copy()

    if "category" in out.columns:
        out = out[out["category"].astype(str) == "vial"].copy()

    image_col = pick_col(out, ["image_path", "source_image_path", "target_image_path"])
    if image_col is None:
        raise RuntimeError(f"{stage}: cannot find image path column.")

    gt_col = pick_col(out, ["gt_label", "is_anomaly", "gt_binary"])
    rank_col = pick_col(out, ["candidate_rank", "rank", "candidate_id"])

    if rank_col is None:
        out["candidate_rank"] = out.groupby(image_col).cumcount()
        rank_col = "candidate_rank"

    keys = out.apply(
        lambda r: make_keys(r[image_col], r[gt_col] if gt_col else None),
        axis=1,
        result_type="expand",
    )

    out["key_stem"] = keys["key_stem"]
    out["key_label_stem"] = keys["key_label_stem"]
    out["candidate_rank_norm"] = pd.to_numeric(out[rank_col], errors="coerce").fillna(999999).astype(int)

    for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    width_col = pick_col(out, ["image_width", "width", "img_width"])
    height_col = pick_col(out, ["image_height", "height", "img_height"])

    out["image_width_norm"] = pd.to_numeric(out[width_col], errors="coerce") if width_col else np.nan
    out["image_height_norm"] = pd.to_numeric(out[height_col], errors="coerce") if height_col else np.nan

    out["bbox_area"] = (
        (out["bbox_x2"] - out["bbox_x1"]).clip(lower=0)
        * (out["bbox_y2"] - out["bbox_y1"]).clip(lower=0)
    )
    out["image_area"] = out["image_width_norm"] * out["image_height_norm"]
    out["bbox_area_ratio"] = out["bbox_area"] / out["image_area"].replace(0, np.nan)

    cov_col = pick_col(out, ["tight_candidate_covers_gt_ratio", "candidate_covers_gt_ratio"])
    out["tight_gt_coverage"] = pd.to_numeric(out[cov_col], errors="coerce") if cov_col else np.nan

    ctx_cov_col = pick_col(out, ["context_candidate_covers_gt_ratio", "context_covers_gt_ratio"])
    out["context_gt_coverage"] = pd.to_numeric(out[ctx_cov_col], errors="coerce") if ctx_cov_col else np.nan

    return out


def top1(df: pd.DataFrame, key_col: str, stage: str) -> pd.DataFrame:
    t = (
        df.sort_values([key_col, "candidate_rank_norm"])
        .groupby(key_col, as_index=False)
        .first()
    )

    # Avoid duplicate columns when key_col is already key_stem or key_label_stem.
    keep = [
        key_col,
        "candidate_rank_norm",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_area_ratio",
        "tight_gt_coverage",
        "context_gt_coverage",
        "key_stem",
        "key_label_stem",
    ]

    seen = set()
    unique_keep = []
    for c in keep:
        if c in t.columns and c not in seen:
            unique_keep.append(c)
            seen.add(c)

    t = t[unique_keep].copy()
    t = t.add_prefix(f"{stage}_")
    t = t.rename(columns={f"{stage}_{key_col}": key_col})

    # Defensive check: merge keys must be single, unique column labels.
    if t.columns.duplicated().any():
        duplicated = t.columns[t.columns.duplicated()].tolist()
        raise RuntimeError(f"{stage}: duplicated columns after top1 normalization: {duplicated}")

    return t


def bbox_iou(row: pd.Series) -> float:
    need = [
        "stage10_bbox_x1", "stage10_bbox_y1", "stage10_bbox_x2", "stage10_bbox_y2",
        "stage11_bbox_x1", "stage11_bbox_y1", "stage11_bbox_x2", "stage11_bbox_y2",
    ]

    vals = [row.get(c, np.nan) for c in need]
    if any(pd.isna(v) for v in vals):
        return np.nan

    ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = vals

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return np.nan

    return float(inter / union)


def match_stats(stage10: pd.DataFrame, stage11: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, dict]:
    t10 = top1(stage10, key_col, "stage10")
    t11 = top1(stage11, key_col, "stage11")

    matched = t10.merge(t11, on=key_col, how="inner")
    matched["top1_bbox_iou"] = matched.apply(bbox_iou, axis=1)

    stats = {
        "match_key": key_col,
        "stage10_unique_images": int(stage10[key_col].nunique()),
        "stage11_unique_images": int(stage11[key_col].nunique()),
        "matched_images": int(len(matched)),
        "stage10_candidate_rows": int(len(stage10)),
        "stage11_candidate_rows": int(len(stage11)),
        "stage10_mean_candidates_per_image": float(len(stage10) / max(1, stage10[key_col].nunique())),
        "stage11_mean_candidates_per_image": float(len(stage11) / max(1, stage11[key_col].nunique())),
        "mean_top1_bbox_iou": float(matched["top1_bbox_iou"].mean()) if len(matched) else np.nan,
        "median_top1_bbox_iou": float(matched["top1_bbox_iou"].median()) if len(matched) else np.nan,
        "mean_abs_bbox_area_ratio_diff": float(
            (matched["stage11_bbox_area_ratio"] - matched["stage10_bbox_area_ratio"]).abs().mean()
        ) if len(matched) else np.nan,
    }

    return matched, stats


def load_original_summaries() -> dict:
    out = {}

    if STAGE10_SUMMARY.exists():
        s10 = pd.read_csv(STAGE10_SUMMARY)
        row = s10[s10["category"] == "vial"].iloc[0]
        out["stage10_summary_num_images"] = int(row["num_images"])
        out["stage10_summary_candidate_rows"] = int(row["num_candidate_rows"])

    if STAGE11_SUMMARY.exists():
        s11 = pd.read_csv(STAGE11_SUMMARY)
        row = s11[s11["category"] == "vial"].iloc[0]
        out["stage11_summary_num_images"] = int(row["num_images"])
        out["stage11_summary_candidate_rows"] = int(row["num_candidate_rows"])

    return out


def write_report(summary: pd.DataFrame, matched: pd.DataFrame, best_key: str, original: dict) -> None:
    lines = []

    lines.append("# Stage 11-F1 Vial Path-aligned Candidate Drift Diagnostic")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report fixes the Stage 11-F image matching issue by comparing Stage 10 and Stage 11 vial candidates using path-aware keys.")
    lines.append("It does not rerun PatchCore, VLM inference, or crop generation.")
    lines.append("")
    lines.append("## 2. Original Candidate Summary")
    lines.append("")
    lines.append("| Source | Images | Candidate rows |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Stage 10-D summary | {original.get('stage10_summary_num_images', '')} | {original.get('stage10_summary_candidate_rows', '')} |")
    lines.append(f"| Stage 11-C summary | {original.get('stage11_summary_num_images', '')} | {original.get('stage11_summary_candidate_rows', '')} |")
    lines.append("")
    lines.append("## 3. Matching Diagnostics")
    lines.append("")
    lines.append("| Match key | Stage10 images | Stage11 images | Matched images | Stage10 cand/img | Stage11 cand/img | Mean top1 bbox IoU | Median IoU | Mean abs bbox-area diff |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['match_key']} | {int(r['stage10_unique_images'])} | {int(r['stage11_unique_images'])} | "
            f"{int(r['matched_images'])} | {float(r['stage10_mean_candidates_per_image']):.4f} | "
            f"{float(r['stage11_mean_candidates_per_image']):.4f} | "
            f"{float(r['mean_top1_bbox_iou']):.4f} | {float(r['median_top1_bbox_iou']):.4f} | "
            f"{float(r['mean_abs_bbox_area_ratio_diff']):.4f} |"
        )

    lines.append("")
    lines.append("## 4. Selected Matching Key")
    lines.append("")
    lines.append(f"The selected matching key is `{best_key}`, because it gives the largest matched image count while preserving label information when possible.")
    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append("If the selected key still matches far fewer than 71 images, the previous Stage 11-F conclusion is not reliable and the next step should inspect the exact unmatched image lists.")
    lines.append("If the selected key matches close to 71 images and bbox IoU is high, the vial inconsistency is less likely to be from candidate region selection and more likely to come from crop image construction, prompt/backend differences, or aggregation.")
    lines.append("If the selected key matches close to 71 images but bbox IoU is low, Stage 11-C should be patched to reuse the Stage 10 candidate policy.")
    lines.append("")
    lines.append("## 6. Output")
    lines.append("")
    lines.append(f"- Matched CSV: `{OUT_MATCHED.relative_to(ROOT)}`")
    lines.append(f"- Summary CSV: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    raw10 = read_csv(STAGE10_REGIONS)
    raw11 = read_csv(STAGE11_REGIONS)

    stage10 = normalize_regions(raw10, "stage10")
    stage11 = normalize_regions(raw11, "stage11")

    candidates = ["key_label_stem", "key_stem"]

    matched_tables = {}
    stats_rows = []

    for key in candidates:
        matched, stats = match_stats(stage10, stage11, key)
        matched_tables[key] = matched
        stats_rows.append(stats)

    summary = pd.DataFrame(stats_rows)

    best_key = (
        summary.sort_values(["matched_images", "mean_top1_bbox_iou"], ascending=[False, False])
        .iloc[0]["match_key"]
    )

    matched = matched_tables[str(best_key)].copy()
    original = load_original_summaries()

    matched.to_csv(OUT_MATCHED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(summary, matched, str(best_key), original)

    print("[DONE]", OUT_MATCHED)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print("")
    print("===== original summaries =====")
    print(original)
    print("")
    print("===== matching summary =====")
    print(summary.to_string(index=False))
    print("")
    print("===== selected key =====")
    print(best_key)
    print("")
    print("===== matched head =====")
    cols = [
        str(best_key),
        "stage10_candidate_rank_norm",
        "stage11_candidate_rank_norm",
        "top1_bbox_iou",
        "stage10_bbox_area_ratio",
        "stage11_bbox_area_ratio",
        "stage10_tight_gt_coverage",
        "stage11_tight_gt_coverage",
    ]
    cols = [c for c in cols if c in matched.columns]
    print(matched[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
