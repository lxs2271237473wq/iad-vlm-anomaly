from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

STAGE10_REGIONS = ROOT / "results/stage10_dataset_expansion/stage10_d_patchcore_candidate_regions.csv"
STAGE10_SUMMARY = ROOT / "results/stage10_dataset_expansion/stage10_d_patchcore_candidate_summary.csv"
STAGE10_VLM = ROOT / "results/stage10_dataset_expansion/stage10_f_multiscale_context_summary.csv"

STAGE11_REGIONS = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv"
STAGE11_SUMMARY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv"
STAGE11_VLM = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_summary.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_MATCHED = OUT_DIR / "stage11_f_vial_candidate_policy_drift_matched.csv"
OUT_SUMMARY = OUT_DIR / "stage11_f_vial_candidate_policy_drift_summary.csv"
OUT_REPORT = DOC_DIR / "stage11_f_vial_candidate_policy_drift_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def image_key(path_text: object) -> str:
    text = "" if pd.isna(path_text) else str(path_text)
    stem = Path(text).stem

    # Stage 11 adapter names often look like 000123_original_stem.png
    stem = re.sub(r"^\d{6}_", "", stem)

    # Remove common crop suffixes if a crop path accidentally appears
    stem = re.sub(r"_cand\d+.*$", "", stem)

    return stem


def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_first_contains(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
    cols = list(df.columns)
    for pat in patterns:
        for c in cols:
            if pat.lower() in c.lower():
                return c
    return None


def normalize_regions(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    out = df.copy()

    if "category" in out.columns:
        out = out[out["category"].astype(str) == "vial"].copy()

    if out.empty:
        raise RuntimeError(f"{stage}: no vial rows after filtering.")

    image_col = pick_col(out, ["image_path", "source_image_path", "target_image_path"])
    if image_col is None:
        raise RuntimeError(f"{stage}: cannot find image path column. columns={out.columns.tolist()}")

    rank_col = pick_col(out, ["candidate_rank", "rank", "candidate_id"])
    if rank_col is None:
        out["candidate_rank"] = out.groupby(image_col).cumcount()
        rank_col = "candidate_rank"

    out["image_key"] = out[image_col].map(image_key)
    out["candidate_rank"] = pd.to_numeric(out[rank_col], errors="coerce").fillna(999999).astype(int)

    # bbox columns for tight candidate
    bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    if not all(c in out.columns for c in bbox_cols):
        alt = ["x1", "y1", "x2", "y2"]
        if all(c in out.columns for c in alt):
            for a, b in zip(alt, bbox_cols):
                out[b] = out[a]
        else:
            for c in bbox_cols:
                if c not in out.columns:
                    out[c] = np.nan

    width_col = pick_col(out, ["image_width", "width", "img_width"])
    height_col = pick_col(out, ["image_height", "height", "img_height"])

    if width_col is None:
        out["image_width"] = np.nan
        width_col = "image_width"

    if height_col is None:
        out["image_height"] = np.nan
        height_col = "image_height"

    out["image_width_norm"] = pd.to_numeric(out[width_col], errors="coerce")
    out["image_height_norm"] = pd.to_numeric(out[height_col], errors="coerce")

    for c in bbox_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["bbox_area"] = (out["bbox_x2"] - out["bbox_x1"]).clip(lower=0) * (out["bbox_y2"] - out["bbox_y1"]).clip(lower=0)
    out["image_area"] = out["image_width_norm"] * out["image_height_norm"]
    out["bbox_area_ratio"] = out["bbox_area"] / out["image_area"].replace(0, np.nan)

    # context bbox columns: Stage 11 uses context_1p50_x1...
    context_cols = ["context_1p50_x1", "context_1p50_y1", "context_1p50_x2", "context_1p50_y2"]
    if all(c in out.columns for c in context_cols):
        for c in context_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out["context_area"] = (
            (out["context_1p50_x2"] - out["context_1p50_x1"]).clip(lower=0)
            * (out["context_1p50_y2"] - out["context_1p50_y1"]).clip(lower=0)
        )
        out["context_area_ratio"] = out["context_area"] / out["image_area"].replace(0, np.nan)
    else:
        out["context_area_ratio"] = np.nan

    # coverage columns
    tight_cov_col = pick_col(out, [
        "tight_candidate_covers_gt_ratio",
        "candidate_covers_gt_ratio",
        "top1_tight_mean_gt_coverage_anomaly",
    ])
    context_cov_col = pick_col(out, [
        "context_candidate_covers_gt_ratio",
        "context_covers_gt_ratio",
        "top1_context_mean_gt_coverage_anomaly",
    ])

    out["tight_gt_coverage"] = pd.to_numeric(out[tight_cov_col], errors="coerce") if tight_cov_col else np.nan
    out["context_gt_coverage"] = pd.to_numeric(out[context_cov_col], errors="coerce") if context_cov_col else np.nan

    score_col = pick_col(out, ["candidate_score_max", "pred_score", "patchcore_score"])
    out["candidate_score"] = pd.to_numeric(out[score_col], errors="coerce") if score_col else np.nan

    keep = [
        "image_key",
        "candidate_rank",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_area_ratio",
        "context_area_ratio",
        "tight_gt_coverage",
        "context_gt_coverage",
        "candidate_score",
    ]

    return out[keep].copy()


def top1(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    out = df.sort_values(["image_key", "candidate_rank"]).groupby("image_key", as_index=False).first()
    out = out.add_prefix(f"{stage}_")
    out = out.rename(columns={f"{stage}_image_key": "image_key"})
    return out


def bbox_iou(row: pd.Series) -> float:
    needed = [
        "stage10_bbox_x1", "stage10_bbox_y1", "stage10_bbox_x2", "stage10_bbox_y2",
        "stage11_bbox_x1", "stage11_bbox_y1", "stage11_bbox_x2", "stage11_bbox_y2",
    ]

    vals = [row.get(c, np.nan) for c in needed]
    if any(pd.isna(v) for v in vals):
        return np.nan

    ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = vals

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return np.nan

    return float(inter / union)


def summarize_regions(stage10: pd.DataFrame, stage11: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name, df in [("stage10", stage10), ("stage11", stage11)]:
        rows.append({
            "scope": name,
            "num_candidate_rows": int(len(df)),
            "num_images": int(df["image_key"].nunique()),
            "mean_candidates_per_image": float(len(df) / max(1, df["image_key"].nunique())),
            "mean_top1_bbox_area_ratio": float(top1(df, name)[f"{name}_bbox_area_ratio"].mean()),
            "mean_top1_context_area_ratio": float(top1(df, name)[f"{name}_context_area_ratio"].mean()) if f"{name}_context_area_ratio" in top1(df, name).columns else np.nan,
            "mean_top1_tight_gt_coverage": float(top1(df, name)[f"{name}_tight_gt_coverage"].mean()),
            "mean_top1_context_gt_coverage": float(top1(df, name)[f"{name}_context_gt_coverage"].mean()),
        })

    rows.append({
        "scope": "matched_stage10_vs_stage11",
        "num_candidate_rows": "",
        "num_images": int(len(matched)),
        "mean_candidates_per_image": "",
        "mean_top1_bbox_area_ratio": "",
        "mean_top1_context_area_ratio": "",
        "mean_top1_tight_gt_coverage": "",
        "mean_top1_context_gt_coverage": "",
        "mean_top1_bbox_iou": float(matched["top1_bbox_iou"].mean()),
        "median_top1_bbox_iou": float(matched["top1_bbox_iou"].median()),
        "mean_abs_bbox_area_ratio_diff": float((matched["stage11_bbox_area_ratio"] - matched["stage10_bbox_area_ratio"]).abs().mean()),
        "mean_abs_tight_coverage_diff": float((matched["stage11_tight_gt_coverage"] - matched["stage10_tight_gt_coverage"]).abs().mean()),
    })

    return pd.DataFrame(rows)


def get_vlm_row(path: Path, category: str, method: str) -> Optional[pd.Series]:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    part = df[(df["category"] == category) & (df["method"] == method)]

    if part.empty:
        return None

    return part.iloc[0]


def f4(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def write_report(summary: pd.DataFrame, matched: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    s10_full = get_vlm_row(STAGE10_VLM, "vial", "full_image")
    s10_ctx = get_vlm_row(STAGE10_VLM, "vial", "context_1.50_top1")

    s11_full = get_vlm_row(STAGE11_VLM, "vial", "full_image")
    s11_ctx = get_vlm_row(STAGE11_VLM, "vial", "context_1.50_top1")

    lines = []
    lines.append("# Stage 11-F Vial Candidate Policy Drift Diagnostic")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This diagnostic compares the single-category Stage 10 vial candidate construction with the multi-category Stage 11 vial candidate construction.")
    lines.append("It does not rerun PatchCore, VLM inference, or crop generation.")
    lines.append("")
    lines.append("## 2. Why this is needed")
    lines.append("")
    lines.append("Stage 10-G reported positive vial context evidence, while Stage 11-D did not reproduce the same margin.")
    lines.append("Therefore, this diagnostic checks whether the mismatch is caused by candidate policy drift rather than by the idea of context-aware crop reasoning itself.")
    lines.append("")
    lines.append("## 3. Vial VLM Result Difference")
    lines.append("")
    lines.append("| Stage | full_image AUROC | context_1.50_top1 AUROC | ΔAUROC |")
    lines.append("|---|---:|---:|---:|")

    if s10_full is not None and s10_ctx is not None:
        lines.append(
            f"| Stage 10-F/G | {f4(s10_full['auroc'])} | {f4(s10_ctx['auroc'])} | "
            f"{f4(float(s10_ctx['auroc']) - float(s10_full['auroc']))} |"
        )

    if s11_full is not None and s11_ctx is not None:
        lines.append(
            f"| Stage 11-D | {f4(s11_full['auroc'])} | {f4(s11_ctx['auroc'])} | "
            f"{f4(float(s11_ctx['auroc']) - float(s11_full['auroc']))} |"
        )

    lines.append("")
    lines.append("## 4. Candidate Construction Summary")
    lines.append("")
    lines.append("| Scope | Candidate rows | Images | Mean cand/img | Top1 bbox area | Top1 context area | Top1 tight GT coverage | Top1 context GT coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in summary.iterrows():
        if r["scope"] == "matched_stage10_vs_stage11":
            continue

        lines.append(
            f"| {r['scope']} | {r['num_candidate_rows']} | {r['num_images']} | "
            f"{f4(r['mean_candidates_per_image'])} | {f4(r['mean_top1_bbox_area_ratio'])} | "
            f"{f4(r['mean_top1_context_area_ratio'])} | {f4(r['mean_top1_tight_gt_coverage'])} | "
            f"{f4(r['mean_top1_context_gt_coverage'])} |"
        )

    lines.append("")
    lines.append("## 5. Matched-image Top1 BBox Difference")
    lines.append("")
    matched_row = summary[summary["scope"] == "matched_stage10_vs_stage11"]
    if not matched_row.empty:
        r = matched_row.iloc[0]
        lines.append(f"- Matched images: `{int(r['num_images'])}`")
        lines.append(f"- Mean top1 bbox IoU: `{f4(r.get('mean_top1_bbox_iou', None))}`")
        lines.append(f"- Median top1 bbox IoU: `{f4(r.get('median_top1_bbox_iou', None))}`")
        lines.append(f"- Mean absolute bbox area-ratio difference: `{f4(r.get('mean_abs_bbox_area_ratio_diff', None))}`")
        lines.append(f"- Mean absolute tight GT coverage difference: `{f4(r.get('mean_abs_tight_coverage_diff', None))}`")

    lines.append("")
    lines.append("## 6. Interpretation Rule")
    lines.append("")
    lines.append("- If bbox IoU is low, Stage 10 and Stage 11 are selecting different visual regions.")
    lines.append("- If GT coverage differs strongly, the VLM discrepancy is likely caused by candidate construction.")
    lines.append("- If candidate count or bbox area ratio differs strongly, Stage 11-C should be patched to reuse the Stage 10 candidate policy before extending to fabric.")
    lines.append("")
    lines.append("## 7. Output")
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

    t10 = top1(stage10, "stage10")
    t11 = top1(stage11, "stage11")

    matched = t10.merge(t11, on="image_key", how="inner")
    matched["top1_bbox_iou"] = matched.apply(bbox_iou, axis=1)

    summary = summarize_regions(stage10, stage11, matched)

    matched.to_csv(OUT_MATCHED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    write_report(summary, matched)

    print("[DONE]", OUT_MATCHED)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)

    print("\n===== summary =====")
    print(summary.to_string(index=False))

    print("\n===== matched head =====")
    show_cols = [
        "image_key",
        "stage10_candidate_rank",
        "stage11_candidate_rank",
        "top1_bbox_iou",
        "stage10_bbox_area_ratio",
        "stage11_bbox_area_ratio",
        "stage10_tight_gt_coverage",
        "stage11_tight_gt_coverage",
    ]
    show_cols = [c for c in show_cols if c in matched.columns]
    print(matched[show_cols].head(30).to_string(index=False))

    if matched.empty:
        raise SystemExit("[ERROR] No matched vial images between Stage 10 and Stage 11. Check image key mapping.")


if __name__ == "__main__":
    main()
