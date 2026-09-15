from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(".").resolve()

STAGE10_REGIONS = ROOT / "results/stage10_dataset_expansion/stage10_d_patchcore_candidate_regions.csv"
STAGE10_SUMMARY = ROOT / "results/stage10_dataset_expansion/stage10_d_patchcore_candidate_summary.csv"
STAGE11_REGIONS = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv"
STAGE11_SUMMARY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv"
STAGE11A_MAPPING = ROOT / "results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_mapping.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_MATCHED = OUT_DIR / "stage11_f2_vial_source_aligned_drift_matched.csv"
OUT_UNMATCHED = OUT_DIR / "stage11_f2_vial_source_aligned_drift_unmatched.csv"
OUT_SUMMARY = OUT_DIR / "stage11_f2_vial_source_aligned_drift_summary.csv"
OUT_REPORT = DOC_DIR / "stage11_f2_vial_source_aligned_drift_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def rel_norm(path_text: object) -> str:
    text = "" if pd.isna(path_text) else str(path_text).replace("\\", "/")
    if not text:
        return ""

    root_text = str(ROOT).replace("\\", "/")
    if text.startswith(root_text + "/"):
        text = text[len(root_text) + 1:]

    return text.strip()


def label_from_path_or_gt(path_text: object, gt: object = None) -> str:
    text = rel_norm(path_text).lower()

    if "/test/good/" in text or "/train/good/" in text:
        return "good"
    if "/test/bad/" in text or "/ground_truth/bad/" in text:
        return "bad"

    if gt is not None and not pd.isna(gt):
        s = str(gt).strip().lower()
        if s in {"1", "true", "bad", "anomaly", "abnormal", "tensor(1)"}:
            return "bad"
        if s in {"0", "false", "good", "normal", "tensor(0)"}:
            return "good"
        try:
            return "bad" if float(s) > 0 else "good"
        except Exception:
            pass

    return "unknown"


def strip_adapter_prefix(stem: str) -> str:
    return re.sub(r"^\d{6}_", "", stem)


def source_key_from_source_path(path_text: object, gt: object = None) -> str:
    p = Path(rel_norm(path_text))
    label = label_from_path_or_gt(path_text, gt)
    stem = strip_adapter_prefix(p.stem)
    return f"{label}:{stem}"


def source_key_from_adapter_path(path_text: object, gt: object = None) -> str:
    p = Path(rel_norm(path_text))
    label = label_from_path_or_gt(path_text, gt)
    stem = strip_adapter_prefix(p.stem)
    return f"{label}:{stem}"


def bbox_area_ratio(df: pd.DataFrame, prefix: str) -> pd.Series:
    x1 = pd.to_numeric(df[f"{prefix}_x1"], errors="coerce")
    y1 = pd.to_numeric(df[f"{prefix}_y1"], errors="coerce")
    x2 = pd.to_numeric(df[f"{prefix}_x2"], errors="coerce")
    y2 = pd.to_numeric(df[f"{prefix}_y2"], errors="coerce")
    w = pd.to_numeric(df["image_width"], errors="coerce")
    h = pd.to_numeric(df["image_height"], errors="coerce")

    area = (x2 - x1).clip(lower=0) * (y2 - y1).clip(lower=0)
    return area / (w * h).replace(0, np.nan)


def bbox_iou_row(row: pd.Series, a: str, b: str) -> float:
    cols = [
        f"{a}_x1", f"{a}_y1", f"{a}_x2", f"{a}_y2",
        f"{b}_x1", f"{b}_y1", f"{b}_x2", f"{b}_y2",
    ]
    vals = [row.get(c, np.nan) for c in cols]

    if any(pd.isna(v) for v in vals):
        return np.nan

    ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = [float(v) for v in vals]

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

    return inter / union


def resolve_path(path_text: object) -> Path | None:
    text = rel_norm(path_text)
    if not text:
        return None

    p = Path(text)
    if p.is_absolute():
        return p

    return ROOT / p


def mask_coverage(mask_path: object, bbox: Tuple[float, float, float, float], image_size: Tuple[int, int]) -> float:
    p = resolve_path(mask_path)
    if p is None or not p.exists():
        return np.nan

    try:
        mask = Image.open(p).convert("L").resize(image_size, resample=Image.NEAREST)
        arr = np.asarray(mask) > 0
    except Exception:
        return np.nan

    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    w, h = image_size

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    gt = float(arr.sum())
    if gt <= 0 or x2 <= x1 or y2 <= y1:
        return np.nan

    hit = float(arr[y1:y2, x1:x2].sum())
    return hit / gt


def build_stage11_target_to_source() -> Dict[str, str]:
    mapping = read_csv(STAGE11A_MAPPING)
    mapping = mapping[mapping["category"].astype(str) == "vial"].copy()

    out = {}

    for _, r in mapping.iterrows():
        target = rel_norm(r["target_image_path"])
        source = rel_norm(r["source_image_path"])
        out[target] = source
        out[Path(target).name] = source

    return out


def normalize_stage10() -> pd.DataFrame:
    df = read_csv(STAGE10_REGIONS)
    df = df[df["category"].astype(str) == "vial"].copy()

    df["source_key"] = df.apply(lambda r: source_key_from_adapter_path(r["image_path"], r.get("gt_label", None)), axis=1)
    df["candidate_rank_norm"] = pd.to_numeric(df["candidate_rank"], errors="coerce").fillna(999999).astype(int)

    out = pd.DataFrame()
    out["source_key"] = df["source_key"]
    out["image_path_stage10"] = df["image_path"]
    out["mask_path_stage10"] = df["mask_path"]
    out["candidate_rank_stage10"] = df["candidate_rank_norm"]
    out["pred_score_stage10"] = pd.to_numeric(df["pred_score"], errors="coerce")
    out["image_width"] = pd.to_numeric(df["image_width"], errors="coerce")
    out["image_height"] = pd.to_numeric(df["image_height"], errors="coerce")

    out["stage10_tight_x1"] = pd.to_numeric(df["x1"], errors="coerce")
    out["stage10_tight_y1"] = pd.to_numeric(df["y1"], errors="coerce")
    out["stage10_tight_x2"] = pd.to_numeric(df["x2"], errors="coerce")
    out["stage10_tight_y2"] = pd.to_numeric(df["y2"], errors="coerce")

    out["stage10_context_x1"] = pd.to_numeric(df["crop_x1"], errors="coerce")
    out["stage10_context_y1"] = pd.to_numeric(df["crop_y1"], errors="coerce")
    out["stage10_context_x2"] = pd.to_numeric(df["crop_x2"], errors="coerce")
    out["stage10_context_y2"] = pd.to_numeric(df["crop_y2"], errors="coerce")

    out["stage10_tight_area_ratio"] = bbox_area_ratio(out.rename(columns={
        "stage10_tight_x1": "tight_x1",
        "stage10_tight_y1": "tight_y1",
        "stage10_tight_x2": "tight_x2",
        "stage10_tight_y2": "tight_y2",
    }), "tight")

    out["stage10_context_area_ratio"] = bbox_area_ratio(out.rename(columns={
        "stage10_context_x1": "context_x1",
        "stage10_context_y1": "context_y1",
        "stage10_context_x2": "context_x2",
        "stage10_context_y2": "context_y2",
    }), "context")

    return out


def normalize_stage11() -> pd.DataFrame:
    df = read_csv(STAGE11_REGIONS)
    df = df[df["category"].astype(str) == "vial"].copy()

    target_to_source = build_stage11_target_to_source()

    def get_source_path(image_path):
        rel = rel_norm(image_path)
        return target_to_source.get(rel, target_to_source.get(Path(rel).name, ""))

    df["source_image_path"] = df["image_path"].map(get_source_path)
    df["source_key"] = df.apply(
        lambda r: source_key_from_source_path(r["source_image_path"], r.get("gt_label", None))
        if r["source_image_path"]
        else source_key_from_adapter_path(r["image_path"], r.get("gt_label", None)),
        axis=1,
    )

    df["candidate_rank_norm"] = pd.to_numeric(df["candidate_rank"], errors="coerce").fillna(999999).astype(int)

    out = pd.DataFrame()
    out["source_key"] = df["source_key"]
    out["source_image_path_stage11"] = df["source_image_path"]
    out["image_path_stage11"] = df["image_path"]
    out["mask_path_stage11"] = df["mask_path"]
    out["candidate_rank_stage11"] = df["candidate_rank_norm"]
    out["pred_score_stage11"] = pd.to_numeric(df["pred_score"], errors="coerce")
    out["image_width"] = pd.to_numeric(df["image_width"], errors="coerce")
    out["image_height"] = pd.to_numeric(df["image_height"], errors="coerce")

    out["stage11_tight_x1"] = pd.to_numeric(df["bbox_x1"], errors="coerce")
    out["stage11_tight_y1"] = pd.to_numeric(df["bbox_y1"], errors="coerce")
    out["stage11_tight_x2"] = pd.to_numeric(df["bbox_x2"], errors="coerce")
    out["stage11_tight_y2"] = pd.to_numeric(df["bbox_y2"], errors="coerce")

    out["stage11_context_x1"] = pd.to_numeric(df["context_1p50_x1"], errors="coerce")
    out["stage11_context_y1"] = pd.to_numeric(df["context_1p50_y1"], errors="coerce")
    out["stage11_context_x2"] = pd.to_numeric(df["context_1p50_x2"], errors="coerce")
    out["stage11_context_y2"] = pd.to_numeric(df["context_1p50_y2"], errors="coerce")

    out["stage11_tight_area_ratio"] = bbox_area_ratio(out.rename(columns={
        "stage11_tight_x1": "tight_x1",
        "stage11_tight_y1": "tight_y1",
        "stage11_tight_x2": "tight_x2",
        "stage11_tight_y2": "tight_y2",
    }), "tight")

    out["stage11_context_area_ratio"] = bbox_area_ratio(out.rename(columns={
        "stage11_context_x1": "context_x1",
        "stage11_context_y1": "context_y1",
        "stage11_context_x2": "context_x2",
        "stage11_context_y2": "context_y2",
    }), "context")

    return out


def top1(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    return (
        df.sort_values(["source_key", f"candidate_rank_{stage}"])
        .groupby("source_key", as_index=False)
        .first()
    )


def add_coverage(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in matched.iterrows():
        w = int(r["image_width"])
        h = int(r["image_height"])

        s10_tight = (
            r["stage10_tight_x1"], r["stage10_tight_y1"],
            r["stage10_tight_x2"], r["stage10_tight_y2"],
        )
        s10_context = (
            r["stage10_context_x1"], r["stage10_context_y1"],
            r["stage10_context_x2"], r["stage10_context_y2"],
        )
        s11_tight = (
            r["stage11_tight_x1"], r["stage11_tight_y1"],
            r["stage11_tight_x2"], r["stage11_tight_y2"],
        )
        s11_context = (
            r["stage11_context_x1"], r["stage11_context_y1"],
            r["stage11_context_x2"], r["stage11_context_y2"],
        )

        rows.append({
            "stage10_tight_gt_coverage": mask_coverage(r["mask_path_stage10"], s10_tight, (w, h)),
            "stage10_context_gt_coverage": mask_coverage(r["mask_path_stage10"], s10_context, (w, h)),
            "stage11_tight_gt_coverage": mask_coverage(r["mask_path_stage11"], s11_tight, (w, h)),
            "stage11_context_gt_coverage": mask_coverage(r["mask_path_stage11"], s11_context, (w, h)),
        })

    cov = pd.DataFrame(rows)
    return pd.concat([matched.reset_index(drop=True), cov], axis=1)


def original_summary() -> Dict[str, object]:
    out = {}

    if STAGE10_SUMMARY.exists():
        s10 = pd.read_csv(STAGE10_SUMMARY)
        r = s10[s10["category"] == "vial"].iloc[0]
        out["stage10_summary_images"] = int(r["num_images"])
        out["stage10_summary_candidate_rows"] = int(r["num_candidate_rows"])

    if STAGE11_SUMMARY.exists():
        s11 = pd.read_csv(STAGE11_SUMMARY)
        r = s11[s11["category"] == "vial"].iloc[0]
        out["stage11_summary_images"] = int(r["num_images"])
        out["stage11_summary_candidate_rows"] = int(r["num_candidate_rows"])

    return out


def summarize(stage10: pd.DataFrame, stage11: pd.DataFrame, matched: pd.DataFrame, unmatched: pd.DataFrame) -> pd.DataFrame:
    orig = original_summary()

    row = {
        "category": "vial",
        **orig,
        "stage10_unique_source_keys": int(stage10["source_key"].nunique()),
        "stage11_unique_source_keys": int(stage11["source_key"].nunique()),
        "matched_top1_images": int(len(matched)),
        "unmatched_keys": int(len(unmatched)),
        "stage10_candidate_rows": int(len(stage10)),
        "stage11_candidate_rows": int(len(stage11)),
        "stage10_mean_candidates_per_image": float(len(stage10) / max(1, stage10["source_key"].nunique())),
        "stage11_mean_candidates_per_image": float(len(stage11) / max(1, stage11["source_key"].nunique())),
        "mean_tight_bbox_iou": float(matched["tight_bbox_iou"].mean()),
        "median_tight_bbox_iou": float(matched["tight_bbox_iou"].median()),
        "mean_context_bbox_iou": float(matched["context_bbox_iou"].mean()),
        "median_context_bbox_iou": float(matched["context_bbox_iou"].median()),
        "mean_abs_tight_area_ratio_diff": float((matched["stage11_tight_area_ratio"] - matched["stage10_tight_area_ratio"]).abs().mean()),
        "mean_abs_context_area_ratio_diff": float((matched["stage11_context_area_ratio"] - matched["stage10_context_area_ratio"]).abs().mean()),
        "stage10_mean_tight_gt_coverage": float(matched["stage10_tight_gt_coverage"].mean()),
        "stage11_mean_tight_gt_coverage": float(matched["stage11_tight_gt_coverage"].mean()),
        "stage10_mean_context_gt_coverage": float(matched["stage10_context_gt_coverage"].mean()),
        "stage11_mean_context_gt_coverage": float(matched["stage11_context_gt_coverage"].mean()),
        "mean_abs_tight_gt_coverage_diff": float((matched["stage11_tight_gt_coverage"] - matched["stage10_tight_gt_coverage"]).abs().mean()),
        "mean_abs_context_gt_coverage_diff": float((matched["stage11_context_gt_coverage"] - matched["stage10_context_gt_coverage"]).abs().mean()),
    }

    return pd.DataFrame([row])


def f4(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def write_report(summary: pd.DataFrame, matched: pd.DataFrame, unmatched: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    r = summary.iloc[0]

    lines = []

    lines.append("# Stage 11-F2 Vial Source-aligned Candidate Drift Diagnostic")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report compares Stage 10 and Stage 11 vial candidate construction using source-path alignment.")
    lines.append("Stage 11 adapter paths are mapped back to original AD2 source paths through the Stage 11-A folder adapter mapping.")
    lines.append("")
    lines.append("This step does not rerun PatchCore, VLM inference, or crop generation.")
    lines.append("")
    lines.append("## 2. Alignment Result")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Stage 10-D summary images | {r.get('stage10_summary_images', '')} |")
    lines.append(f"| Stage 11-C summary images | {r.get('stage11_summary_images', '')} |")
    lines.append(f"| Stage 10 unique source keys | {int(r['stage10_unique_source_keys'])} |")
    lines.append(f"| Stage 11 unique source keys | {int(r['stage11_unique_source_keys'])} |")
    lines.append(f"| Matched top1 images | {int(r['matched_top1_images'])} |")
    lines.append(f"| Unmatched keys | {int(r['unmatched_keys'])} |")
    lines.append("")
    lines.append("## 3. Candidate Policy Difference")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Stage 10 candidate rows | {int(r['stage10_candidate_rows'])} |")
    lines.append(f"| Stage 11 candidate rows | {int(r['stage11_candidate_rows'])} |")
    lines.append(f"| Stage 10 mean candidates/image | {f4(r['stage10_mean_candidates_per_image'])} |")
    lines.append(f"| Stage 11 mean candidates/image | {f4(r['stage11_mean_candidates_per_image'])} |")
    lines.append(f"| Mean tight bbox IoU | {f4(r['mean_tight_bbox_iou'])} |")
    lines.append(f"| Median tight bbox IoU | {f4(r['median_tight_bbox_iou'])} |")
    lines.append(f"| Mean context bbox IoU | {f4(r['mean_context_bbox_iou'])} |")
    lines.append(f"| Median context bbox IoU | {f4(r['median_context_bbox_iou'])} |")
    lines.append(f"| Mean abs tight area-ratio diff | {f4(r['mean_abs_tight_area_ratio_diff'])} |")
    lines.append(f"| Mean abs context area-ratio diff | {f4(r['mean_abs_context_area_ratio_diff'])} |")
    lines.append("")
    lines.append("## 4. GT Coverage Difference")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Stage 10 mean tight GT coverage | {f4(r['stage10_mean_tight_gt_coverage'])} |")
    lines.append(f"| Stage 11 mean tight GT coverage | {f4(r['stage11_mean_tight_gt_coverage'])} |")
    lines.append(f"| Stage 10 mean context GT coverage | {f4(r['stage10_mean_context_gt_coverage'])} |")
    lines.append(f"| Stage 11 mean context GT coverage | {f4(r['stage11_mean_context_gt_coverage'])} |")
    lines.append(f"| Mean abs tight GT coverage diff | {f4(r['mean_abs_tight_gt_coverage_diff'])} |")
    lines.append(f"| Mean abs context GT coverage diff | {f4(r['mean_abs_context_gt_coverage_diff'])} |")
    lines.append("")
    lines.append("## 5. Interpretation Rule")
    lines.append("")
    lines.append("- If matched top1 images is close to 71 and bbox IoU is high, the Stage 10/Stage 11 vial discrepancy is unlikely to be candidate selection.")
    lines.append("- If matched top1 images is close to 71 but bbox IoU or GT coverage differs strongly, Stage 11-C should be patched to reuse the Stage 10 candidate/crop policy.")
    lines.append("- If matched top1 images is still far below 71, the alignment key remains wrong and the unmatched CSV should be inspected.")
    lines.append("")
    lines.append("## 6. Current Decision")
    lines.append("")

    matched_n = int(r["matched_top1_images"])
    mean_iou = float(r["mean_tight_bbox_iou"]) if not pd.isna(r["mean_tight_bbox_iou"]) else -1.0

    if matched_n >= 68 and mean_iou >= 0.70:
        decision = "Candidate selection is mostly consistent. Next inspect CLIP backend, prompt, crop image content, or aggregation."
    elif matched_n >= 68 and mean_iou < 0.70:
        decision = "Candidate policy drift is likely. Patch Stage 11-C to reuse Stage 10 candidate/crop construction."
    else:
        decision = "Alignment is still insufficient. Inspect unmatched source keys before drawing a policy conclusion."

    lines.append(decision)
    lines.append("")
    lines.append("## 7. Output")
    lines.append("")
    lines.append(f"- Matched CSV: `{OUT_MATCHED.relative_to(ROOT)}`")
    lines.append(f"- Unmatched CSV: `{OUT_UNMATCHED.relative_to(ROOT)}`")
    lines.append(f"- Summary CSV: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    stage10 = normalize_stage10()
    stage11 = normalize_stage11()

    top10 = top1(stage10, "stage10")
    top11 = top1(stage11, "stage11")

    matched = top10.merge(top11, on="source_key", how="inner", suffixes=("_s10", "_s11"))

    # Use Stage 10 image size as reference.
    if "image_width_s10" in matched.columns:
        matched["image_width"] = matched["image_width_s10"]
    if "image_height_s10" in matched.columns:
        matched["image_height"] = matched["image_height_s10"]

    matched["tight_bbox_iou"] = matched.apply(lambda r: bbox_iou_row(r, "stage10_tight", "stage11_tight"), axis=1)
    matched["context_bbox_iou"] = matched.apply(lambda r: bbox_iou_row(r, "stage10_context", "stage11_context"), axis=1)
    matched = add_coverage(matched)

    keys10 = set(top10["source_key"].tolist())
    keys11 = set(top11["source_key"].tolist())
    unmatched_rows = []

    for k in sorted(keys10 - keys11):
        unmatched_rows.append({"source_key": k, "missing_from": "stage11"})
    for k in sorted(keys11 - keys10):
        unmatched_rows.append({"source_key": k, "missing_from": "stage10"})

    unmatched = pd.DataFrame(unmatched_rows)

    summary = summarize(stage10, stage11, matched, unmatched)

    matched.to_csv(OUT_MATCHED, index=False)
    unmatched.to_csv(OUT_UNMATCHED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(summary, matched, unmatched)

    print("[DONE]", OUT_MATCHED)
    print("[DONE]", OUT_UNMATCHED)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)

    print("\n===== summary =====")
    print(summary.to_string(index=False))

    print("\n===== unmatched =====")
    print(unmatched.head(50).to_string(index=False) if not unmatched.empty else "none")

    print("\n===== lowest tight IoU =====")
    show_cols = [
        "source_key",
        "tight_bbox_iou",
        "context_bbox_iou",
        "stage10_tight_area_ratio",
        "stage11_tight_area_ratio",
        "stage10_context_area_ratio",
        "stage11_context_area_ratio",
        "stage10_tight_gt_coverage",
        "stage11_tight_gt_coverage",
        "stage10_context_gt_coverage",
        "stage11_context_gt_coverage",
    ]
    show_cols = [c for c in show_cols if c in matched.columns]
    print(matched[show_cols].sort_values("tight_bbox_iou").head(30).to_string(index=False))


if __name__ == "__main__":
    main()
