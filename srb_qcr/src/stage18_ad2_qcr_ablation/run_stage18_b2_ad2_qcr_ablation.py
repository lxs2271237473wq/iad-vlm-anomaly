from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score


ROOT = Path(".").resolve()

AD2_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

IN_IMAGE = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv"
IN_CAND = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv"

OUT_DIR = ROOT / "results/stage18_ad2_qcr_ablation"
DOC_DIR = ROOT / "docs/stage18_ad2_qcr_ablation"

OUT_PRED = OUT_DIR / "stage18_b2_ad2_qcr_assembled_predictions.csv"
OUT_PER_CATEGORY = OUT_DIR / "stage18_b2_ad2_qcr_ablation_per_category.csv"
OUT_SUMMARY = OUT_DIR / "stage18_b2_ad2_qcr_ablation_summary.csv"
OUT_DELTAS = OUT_DIR / "stage18_b2_ad2_qcr_claim_ready_deltas.csv"
OUT_REPORT = DOC_DIR / "stage18_b2_ad2_qcr_ablation_report.md"


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def minmax_norm(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mn = x.min()
    mx = x.max()
    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.full(len(x), 0.5), index=s.index)
    return (x - mn) / (mx - mn)


def norm_by_category(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    df[out_col] = np.nan
    for cat, sub_idx in df.groupby("category").groups.items():
        df.loc[sub_idx, out_col] = minmax_norm(df.loc[sub_idx, col])
    return df


def best_f1_and_acc(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    thresholds = np.unique(score)
    if len(thresholds) > 512:
        thresholds = np.quantile(score, np.linspace(0, 1, 512))

    best_f1 = -1.0
    best_acc = -1.0
    best_thr = 0.5

    for thr in thresholds:
        pred = (score >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_acc = float(acc)
            best_thr = float(thr)

    return best_f1, best_acc, best_thr


def safe_auroc(y_true: pd.Series, score: pd.Series):
    y = pd.to_numeric(y_true, errors="coerce").astype(int)
    s = pd.to_numeric(score, errors="coerce")
    ok = y.notna() & s.notna()
    y = y[ok]
    s = s[ok]
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_ap(y_true: pd.Series, score: pd.Series):
    y = pd.to_numeric(y_true, errors="coerce").astype(int)
    s = pd.to_numeric(score, errors="coerce")
    ok = y.notna() & s.notna()
    y = y[ok]
    s = s[ok]
    if y.nunique() < 2:
        return np.nan
    return float(average_precision_score(y, s))


def build_predictions() -> pd.DataFrame:
    img = read_csv_strict(IN_IMAGE)
    cand = read_csv_strict(IN_CAND)

    img = img[img["category"].isin(AD2_CATEGORIES)].copy()
    cand = cand[cand["category"].isin(AD2_CATEGORIES)].copy()

    required_img = ["category", "image_path", "gt_binary", "patchcore_score"]
    for c in required_img:
        if c not in img.columns:
            raise RuntimeError(f"Missing required image-level column: {c}")

    # Aggregate candidate quality per image. Do not use GT coverage columns as quality.
    q_cols = [
        c for c in [
            "candidate_score_max",
            "candidate_score_mean",
            "tight_candidate_mask_density",
            "context_candidate_mask_density",
            "map_area",
        ]
        if c in cand.columns
    ]

    if not q_cols:
        raise RuntimeError("No candidate quality/source columns found in candidate score file.")

    agg_dict = {}
    if "candidate_score_mean" in cand.columns:
        agg_dict["candidate_score_mean_max"] = ("candidate_score_mean", "max")
        agg_dict["candidate_score_mean_mean"] = ("candidate_score_mean", "mean")
    if "candidate_score_max" in cand.columns:
        agg_dict["candidate_score_max_max"] = ("candidate_score_max", "max")
    if "tight_candidate_mask_density" in cand.columns:
        agg_dict["tight_candidate_mask_density_max"] = ("tight_candidate_mask_density", "max")
    if "context_candidate_mask_density" in cand.columns:
        agg_dict["context_candidate_mask_density_max"] = ("context_candidate_mask_density", "max")
    if "candidate_rank" in cand.columns:
        agg_dict["num_candidates"] = ("candidate_rank", "count")

    q = cand.groupby(["category", "image_path"], as_index=False).agg(**agg_dict)

    df = img.merge(q, on=["category", "image_path"], how="left", validate="one_to_one")

    # Raw evidence.
    df["D_raw_patchcore"] = pd.to_numeric(df["patchcore_score"], errors="coerce")

    if "context_topk_mean_score" in df.columns:
        df["M_raw_crop_topk"] = pd.to_numeric(df["context_topk_mean_score"], errors="coerce")
        m_source = "context_topk_mean_score"
    elif "context_topk_max_score" in df.columns:
        df["M_raw_crop_topk"] = pd.to_numeric(df["context_topk_max_score"], errors="coerce")
        m_source = "context_topk_max_score"
    elif "context_top1_score" in df.columns:
        df["M_raw_crop_topk"] = pd.to_numeric(df["context_top1_score"], errors="coerce")
        m_source = "context_top1_score"
    else:
        raise RuntimeError("No context crop VLM score column found in image-level predictions.")

    if "full_image_score" in df.columns:
        df["F_raw_full_image_vlm"] = pd.to_numeric(df["full_image_score"], errors="coerce")
        full_source = "full_image_score"
    elif "full_image_anomaly_score" in df.columns:
        df["F_raw_full_image_vlm"] = pd.to_numeric(df["full_image_anomaly_score"], errors="coerce")
        full_source = "full_image_anomaly_score"
    else:
        df["F_raw_full_image_vlm"] = np.nan
        full_source = "missing"

    if "candidate_score_mean_max" in df.columns:
        df["Q_raw_candidate_quality"] = pd.to_numeric(df["candidate_score_mean_max"], errors="coerce")
        q_source = "max(candidate_score_mean)"
    elif "candidate_score_max_max" in df.columns:
        df["Q_raw_candidate_quality"] = pd.to_numeric(df["candidate_score_max_max"], errors="coerce")
        q_source = "max(candidate_score_max)"
    else:
        raise RuntimeError("No usable non-GT candidate quality column found after aggregation.")

    # Per-category normalization to avoid cross-category scale leakage.
    df = norm_by_category(df, "D_raw_patchcore", "D")
    df = norm_by_category(df, "M_raw_crop_topk", "M")
    df = norm_by_category(df, "Q_raw_candidate_quality", "Q")

    if df["F_raw_full_image_vlm"].notna().any():
        df = norm_by_category(df, "F_raw_full_image_vlm", "F")
    else:
        df["F"] = np.nan

    df["K"] = df["D"] * df["M"]
    df["agreement"] = 1.0 - (df["D"] - df["M"]).abs()
    df["mutual_anomaly_evidence"] = np.minimum(df["D"], df["M"])
    df["adaptive_gate"] = df["Q"] * df["K"] * df["agreement"] * df["mutual_anomaly_evidence"]

    # Variants.
    df["V0_detector_only"] = df["D"]
    df["V1_full_image_vlm"] = df["F"]
    df["V2_crop_topk_vlm"] = df["M"]
    df["V3_naive_detector_crop_fusion"] = 0.5 * df["D"] + 0.5 * df["M"]
    df["V4_quality_calibrated_qcr"] = 0.5 * df["D"] + 0.5 * df["M"] * (0.5 + 0.5 * df["Q"])
    df["V5_fixed_qc_diagnostic"] = 0.4 * df["D"] + 0.4 * df["M"] + 0.1 * df["Q"] + 0.1 * df["K"]
    df["V6_adaptive_qcr_refinement"] = df["V4_quality_calibrated_qcr"] + 0.05 * df["adaptive_gate"]

    df["stage18_m_score_source"] = m_source
    df["stage18_q_score_source"] = q_source
    df["stage18_full_image_source"] = full_source
    df["stage18_note"] = (
        "AD2 QCR ablation assembled from Stage11 image-level VLM predictions and "
        "candidate-level non-GT quality evidence."
    )

    keep_cols = [
        "category",
        "image_path",
        "gt_binary",
        "D_raw_patchcore",
        "M_raw_crop_topk",
        "Q_raw_candidate_quality",
        "F_raw_full_image_vlm",
        "D",
        "M",
        "Q",
        "F",
        "K",
        "agreement",
        "mutual_anomaly_evidence",
        "adaptive_gate",
        "V0_detector_only",
        "V1_full_image_vlm",
        "V2_crop_topk_vlm",
        "V3_naive_detector_crop_fusion",
        "V4_quality_calibrated_qcr",
        "V5_fixed_qc_diagnostic",
        "V6_adaptive_qcr_refinement",
        "num_candidates",
        "stage18_m_score_source",
        "stage18_q_score_source",
        "stage18_full_image_source",
        "stage18_note",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def evaluate(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variants = [
        ("V0", "Detector only", "V0_detector_only", "baseline_detector"),
        ("V1", "Full-image VLM", "V1_full_image_vlm", "baseline_vlm"),
        ("V2", "Crop top-k VLM", "V2_crop_topk_vlm", "baseline_crop_vlm"),
        ("V3", "Naive detector-crop fusion", "V3_naive_detector_crop_fusion", "fusion_baseline"),
        ("V4", "Quality-Calibrated QCR", "V4_quality_calibrated_qcr", "main_method_core"),
        ("V5", "Fixed Q+C fusion", "V5_fixed_qc_diagnostic", "diagnostic_not_final"),
        ("V6", "Quality-Calibrated QCR + adaptive refinement", "V6_adaptive_qcr_refinement", "final_refinement"),
    ]

    rows = []

    for cat, sub in pred.groupby("category"):
        y = pd.to_numeric(sub["gt_binary"], errors="coerce").astype(int)

        for vid, method, col, role in variants:
            if col not in sub.columns:
                continue

            score = pd.to_numeric(sub[col], errors="coerce")
            if score.notna().sum() == 0:
                continue

            auroc = safe_auroc(y, score)
            ap = safe_ap(y, score)

            ok = y.notna() & score.notna()
            if ok.sum() > 0 and y[ok].nunique() >= 2:
                best_f1, best_acc, best_thr = best_f1_and_acc(y[ok].values, score[ok].values)
            else:
                best_f1, best_acc, best_thr = np.nan, np.nan, np.nan

            rows.append(
                {
                    "category": cat,
                    "variant_id": vid,
                    "method": method,
                    "score_col": col,
                    "paper_role": role,
                    "num_images": int(ok.sum()),
                    "num_normal": int((y[ok] == 0).sum()),
                    "num_anomaly": int((y[ok] == 1).sum()),
                    "image_auroc": auroc,
                    "average_precision": ap,
                    "best_f1": best_f1,
                    "best_accuracy": best_acc,
                    "best_threshold": best_thr,
                }
            )

    per_cat = pd.DataFrame(rows)

    summary_rows = []
    for vid, method, col, role in variants:
        sub = per_cat[per_cat["variant_id"] == vid].copy()
        if sub.empty:
            continue

        summary_rows.append(
            {
                "variant_id": vid,
                "method": method,
                "score_col": col,
                "paper_role": role,
                "num_categories": int(sub["category"].nunique()),
                "mean_image_auroc": float(sub["image_auroc"].mean()),
                "std_image_auroc": float(sub["image_auroc"].std(ddof=0)),
                "mean_average_precision": float(sub["average_precision"].mean()),
                "mean_best_f1": float(sub["best_f1"].mean()),
                "mean_best_accuracy": float(sub["best_accuracy"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)

    delta_pairs = [
        ("V4", "V3", "Quality-Calibrated QCR vs naive fusion"),
        ("V6", "V4", "Adaptive refinement vs Quality-Calibrated QCR"),
        ("V6", "V3", "Adaptive refinement vs naive fusion"),
        ("V5", "V4", "Fixed Q+C diagnostic vs Quality-Calibrated QCR"),
        ("V4", "V0", "Quality-Calibrated QCR vs detector only"),
        ("V4", "V2", "Quality-Calibrated QCR vs crop top-k VLM"),
    ]

    delta_rows = []
    for a, b, name in delta_pairs:
        ra = summary[summary["variant_id"] == a]
        rb = summary[summary["variant_id"] == b]
        if ra.empty or rb.empty:
            continue

        delta_rows.append(
            {
                "comparison": name,
                "variant_a": a,
                "variant_b": b,
                "mean_image_auroc_a": float(ra.iloc[0]["mean_image_auroc"]),
                "mean_image_auroc_b": float(rb.iloc[0]["mean_image_auroc"]),
                "delta_a_minus_b": float(ra.iloc[0]["mean_image_auroc"] - rb.iloc[0]["mean_image_auroc"]),
            }
        )

    deltas = pd.DataFrame(delta_rows)

    return per_cat, summary, deltas


def fmt(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.4f}"


def signed(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):+.4f}"


def write_report(pred: pd.DataFrame, per_cat: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    lines = [
        "# Stage 18-B2 AD2 Four-category QCR Ablation",
        "",
        "## Purpose",
        "",
        "Assemble AD2 four-category QCR ablation from existing Stage11 image-level VLM predictions and candidate-level quality evidence.",
        "",
        "This aligns the QCR ablation with the AD2 four-category system-level baseline setting.",
        "",
        "## Data",
        "",
        f"- input image-level predictions: `{IN_IMAGE.relative_to(ROOT)}`",
        f"- input candidate scores: `{IN_CAND.relative_to(ROOT)}`",
        f"- assembled images: `{len(pred)}`",
        f"- categories: `{'; '.join(sorted(pred['category'].unique()))}`",
        "- detector evidence `D`: normalized `patchcore_score`",
        "- crop VLM evidence `M`: normalized context top-k VLM score",
        "- candidate quality `Q`: normalized non-GT candidate score evidence",
        "- consistency `K`: soft high-high consistency `D*M`",
        "",
        "## Summary table",
        "",
        "| Variant | Method | Role | Mean AUROC | Mean F1 |",
        "|---|---|---|---:|---:|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['variant_id']} | {r['method']} | {r['paper_role']} | "
            f"{fmt(r['mean_image_auroc'])} | {fmt(r['mean_best_f1'])} |"
        )

    lines += [
        "",
        "## Claim-ready deltas",
        "",
        "| Comparison | Delta AUROC | A | B |",
        "|---|---:|---:|---:|",
    ]

    for _, r in deltas.iterrows():
        lines.append(
            f"| {r['comparison']} | {signed(r['delta_a_minus_b'])} | "
            f"{fmt(r['mean_image_auroc_a'])} | {fmt(r['mean_image_auroc_b'])} |"
        )

    lines += [
        "",
        "## Per-category AUROC",
        "",
        "| Category | Variant | Method | AUROC | F1 |",
        "|---|---|---|---:|---:|",
    ]

    for _, r in per_cat.iterrows():
        lines.append(
            f"| {r['category']} | {r['variant_id']} | {r['method']} | "
            f"{fmt(r['image_auroc'])} | {fmt(r['best_f1'])} |"
        )

    lines += [
        "",
        "## Interpretation rules",
        "",
        "- If V4 improves over V3, AD2 supports candidate quality calibration.",
        "- If V6 only slightly improves over V4, keep adaptive consistency as refinement.",
        "- If V5 is strong but unstable or not selected, keep fixed Q+C as diagnostic.",
        "- Do not use this table to claim pixel-level segmentation SOTA.",
        "",
        "## Outputs",
        "",
        f"- `{OUT_PRED.relative_to(ROOT)}`",
        f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
        f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
        f"- `{OUT_DELTAS.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    pred = build_predictions()
    pred.to_csv(OUT_PRED, index=False, lineterminator="\n")

    per_cat, summary, deltas = evaluate(pred)

    per_cat.to_csv(OUT_PER_CATEGORY, index=False, lineterminator="\n")
    summary.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")
    deltas.to_csv(OUT_DELTAS, index=False, lineterminator="\n")

    write_report(pred, per_cat, summary, deltas)

    print("[DONE]", OUT_PRED)
    print("[DONE]", OUT_PER_CATEGORY)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_DELTAS)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== summary =====")
    print(summary.to_string(index=False))
    print()
    print("===== deltas =====")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
