from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(".").resolve()

AD2_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

IN_IMAGE = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv"
IN_CAND = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv"

OUT_DIR = ROOT / "results/stage18_ad2_qcr_ablation"
DOC_DIR = ROOT / "docs/stage18_ad2_qcr_ablation"

OUT_PER_CATEGORY = OUT_DIR / "stage18_b3_ad2_q_source_sweep_per_category.csv"
OUT_SUMMARY = OUT_DIR / "stage18_b3_ad2_q_source_sweep_summary.csv"
OUT_RANKED = OUT_DIR / "stage18_b3_ad2_q_source_sweep_ranked.csv"
OUT_REPORT = DOC_DIR / "stage18_b3_ad2_q_source_sweep_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"Bad CSV format: {path}")
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
    for _, idx in df.groupby("category").groups.items():
        df.loc[idx, out_col] = minmax_norm(df.loc[idx, col])
    return df


def safe_auroc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    ok = y.notna() & s.notna()
    y = y[ok].astype(int)
    s = s[ok]
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def aggregate_candidate_sources(cand: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["category", "image_path"]

    possible_cols = [
        "candidate_score_mean",
        "candidate_score_max",
        "tight_candidate_mask_density",
        "context_candidate_mask_density",
        "map_area",
    ]

    agg_spec = {}

    for col in possible_cols:
        if col not in cand.columns:
            continue
        agg_spec[f"{col}_max"] = (col, "max")
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_min"] = (col, "min")

    if "candidate_rank" in cand.columns:
        agg_spec["num_candidates"] = ("candidate_rank", "count")

    if not agg_spec:
        raise RuntimeError("No candidate source columns found.")

    return cand.groupby(group_cols, as_index=False).agg(**agg_spec)


def build_base() -> pd.DataFrame:
    img = read_csv(IN_IMAGE)
    cand = read_csv(IN_CAND)

    img = img[img["category"].isin(AD2_CATEGORIES)].copy()
    cand = cand[cand["category"].isin(AD2_CATEGORIES)].copy()

    for c in ["category", "image_path", "gt_binary", "patchcore_score"]:
        if c not in img.columns:
            raise RuntimeError(f"Missing required image column: {c}")

    if "context_topk_mean_score" in img.columns:
        m_col = "context_topk_mean_score"
    elif "context_topk_max_score" in img.columns:
        m_col = "context_topk_max_score"
    elif "context_top1_score" in img.columns:
        m_col = "context_top1_score"
    else:
        raise RuntimeError("Missing crop/context VLM score column.")

    q = aggregate_candidate_sources(cand)

    df = img.merge(q, on=["category", "image_path"], how="left", validate="one_to_one")

    df["D_raw"] = pd.to_numeric(df["patchcore_score"], errors="coerce")
    df["M_raw"] = pd.to_numeric(df[m_col], errors="coerce")

    df = norm_by_category(df, "D_raw", "D")
    df = norm_by_category(df, "M_raw", "M")

    df["V3_naive"] = 0.5 * df["D"] + 0.5 * df["M"]

    return df


def q_candidate_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "category",
        "image_path",
        "gt_binary",
        "patchcore_score",
        "D_raw",
        "M_raw",
        "D",
        "M",
        "V3_naive",
    }

    cols = []
    for c in df.columns:
        if c in exclude:
            continue

        lc = c.lower()

        # Hard anti-leakage filter.
        if any(x in lc for x in ["gt", "label", "target", "anomaly_binary", "is_anomaly"]):
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 10 and s.nunique(dropna=True) > 1:
            cols.append(c)

    return cols


def evaluate_one(df: pd.DataFrame, q_raw_col: str, invert_q: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()

    q_name = q_raw_col + ("__inverted" if invert_q else "__direct")

    work = norm_by_category(work, q_raw_col, "Q")
    if invert_q:
        work["Q"] = 1.0 - work["Q"]

    work["K"] = work["D"] * work["M"]
    work["agreement"] = 1.0 - (work["D"] - work["M"]).abs()
    work["mutual_anomaly_evidence"] = np.minimum(work["D"], work["M"])
    work["adaptive_gate"] = work["Q"] * work["K"] * work["agreement"] * work["mutual_anomaly_evidence"]

    # Keep the same formulas as the current paper method.
    work["V4_quality"] = 0.5 * work["D"] + 0.5 * work["M"] * (0.5 + 0.5 * work["Q"])
    work["V5_fixed_qc"] = 0.4 * work["D"] + 0.4 * work["M"] + 0.1 * work["Q"] + 0.1 * work["K"]
    work["V6_adaptive"] = work["V4_quality"] + 0.05 * work["adaptive_gate"]

    rows = []

    for cat, sub in work.groupby("category"):
        y = sub["gt_binary"]

        au_v3 = safe_auroc(y, sub["V3_naive"])
        au_v4 = safe_auroc(y, sub["V4_quality"])
        au_v5 = safe_auroc(y, sub["V5_fixed_qc"])
        au_v6 = safe_auroc(y, sub["V6_adaptive"])
        au_d = safe_auroc(y, sub["D"])
        au_m = safe_auroc(y, sub["M"])
        au_q = safe_auroc(y, sub["Q"])

        rows.append(
            {
                "q_source": q_raw_col,
                "q_direction": "inverted" if invert_q else "direct",
                "q_name": q_name,
                "category": cat,
                "num_images": len(sub),
                "auroc_detector_D": au_d,
                "auroc_crop_M": au_m,
                "auroc_quality_Q_alone": au_q,
                "auroc_V3_naive": au_v3,
                "auroc_V4_quality": au_v4,
                "auroc_V5_fixed_qc": au_v5,
                "auroc_V6_adaptive": au_v6,
                "delta_V4_minus_V3": au_v4 - au_v3,
                "delta_V6_minus_V4": au_v6 - au_v4,
                "delta_V6_minus_V3": au_v6 - au_v3,
            }
        )

    per_cat = pd.DataFrame(rows)

    summary = {
        "q_source": q_raw_col,
        "q_direction": "inverted" if invert_q else "direct",
        "q_name": q_name,
        "num_categories": int(per_cat["category"].nunique()),
        "mean_auroc_detector_D": float(per_cat["auroc_detector_D"].mean()),
        "mean_auroc_crop_M": float(per_cat["auroc_crop_M"].mean()),
        "mean_auroc_quality_Q_alone": float(per_cat["auroc_quality_Q_alone"].mean()),
        "mean_auroc_V3_naive": float(per_cat["auroc_V3_naive"].mean()),
        "mean_auroc_V4_quality": float(per_cat["auroc_V4_quality"].mean()),
        "mean_auroc_V5_fixed_qc": float(per_cat["auroc_V5_fixed_qc"].mean()),
        "mean_auroc_V6_adaptive": float(per_cat["auroc_V6_adaptive"].mean()),
        "mean_delta_V4_minus_V3": float(per_cat["delta_V4_minus_V3"].mean()),
        "mean_delta_V6_minus_V4": float(per_cat["delta_V6_minus_V4"].mean()),
        "mean_delta_V6_minus_V3": float(per_cat["delta_V6_minus_V3"].mean()),
        "wins_V4_over_V3": int((per_cat["delta_V4_minus_V3"] > 0).sum()),
        "wins_V6_over_V4": int((per_cat["delta_V6_minus_V4"] > 0).sum()),
        "wins_V6_over_V3": int((per_cat["delta_V6_minus_V3"] > 0).sum()),
        "worst_category_delta_V4_minus_V3": float(per_cat["delta_V4_minus_V3"].min()),
        "worst_category": str(per_cat.sort_values("delta_V4_minus_V3").iloc[0]["category"]),
    }

    return per_cat, pd.DataFrame([summary])


def write_report(per_cat: pd.DataFrame, summary: pd.DataFrame, ranked: pd.DataFrame) -> None:
    best = ranked.iloc[0]

    lines = [
        "# Stage 18-B3 AD2 Q Source Sweep",
        "",
        "## Purpose",
        "",
        "Diagnose whether the Stage 18-B2 AD2 QCR drop against naive fusion is caused by the selected candidate quality source.",
        "",
        "The sweep keeps the same QCR formulas as the current paper method and only changes the non-GT candidate quality source.",
        "",
        "## Best ranked source",
        "",
        f"- q_source: `{best['q_source']}`",
        f"- q_direction: `{best['q_direction']}`",
        f"- mean V3 AUROC: `{best['mean_auroc_V3_naive']:.4f}`",
        f"- mean V4 AUROC: `{best['mean_auroc_V4_quality']:.4f}`",
        f"- mean V6 AUROC: `{best['mean_auroc_V6_adaptive']:.4f}`",
        f"- V4 minus V3: `{best['mean_delta_V4_minus_V3']:+.4f}`",
        f"- V6 minus V3: `{best['mean_delta_V6_minus_V3']:+.4f}`",
        f"- V4 wins over V3: `{int(best['wins_V4_over_V3'])}/4`",
        f"- worst category: `{best['worst_category']}`",
        f"- worst category delta V4-V3: `{best['worst_category_delta_V4_minus_V3']:+.4f}`",
        "",
        "## Top 10 sources",
        "",
        "| Rank | Q source | Direction | V3 | V4 | V6 | V4-V3 | V6-V3 | Wins V4/V3 | Worst category |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for i, (_, r) in enumerate(ranked.head(10).iterrows(), start=1):
        lines.append(
            f"| {i} | {r['q_source']} | {r['q_direction']} | "
            f"{r['mean_auroc_V3_naive']:.4f} | {r['mean_auroc_V4_quality']:.4f} | "
            f"{r['mean_auroc_V6_adaptive']:.4f} | {r['mean_delta_V4_minus_V3']:+.4f} | "
            f"{r['mean_delta_V6_minus_V3']:+.4f} | {int(r['wins_V4_over_V3'])}/4 | "
            f"{r['worst_category']} |"
        )

    lines += [
        "",
        "## Decision rule",
        "",
        "- If a non-GT Q source gives V4 > V3 on mean AUROC and wins at least 3/4 categories, AD2 QCR can be promoted to a stronger supporting ablation.",
        "- If no Q source passes that threshold, AD2 QCR should be reported as a boundary/diagnostic result rather than a main claim.",
        "- Do not select a Q source using ground-truth overlap, ground-truth mask quality, or label-derived information.",
        "",
        "## Outputs",
        "",
        f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
        f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
        f"- `{OUT_RANKED.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    base = build_base()
    q_cols = q_candidate_columns(base)

    if not q_cols:
        raise RuntimeError("No valid non-GT Q candidate columns found.")

    all_per_cat = []
    all_summary = []

    for q_col in q_cols:
        for invert in [False, True]:
            per_cat, summary = evaluate_one(base, q_col, invert)
            all_per_cat.append(per_cat)
            all_summary.append(summary)

    per_cat_all = pd.concat(all_per_cat, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)

    ranked = summary_all.sort_values(
        [
            "mean_delta_V4_minus_V3",
            "wins_V4_over_V3",
            "mean_auroc_V4_quality",
            "worst_category_delta_V4_minus_V3",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    per_cat_all.to_csv(OUT_PER_CATEGORY, index=False, lineterminator="\n")
    summary_all.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")
    ranked.to_csv(OUT_RANKED, index=False, lineterminator="\n")

    write_report(per_cat_all, summary_all, ranked)

    print("[DONE]", OUT_PER_CATEGORY)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_RANKED)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== top ranked Q sources =====")
    cols = [
        "q_source",
        "q_direction",
        "mean_auroc_V3_naive",
        "mean_auroc_V4_quality",
        "mean_auroc_V6_adaptive",
        "mean_delta_V4_minus_V3",
        "mean_delta_V6_minus_V3",
        "wins_V4_over_V3",
        "worst_category",
        "worst_category_delta_V4_minus_V3",
    ]
    print(ranked[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
