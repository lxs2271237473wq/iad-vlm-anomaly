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

OUT_ALL_CONFIGS = OUT_DIR / "stage18_b5_ad2_loco_qcr_all_configs_per_category.csv"
OUT_SELECTED = OUT_DIR / "stage18_b5_ad2_loco_qcr_selected_folds.csv"
OUT_SUMMARY = OUT_DIR / "stage18_b5_ad2_loco_qcr_summary.csv"
OUT_REPORT = DOC_DIR / "stage18_b5_ad2_loco_qcr_policy_optimization_report.md"


ETA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
GAMMA_GRID = [0.00, 0.01, 0.03, 0.05]


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
    if len(y) == 0 or y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def aggregate_candidate_quality_sources(cand: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["category", "image_path"]

    allowed_base_cols = [
        "candidate_score_mean",
        "candidate_score_max",
        "tight_candidate_mask_density",
        "context_candidate_mask_density",
        "map_area",
    ]

    agg_spec = {}

    for col in allowed_base_cols:
        if col not in cand.columns:
            continue

        # Non-GT candidate/source statistics only.
        agg_spec[f"{col}_max"] = (col, "max")
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_min"] = (col, "min")

    if "candidate_rank" in cand.columns:
        agg_spec["num_candidates"] = ("candidate_rank", "count")

    if not agg_spec:
        raise RuntimeError("No valid candidate-quality source columns found.")

    return cand.groupby(group_cols, as_index=False).agg(**agg_spec)


def build_base() -> tuple[pd.DataFrame, list[str]]:
    img = read_csv(IN_IMAGE)
    cand = read_csv(IN_CAND)

    img = img[img["category"].isin(AD2_CATEGORIES)].copy()
    cand = cand[cand["category"].isin(AD2_CATEGORIES)].copy()

    for c in ["category", "image_path", "gt_binary", "patchcore_score"]:
        if c not in img.columns:
            raise RuntimeError(f"Missing required image-level column: {c}")

    if "context_topk_mean_score" in img.columns:
        m_col = "context_topk_mean_score"
    elif "context_topk_max_score" in img.columns:
        m_col = "context_topk_max_score"
    elif "context_top1_score" in img.columns:
        m_col = "context_top1_score"
    else:
        raise RuntimeError("No crop/context VLM score column found.")

    q = aggregate_candidate_quality_sources(cand)
    df = img.merge(q, on=["category", "image_path"], how="left", validate="one_to_one")

    df["D_raw"] = pd.to_numeric(df["patchcore_score"], errors="coerce")
    df["M_raw"] = pd.to_numeric(df[m_col], errors="coerce")

    df = norm_by_category(df, "D_raw", "D")
    df = norm_by_category(df, "M_raw", "M")

    df["V3_naive"] = 0.5 * df["D"] + 0.5 * df["M"]

    protected = {
        "category",
        "image_path",
        "gt_binary",
        "patchcore_score",
        m_col,
        "D_raw",
        "M_raw",
        "D",
        "M",
        "V3_naive",
    }

    q_cols = []
    for c in df.columns:
        if c in protected:
            continue

        lc = c.lower()

        # Hard leakage / invalid evidence filter.
        if any(t in lc for t in ["gt", "label", "target", "full_image", "context_top", "vlm", "clip"]):
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 10 and s.nunique(dropna=True) > 1:
            q_cols.append(c)

    if not q_cols:
        raise RuntimeError("No safe Q candidate columns after filtering.")

    return df, q_cols


def score_config(base: pd.DataFrame, q_source: str, q_direction: str, eta: float, gamma: float) -> pd.DataFrame:
    work = base.copy()

    work = norm_by_category(work, q_source, "Q")
    if q_direction == "inverted":
        work["Q"] = 1.0 - work["Q"]

    work["K"] = work["D"] * work["M"]
    work["agreement"] = 1.0 - (work["D"] - work["M"]).abs()
    work["mutual_anomaly_evidence"] = np.minimum(work["D"], work["M"])
    work["adaptive_gate"] = work["Q"] * work["K"] * work["agreement"] * work["mutual_anomaly_evidence"]

    # Generalized QCR policy.
    work["S_quality"] = work["V3_naive"] - eta * work["M"] * (1.0 - work["Q"])
    work["S_adaptive"] = work["S_quality"] + gamma * work["adaptive_gate"]

    rows = []

    for cat, sub in work.groupby("category"):
        au_v3 = safe_auroc(sub["gt_binary"], sub["V3_naive"])
        au_q = safe_auroc(sub["gt_binary"], sub["S_quality"])
        au_a = safe_auroc(sub["gt_binary"], sub["S_adaptive"])
        au_qalone = safe_auroc(sub["gt_binary"], sub["Q"])

        rows.append(
            {
                "category": cat,
                "q_source": q_source,
                "q_direction": q_direction,
                "eta": eta,
                "gamma": gamma,
                "auroc_V3_naive": au_v3,
                "auroc_quality_qcr": au_q,
                "auroc_adaptive_qcr": au_a,
                "auroc_Q_alone": au_qalone,
                "delta_quality_minus_V3": au_q - au_v3,
                "delta_adaptive_minus_V3": au_a - au_v3,
                "delta_adaptive_minus_quality": au_a - au_q,
            }
        )

    return pd.DataFrame(rows)


def build_all_configs(base: pd.DataFrame, q_cols: list[str]) -> pd.DataFrame:
    frames = []

    for q_source in q_cols:
        for q_direction in ["direct", "inverted"]:
            for eta in ETA_GRID:
                for gamma in GAMMA_GRID:
                    frames.append(score_config(base, q_source, q_direction, eta, gamma))

    return pd.concat(frames, ignore_index=True)


def select_loco_configs(all_configs: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []

    for heldout in AD2_CATEGORIES:
        train_cats = [c for c in AD2_CATEGORIES if c != heldout]

        train = all_configs[all_configs["category"].isin(train_cats)].copy()
        test = all_configs[all_configs["category"] == heldout].copy()

        group_cols = ["q_source", "q_direction", "eta", "gamma"]

        train_summary = (
            train.groupby(group_cols, as_index=False)
            .agg(
                train_mean_V3=("auroc_V3_naive", "mean"),
                train_mean_quality=("auroc_quality_qcr", "mean"),
                train_mean_adaptive=("auroc_adaptive_qcr", "mean"),
                train_mean_delta_quality=("delta_quality_minus_V3", "mean"),
                train_mean_delta_adaptive=("delta_adaptive_minus_V3", "mean"),
                train_wins_quality=("delta_quality_minus_V3", lambda x: int((x > 0).sum())),
                train_wins_adaptive=("delta_adaptive_minus_V3", lambda x: int((x > 0).sum())),
                train_worst_delta_adaptive=("delta_adaptive_minus_V3", "min"),
            )
        )

        # Select by adaptive mean AUROC first, then positive delta/wins/stability.
        train_summary = train_summary.sort_values(
            [
                "train_mean_adaptive",
                "train_mean_delta_adaptive",
                "train_wins_adaptive",
                "train_worst_delta_adaptive",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        best = train_summary.iloc[0]

        match = (
            (test["q_source"] == best["q_source"])
            & (test["q_direction"] == best["q_direction"])
            & (test["eta"] == best["eta"])
            & (test["gamma"] == best["gamma"])
        )

        test_row = test[match].iloc[0]

        selected_rows.append(
            {
                "heldout_category": heldout,
                "train_categories": ";".join(train_cats),
                "selected_q_source": best["q_source"],
                "selected_q_direction": best["q_direction"],
                "selected_eta": float(best["eta"]),
                "selected_gamma": float(best["gamma"]),
                "train_mean_V3": float(best["train_mean_V3"]),
                "train_mean_quality": float(best["train_mean_quality"]),
                "train_mean_adaptive": float(best["train_mean_adaptive"]),
                "train_mean_delta_quality": float(best["train_mean_delta_quality"]),
                "train_mean_delta_adaptive": float(best["train_mean_delta_adaptive"]),
                "train_wins_quality": int(best["train_wins_quality"]),
                "train_wins_adaptive": int(best["train_wins_adaptive"]),
                "test_V3": float(test_row["auroc_V3_naive"]),
                "test_quality_qcr": float(test_row["auroc_quality_qcr"]),
                "test_adaptive_qcr": float(test_row["auroc_adaptive_qcr"]),
                "test_delta_quality_minus_V3": float(test_row["delta_quality_minus_V3"]),
                "test_delta_adaptive_minus_V3": float(test_row["delta_adaptive_minus_V3"]),
                "test_delta_adaptive_minus_quality": float(test_row["delta_adaptive_minus_quality"]),
                "test_Q_alone": float(test_row["auroc_Q_alone"]),
            }
        )

    return pd.DataFrame(selected_rows)


def summarize_loco(selected: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol": "AD2_leave_one_category_out_qcr_policy_selection",
                "num_folds": len(selected),
                "mean_test_V3": float(selected["test_V3"].mean()),
                "mean_test_quality_qcr": float(selected["test_quality_qcr"].mean()),
                "mean_test_adaptive_qcr": float(selected["test_adaptive_qcr"].mean()),
                "mean_delta_quality_minus_V3": float(selected["test_delta_quality_minus_V3"].mean()),
                "mean_delta_adaptive_minus_V3": float(selected["test_delta_adaptive_minus_V3"].mean()),
                "wins_quality_over_V3": int((selected["test_delta_quality_minus_V3"] > 0).sum()),
                "wins_adaptive_over_V3": int((selected["test_delta_adaptive_minus_V3"] > 0).sum()),
                "wins_adaptive_over_quality": int((selected["test_delta_adaptive_minus_quality"] > 0).sum()),
                "worst_quality_delta": float(selected["test_delta_quality_minus_V3"].min()),
                "worst_adaptive_delta": float(selected["test_delta_adaptive_minus_V3"].min()),
                "worst_quality_category": str(
                    selected.sort_values("test_delta_quality_minus_V3").iloc[0]["heldout_category"]
                ),
                "worst_adaptive_category": str(
                    selected.sort_values("test_delta_adaptive_minus_V3").iloc[0]["heldout_category"]
                ),
            }
        ]
    )


def fmt(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.4f}"


def signed(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):+.4f}"


def write_report(q_cols: list[str], selected: pd.DataFrame, summary: pd.DataFrame) -> None:
    s = summary.iloc[0]

    if s["mean_delta_adaptive_minus_V3"] > 0 and s["wins_adaptive_over_V3"] >= 3:
        final_status = "promote_qcr_as_cross_category_calibrated_ad2_support"
    elif s["mean_delta_quality_minus_V3"] > 0 and s["wins_quality_over_V3"] >= 3:
        final_status = "promote_quality_qcr_without_adaptive_as_ad2_support"
    elif s["mean_delta_adaptive_minus_V3"] > 0:
        final_status = "weak_positive_mean_but_not_category_stable"
    else:
        final_status = "do_not_promote_ad2_qcr_main_claim"

    lines = [
        "# Stage 18-B5 AD2 LOCO QCR Policy Optimization",
        "",
        "## Purpose",
        "",
        "Optimize QCR policy without using the held-out AD2 category labels for selection.",
        "",
        "Each fold selects Q source, Q direction, eta, and gamma on three AD2 categories, then evaluates on the held-out category.",
        "",
        "## Safe Q source candidates",
        "",
        "```text",
        *q_cols,
        "```",
        "",
        "## Summary",
        "",
        f"- final_status: `{final_status}`",
        f"- mean test V3 naive: `{fmt(s['mean_test_V3'])}`",
        f"- mean test quality QCR: `{fmt(s['mean_test_quality_qcr'])}`",
        f"- mean test adaptive QCR: `{fmt(s['mean_test_adaptive_qcr'])}`",
        f"- quality QCR minus V3: `{signed(s['mean_delta_quality_minus_V3'])}`",
        f"- adaptive QCR minus V3: `{signed(s['mean_delta_adaptive_minus_V3'])}`",
        f"- quality QCR wins over V3: `{int(s['wins_quality_over_V3'])}/4`",
        f"- adaptive QCR wins over V3: `{int(s['wins_adaptive_over_V3'])}/4`",
        f"- worst adaptive category: `{s['worst_adaptive_category']}`",
        f"- worst adaptive delta: `{signed(s['worst_adaptive_delta'])}`",
        "",
        "## Selected folds",
        "",
        "| Held-out | Selected Q | Direction | eta | gamma | Test V3 | Test Quality | Test Adaptive | Adaptive-V3 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in selected.iterrows():
        lines.append(
            f"| {r['heldout_category']} | {r['selected_q_source']} | {r['selected_q_direction']} | "
            f"{r['selected_eta']:.2f} | {r['selected_gamma']:.2f} | "
            f"{fmt(r['test_V3'])} | {fmt(r['test_quality_qcr'])} | "
            f"{fmt(r['test_adaptive_qcr'])} | {signed(r['test_delta_adaptive_minus_V3'])} |"
        )

    lines += [
        "",
        "## Decision rule",
        "",
        "- If adaptive QCR has positive mean delta and wins at least 3/4 held-out categories, QCR can be promoted as cross-category calibrated AD2 support.",
        "- If only mean delta is positive but wins fewer than 3/4, report AD2 as weak/boundary support.",
        "- If mean delta is negative, keep AD2 QCR as source-sensitivity diagnostic and retain VisA as the main ablation.",
        "",
        "## Outputs",
        "",
        f"- `{OUT_ALL_CONFIGS.relative_to(ROOT)}`",
        f"- `{OUT_SELECTED.relative_to(ROOT)}`",
        f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    base, q_cols = build_base()
    all_configs = build_all_configs(base, q_cols)
    selected = select_loco_configs(all_configs)
    summary = summarize_loco(selected)

    all_configs.to_csv(OUT_ALL_CONFIGS, index=False, lineterminator="\n")
    selected.to_csv(OUT_SELECTED, index=False, lineterminator="\n")
    summary.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")

    write_report(q_cols, selected, summary)

    print("[DONE]", OUT_ALL_CONFIGS)
    print("[DONE]", OUT_SELECTED)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== selected folds =====")
    print(selected.to_string(index=False))
    print()
    print("===== summary =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
