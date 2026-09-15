from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_PER_CONFIG = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_config.csv"
OUT_PER_CATEGORY = OUT_DIR / "stage16_b_adaptive_qcru_all_variants_per_category.csv"
OUT_PRIMARY = OUT_DIR / "stage16_b_adaptive_qcru_primary_protocol_table.csv"
OUT_DECISION = OUT_DIR / "stage16_b_adaptive_qcru_final_method_decision.csv"
OUT_REPORT = DOC_DIR / "stage16_b_adaptive_qcru_paper_facing_comparison_report.md"


REQUIRED_COLUMNS = [
    "backbone",
    "dataset",
    "category",
    "strategy",
    "eval_mode",
    "image_key",
    "is_anomaly_final",
    "vlm_score_norm",
    "detector_score_norm",
    "candidate_quality_norm",
    "high_high_consistency",
]


VARIANTS = [
    ("V0", "detector_only", "score_detector_only"),
    ("V2", "crop_topk_vlm", "score_crop_vlm"),
    ("V3", "naive_detector_crop_fusion", "score_naive"),
    ("V4", "quality_weighted_crop", "score_quality"),
    ("V5", "fixed_quality_consistency", "score_fixed_qc"),
    ("V6", "adaptive_qcru", "score_adaptive_qcru"),
]


def read_csv_strict(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
    return df


def minmax_safe(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    lo = x.min()
    hi = x.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)

    i = 0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    positives = y.sum()
    if positives <= 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / positives)


def best_f1_acc_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    thresholds = np.unique(y_score)
    if len(thresholds) > 1000:
        thresholds = np.unique(np.quantile(y_score, np.linspace(0, 1, 1000)))

    best_f1 = -1.0
    best_acc = -1.0
    best_thr = float("nan")

    for thr in thresholds:
        pred = (y_score >= thr).astype(int)

        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        acc = float((pred == y_true).mean())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if f1 > best_f1:
            best_f1 = float(f1)
            best_acc = float(acc)
            best_thr = float(thr)

    return best_f1, best_acc, best_thr


def eval_binary(y_true: pd.Series, y_score: pd.Series) -> dict:
    y = pd.to_numeric(y_true, errors="coerce").astype(int).to_numpy()
    s = pd.to_numeric(y_score, errors="coerce").astype(float).to_numpy()

    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]

    num_anomaly = int(y.sum())
    num_normal = int(len(y) - num_anomaly)
    has_both = num_anomaly > 0 and num_normal > 0

    if not has_both:
        auroc = float("nan")
        ap = float("nan")
    else:
        auroc = roc_auc_score_np(y, s)
        ap = average_precision_score_np(y, s)

    best_f1, best_acc, best_thr = best_f1_acc_threshold(y, s)

    return {
        "num_images": int(len(y)),
        "num_normal": num_normal,
        "num_anomaly": num_anomaly,
        "auroc": auroc,
        "ap": ap,
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "best_threshold": best_thr,
    }


def build_base_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    base_cols = [
        "backbone",
        "dataset",
        "category",
        "strategy",
        "eval_mode",
        "image_key",
        "is_anomaly_final",
        "fallback",
        "has_candidate",
        "num_candidates",
        "vlm_score_norm",
        "detector_score_norm",
        "candidate_quality_norm",
        "high_high_consistency",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    base = df[base_cols].copy()
    base = base.drop_duplicates(
        subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
    ).reset_index(drop=True)

    for c in [
        "is_anomaly_final",
        "vlm_score_norm",
        "detector_score_norm",
        "candidate_quality_norm",
        "high_high_consistency",
        "num_candidates",
    ]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    base["D"] = base["detector_score_norm"].fillna(0.0)
    base["M"] = base["vlm_score_norm"].fillna(0.0)
    base["Q"] = base["candidate_quality_norm"].fillna(0.0)
    base["K"] = base["high_high_consistency"].fillna(0.0)

    base["score_detector_only"] = base["D"]
    base["score_crop_vlm"] = base["M"]
    base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]

    base["score_quality_raw"] = (
        0.5 * base["D"]
        + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
    )

    base["score_fixed_qc_raw"] = (
        0.40 * base["D"]
        + 0.40 * base["M"]
        + 0.10 * base["Q"]
        + 0.10 * base["K"]
    )

    agreement = (1.0 - (base["D"] - base["M"]).abs()).clip(lower=0.0, upper=1.0)
    mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
    adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence

    base["adaptive_gate"] = adaptive_gate
    base["score_adaptive_qcru_raw"] = base["score_quality_raw"] + 0.05 * adaptive_gate

    group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
    for raw_col, out_col in [
        ("score_quality_raw", "score_quality"),
        ("score_fixed_qc_raw", "score_fixed_qc"),
        ("score_adaptive_qcru_raw", "score_adaptive_qcru"),
    ]:
        base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)

    return base


def make_long(base: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "backbone",
        "dataset",
        "category",
        "strategy",
        "eval_mode",
        "image_key",
        "is_anomaly_final",
    ]

    rows = []
    for vid, variant, score_col in VARIANTS:
        tmp = base[id_cols].copy()
        tmp["variant_id"] = vid
        tmp["variant"] = variant
        tmp["score"] = base[score_col].astype(float)
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def summarize(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_cols = ["backbone", "dataset", "strategy", "eval_mode", "variant_id", "variant"]
    cat_cols = ["backbone", "dataset", "strategy", "eval_mode", "category", "variant_id", "variant"]

    per_config_rows = []
    for keys, g in long_df.groupby(config_cols, dropna=False):
        row = dict(zip(config_cols, keys))
        row.update(eval_binary(g["is_anomaly_final"], g["score"]))
        per_config_rows.append(row)

    per_category_rows = []
    for keys, g in long_df.groupby(cat_cols, dropna=False):
        row = dict(zip(cat_cols, keys))
        row.update(eval_binary(g["is_anomaly_final"], g["score"]))
        per_category_rows.append(row)

    return pd.DataFrame(per_config_rows), pd.DataFrame(per_category_rows)


def build_primary_table(per_config: pd.DataFrame) -> pd.DataFrame:
    # Paper-facing primary test:
    # QCR-U is a crop/candidate method, so crop_topk_ensemble is the relevant primary setting.
    primary = per_config[
        (per_config["dataset"] == "VisA")
        & (per_config["strategy"] == "inspection_binary")
        & (per_config["eval_mode"] == "crop_topk_ensemble")
    ].copy()

    if primary.empty:
        primary = per_config.copy()
        primary["primary_selection_note"] = "fallback_all_protocols"
    else:
        primary["primary_selection_note"] = "VisA_inspection_binary_crop_topk_ensemble"

    primary = primary.sort_values(["backbone", "variant_id"]).reset_index(drop=True)
    return primary


def build_decision(primary: pd.DataFrame, per_config: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def get_delta(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
        idx = ["backbone", "dataset", "strategy", "eval_mode"]
        piv = df.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
        piv.columns.name = None
        if left not in piv.columns or right not in piv.columns:
            return pd.DataFrame()
        piv[f"delta_{left}_minus_{right}"] = piv[left] - piv[right]
        return piv

    for scope, df in [("primary_protocol", primary), ("all_protocols", per_config)]:
        for left, right, label in [
            ("V6", "V3", "adaptive_qcru_minus_naive"),
            ("V6", "V4", "adaptive_qcru_minus_quality"),
            ("V6", "V5", "adaptive_qcru_minus_fixed_qc"),
            ("V4", "V3", "quality_minus_naive"),
        ]:
            d = get_delta(df, left, right)
            if d.empty:
                continue
            delta_col = f"delta_{left}_minus_{right}"
            rows.append(
                {
                    "scope": scope,
                    "comparison": label,
                    "num_protocols": len(d),
                    "wins": int((d[delta_col] > 0).sum()),
                    "win_rate": float((d[delta_col] > 0).mean()),
                    "mean_delta": float(d[delta_col].mean()),
                    "median_delta": float(d[delta_col].median()),
                    "min_delta": float(d[delta_col].min()),
                    "max_delta": float(d[delta_col].max()),
                }
            )

    decision = pd.DataFrame(rows)

    # Conservative final recommendation.
    primary_v6_v4 = decision[
        (decision["scope"] == "primary_protocol")
        & (decision["comparison"] == "adaptive_qcru_minus_quality")
    ]

    if not primary_v6_v4.empty:
        mean_delta = float(primary_v6_v4.iloc[0]["mean_delta"])
        if mean_delta >= 0.005:
            recommendation = "Adaptive QCR-U can be presented as the final candidate method."
            method_name = "Adaptive QCR-U"
        elif mean_delta > 0:
            recommendation = "Use Quality-Calibrated QCR as the main method; describe adaptive consistency as a small gated refinement."
            method_name = "Quality-Calibrated QCR with adaptive consistency refinement"
        else:
            recommendation = "Do not use Adaptive QCR-U as final method; use quality-weighted fusion."
            method_name = "Quality-Calibrated Localization-Guided Fusion"
    else:
        recommendation = "Insufficient primary comparison."
        method_name = "undecided"

    decision["final_recommendation"] = ""
    decision["recommended_method_name"] = ""
    if len(decision) > 0:
        decision.loc[0, "final_recommendation"] = recommendation
        decision.loc[0, "recommended_method_name"] = method_name

    return decision


def write_report(
    per_config: pd.DataFrame,
    per_category: pd.DataFrame,
    primary: pd.DataFrame,
    decision: pd.DataFrame,
) -> None:
    lines = []
    lines += [
        "# Stage 16-B Adaptive QCR-U Paper-facing Comparison",
        "",
        "## 1. Purpose",
        "",
        "This stage connects the Adaptive QCR-U candidate back to a paper-facing comparison table.",
        "",
        "It tests whether Adaptive QCR-U should be the final method name, or whether the method should be downgraded to quality-calibrated localization-guided fusion.",
        "",
        "## 2. Primary Protocol",
        "",
        "The primary protocol is:",
        "",
        "```text",
        "dataset = VisA",
        "strategy = inspection_binary",
        "eval_mode = crop_topk_ensemble",
        "```",
        "",
        "Reason: QCR-U is a candidate/crop reliability method. `full_all` is useful for diagnostics but is not the correct primary protocol for a crop-based reliability module.",
        "",
        "## 3. Primary Protocol Table",
        "",
        "| Backbone | Variant | AUROC | AP | Best F1 | Best Acc |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, r in primary.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['variant']} | "
            f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['best_accuracy']:.4f} |"
        )

    lines += [
        "",
        "## 4. Decision Summary",
        "",
        "| Scope | Comparison | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in decision.iterrows():
        lines.append(
            f"| {r['scope']} | {r['comparison']} | {int(r['wins'])} | {int(r['num_protocols'])} | "
            f"{r['win_rate']:.4f} | {r['mean_delta']:+.4f} | {r['median_delta']:+.4f} | "
            f"{r['min_delta']:+.4f} | {r['max_delta']:+.4f} |"
        )

    if not decision.empty:
        rec = decision.iloc[0]["final_recommendation"]
        name = decision.iloc[0]["recommended_method_name"]
    else:
        rec = "insufficient evidence"
        name = "undecided"

    lines += [
        "",
        "## 5. Final Trial Recommendation",
        "",
        f"- recommended method name: `{name}`",
        f"- recommendation: {rec}",
        "",
        "## 6. Interpretation Rule",
        "",
        "If Adaptive QCR-U only improves over quality-only by a negligible margin, the paper should not overclaim adaptive consistency.",
        "",
        "In that case, the correct claim is:",
        "",
        "```text",
        "Candidate quality provides the main reliability calibration gain, while adaptive consistency is a conservative refinement that avoids fixed-consistency degradation.",
        "```",
        "",
        "## 7. Outputs",
        "",
        f"- `{OUT_PER_CONFIG.relative_to(ROOT)}`",
        f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
        f"- `{OUT_PRIMARY.relative_to(ROOT)}`",
        f"- `{OUT_DECISION.relative_to(ROOT)}`",
        f"- `{OUT_REPORT.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    df = read_csv_strict(IN_PRED)
    base = build_base_table(df)
    long_df = make_long(base)
    per_config, per_category = summarize(long_df)

    primary = build_primary_table(per_config)
    decision = build_decision(primary, per_config)

    per_config.to_csv(OUT_PER_CONFIG, index=False, lineterminator="\n")
    per_category.to_csv(OUT_PER_CATEGORY, index=False, lineterminator="\n")
    primary.to_csv(OUT_PRIMARY, index=False, lineterminator="\n")
    decision.to_csv(OUT_DECISION, index=False, lineterminator="\n")

    write_report(per_config, per_category, primary, decision)

    print("[DONE]", OUT_PER_CONFIG)
    print("[DONE]", OUT_PER_CATEGORY)
    print("[DONE]", OUT_PRIMARY)
    print("[DONE]", OUT_DECISION)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== primary table =====")
    print(primary[["backbone", "variant_id", "variant", "auroc", "ap", "best_f1", "best_accuracy"]].to_string(index=False))
    print()
    print("===== decision =====")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
