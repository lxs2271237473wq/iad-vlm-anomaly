from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_PER_CONFIG = OUT_DIR / "stage16_a3_adaptive_qcru_per_config.csv"
OUT_DELTA = OUT_DIR / "stage16_a3_adaptive_qcru_delta_by_protocol.csv"
OUT_FAILURES = OUT_DIR / "stage16_a3_adaptive_qcru_failure_cases.csv"
OUT_DOC = DOC_DIR / "stage16_a3_adaptive_qcru_report.md"


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


def read_csv_strict(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Fix CSV line breaks before running Stage 16-A3.")
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

    # Existing baselines.
    base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]
    base["score_quality_raw"] = 0.5 * base["D"] + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
    base["score_fixed_qc_raw"] = 0.40 * base["D"] + 0.40 * base["M"] + 0.10 * base["Q"] + 0.10 * base["K"]

    # Adaptive QCR-U:
    # Start from quality-weighted core.
    # Add a conservative consistency bonus only when:
    # - candidate quality is high,
    # - detector and VLM agree,
    # - both detector and VLM provide high anomaly evidence.
    #
    # This is label-free and intentionally conservative.
    agreement = 1.0 - (base["D"] - base["M"]).abs()
    agreement = agreement.clip(lower=0.0, upper=1.0)

    mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
    adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence

    base["adaptive_gate"] = adaptive_gate

    # Small fixed coefficient. This is not selected from test labels.
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
    variants = [
        ("V3", "naive_detector_crop_fusion", "score_naive"),
        ("V4", "quality_weighted_crop", "score_quality"),
        ("V5", "fixed_quality_consistency", "score_fixed_qc"),
        ("V6", "adaptive_qcru", "score_adaptive_qcru"),
    ]

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
    for vid, variant, col in variants:
        tmp = base[id_cols].copy()
        tmp["variant_id"] = vid
        tmp["variant"] = variant
        tmp["score"] = base[col].astype(float)
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["backbone", "dataset", "strategy", "eval_mode", "variant_id", "variant"]
    rows = []

    for keys, g in long_df.groupby(group_cols, dropna=False):
        metric = eval_binary(g["is_anomaly_final"], g["score"])
        row = dict(zip(group_cols, keys))
        row.update(metric)
        rows.append(row)

    return pd.DataFrame(rows)


def build_delta(per_config: pd.DataFrame) -> pd.DataFrame:
    idx = ["backbone", "dataset", "strategy", "eval_mode"]
    piv = per_config.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
    piv.columns.name = None

    for col in ["V3", "V4", "V5", "V6"]:
        if col not in piv.columns:
            raise RuntimeError(f"Missing {col} in per-config pivot.")

    piv["delta_v6_minus_v3_naive"] = piv["V6"] - piv["V3"]
    piv["delta_v6_minus_v4_quality"] = piv["V6"] - piv["V4"]
    piv["delta_v6_minus_v5_fixed_qc"] = piv["V6"] - piv["V5"]
    piv["delta_v5_minus_v4_quality"] = piv["V5"] - piv["V4"]

    piv["v6_beats_naive"] = piv["delta_v6_minus_v3_naive"] > 0
    piv["v6_beats_quality"] = piv["delta_v6_minus_v4_quality"] > 0
    piv["v6_beats_fixed_qc"] = piv["delta_v6_minus_v5_fixed_qc"] > 0
    piv["v5_beats_quality"] = piv["delta_v5_minus_v4_quality"] > 0

    return piv


def summarize_delta(delta: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("V6 > V3 naive", "v6_beats_naive", "delta_v6_minus_v3_naive"),
        ("V6 > V4 quality", "v6_beats_quality", "delta_v6_minus_v4_quality"),
        ("V6 > V5 fixed Q+C", "v6_beats_fixed_qc", "delta_v6_minus_v5_fixed_qc"),
        ("V5 > V4 quality", "v5_beats_quality", "delta_v5_minus_v4_quality"),
    ]

    rows = []
    for name, win_col, delta_col in checks:
        rows.append(
            {
                "check": name,
                "wins": int(delta[win_col].sum()),
                "total_protocols": int(len(delta)),
                "win_rate": float(delta[win_col].mean()),
                "mean_delta": float(delta[delta_col].mean()),
                "median_delta": float(delta[delta_col].median()),
                "min_delta": float(delta[delta_col].min()),
                "max_delta": float(delta[delta_col].max()),
            }
        )

    for eval_mode, g in delta.groupby("eval_mode"):
        rows.append(
            {
                "check": f"V6 > V4 quality by eval_mode={eval_mode}",
                "wins": int(g["v6_beats_quality"].sum()),
                "total_protocols": int(len(g)),
                "win_rate": float(g["v6_beats_quality"].mean()),
                "mean_delta": float(g["delta_v6_minus_v4_quality"].mean()),
                "median_delta": float(g["delta_v6_minus_v4_quality"].median()),
                "min_delta": float(g["delta_v6_minus_v4_quality"].min()),
                "max_delta": float(g["delta_v6_minus_v4_quality"].max()),
            }
        )

        rows.append(
            {
                "check": f"V6 > V5 fixed Q+C by eval_mode={eval_mode}",
                "wins": int(g["v6_beats_fixed_qc"].sum()),
                "total_protocols": int(len(g)),
                "win_rate": float(g["v6_beats_fixed_qc"].mean()),
                "mean_delta": float(g["delta_v6_minus_v5_fixed_qc"].mean()),
                "median_delta": float(g["delta_v6_minus_v5_fixed_qc"].median()),
                "min_delta": float(g["delta_v6_minus_v5_fixed_qc"].min()),
                "max_delta": float(g["delta_v6_minus_v5_fixed_qc"].max()),
            }
        )

    return pd.DataFrame(rows)


def write_report(per_config: pd.DataFrame, delta: pd.DataFrame, summary: pd.DataFrame) -> None:
    best = per_config[per_config["variant_id"] == "V6"].sort_values("auroc", ascending=False).reset_index(drop=True)
    best["rank_by_v6_auroc"] = range(1, len(best) + 1)

    lines = []
    lines += [
        "# Stage 16-A3 Adaptive QCR-U",
        "",
        "## 1. Purpose",
        "",
        "Stage 16-A2 showed that candidate quality is stable, while fixed consistency is not universally beneficial.",
        "",
        "This stage tests an adaptive QCR-U score that uses quality-weighted crop scoring as the stable core and applies consistency only as a conservative reliability-gated bonus.",
        "",
        "## 2. Formula",
        "",
        "```text",
        "D = detector anomaly score",
        "M = crop VLM anomaly score",
        "Q = candidate quality",
        "K = high-high detector/VLM consistency",
        "",
        "S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)",
        "agreement = 1 - |D - M|",
        "mutual_anomaly_evidence = min(D, M)",
        "gate = Q * K * agreement * mutual_anomaly_evidence",
        "S_adaptive = S_quality + 0.05 * gate",
        "```",
        "",
        "The coefficient `0.05` is fixed and intentionally conservative. It is not selected by test-set tuning.",
        "",
        "## 3. Robustness Summary",
        "",
        "| Check | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['check']} | {int(r['wins'])} | {int(r['total_protocols'])} | "
            f"{r['win_rate']:.4f} | {r['mean_delta']:.4f} | {r['median_delta']:.4f} | "
            f"{r['min_delta']:.4f} | {r['max_delta']:.4f} |"
        )

    lines += [
        "",
        "## 4. Adaptive QCR-U Protocol Ranking",
        "",
        "| Rank | Backbone | Strategy | Eval Mode | V6 AUROC | AP | Best F1 |",
        "|---:|---|---|---|---:|---:|---:|",
    ]

    for _, r in best.iterrows():
        lines.append(
            f"| {int(r['rank_by_v6_auroc'])} | {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
            f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} |"
        )

    lines += [
        "",
        "## 5. Protocol-level Delta Table",
        "",
        "| Backbone | Strategy | Eval Mode | V3 | V4 | V5 | V6 | V6-V3 | V6-V4 | V6-V5 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    display = delta.sort_values("delta_v6_minus_v3_naive", ascending=False)
    for _, r in display.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
            f"{r['V3']:.4f} | {r['V4']:.4f} | {r['V5']:.4f} | {r['V6']:.4f} | "
            f"{r['delta_v6_minus_v3_naive']:+.4f} | "
            f"{r['delta_v6_minus_v4_quality']:+.4f} | "
            f"{r['delta_v6_minus_v5_fixed_qc']:+.4f} |"
        )

    lines += [
        "",
        "## 6. Decision Rule",
        "",
        "If adaptive QCR-U beats naive fusion consistently and avoids the full_all degradation of fixed Q+C, it can replace fixed Q+C as the next method candidate.",
        "",
        "If adaptive QCR-U still fails to beat quality-only, the method should be simplified to quality-weighted crop fusion and consistency should be moved to analysis rather than method.",
        "",
        "## 7. Outputs",
        "",
        f"- `{OUT_PER_CONFIG.relative_to(ROOT)}`",
        f"- `{OUT_DELTA.relative_to(ROOT)}`",
        f"- `{OUT_FAILURES.relative_to(ROOT)}`",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    df = read_csv_strict(IN_PRED)
    base = build_base_table(df)
    long_df = make_long(base)
    per_config = summarize(long_df)
    delta = build_delta(per_config)
    summary = summarize_delta(delta)

    failures = delta[
        (~delta["v6_beats_naive"])
        | (~delta["v6_beats_quality"])
        | (~delta["v6_beats_fixed_qc"])
    ].copy()

    reasons = []
    for _, r in failures.iterrows():
        rs = []
        if not bool(r["v6_beats_naive"]):
            rs.append("V6_not_better_than_naive")
        if not bool(r["v6_beats_quality"]):
            rs.append("V6_not_better_than_quality")
        if not bool(r["v6_beats_fixed_qc"]):
            rs.append("V6_not_better_than_fixed_qc")
        reasons.append(";".join(rs))
    if not failures.empty:
        failures["failure_reason"] = reasons

    per_config.to_csv(OUT_PER_CONFIG, index=False, lineterminator="\n")
    delta.to_csv(OUT_DELTA, index=False, lineterminator="\n")
    failures.to_csv(OUT_FAILURES, index=False, lineterminator="\n")

    write_report(per_config, delta, summary)

    print("[DONE]", OUT_PER_CONFIG)
    print("[DONE]", OUT_DELTA)
    print("[DONE]", OUT_FAILURES)
    print("[DONE]", OUT_DOC)
    print()
    print("===== summary =====")
    print(summary.to_string(index=False))
    print()
    print("===== failures =====")
    if failures.empty:
        print("none")
    else:
        print(
            failures[
                [
                    "backbone",
                    "strategy",
                    "eval_mode",
                    "delta_v6_minus_v3_naive",
                    "delta_v6_minus_v4_quality",
                    "delta_v6_minus_v5_fixed_qc",
                    "failure_reason",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
