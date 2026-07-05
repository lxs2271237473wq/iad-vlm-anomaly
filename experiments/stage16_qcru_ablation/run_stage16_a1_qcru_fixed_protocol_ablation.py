from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_PER_CONFIG = OUT_DIR / "stage16_a1_qcru_fixed_ablation_per_config.csv"
OUT_PER_CATEGORY = OUT_DIR / "stage16_a1_qcru_fixed_ablation_per_category.csv"
OUT_BEST = OUT_DIR / "stage16_a1_qcru_fixed_ablation_best_by_protocol.csv"
OUT_REPORT = DOC_DIR / "stage16_a1_qcru_fixed_protocol_ablation_report.md"

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
        raise RuntimeError(
            f"{path} was read as <=1 column. Local CSV formatting is broken; repair line breaks before Stage 16-A1."
        )
    return df


def minmax_safe(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    lo = x.min()
    hi = x.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    positives = y.sum()
    if positives <= 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / positives)


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    # Mann-Whitney U with average ranks for ties.
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
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def best_f1_acc_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    thresholds = np.unique(y_score)
    if len(thresholds) > 1000:
        thresholds = np.quantile(y_score, np.linspace(0, 1, 1000))
        thresholds = np.unique(thresholds)

    best_f1 = -1.0
    best_acc = -1.0
    best_thr = float("nan")

    for thr in thresholds:
        pred = (y_score >= thr).astype(int)

        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        acc = float((pred == y_true).mean())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

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

    if len(y) == 0:
        return {
            "num_images": 0,
            "num_normal": 0,
            "num_anomaly": 0,
            "has_both_classes": False,
            "auroc": float("nan"),
            "ap": float("nan"),
            "best_f1": float("nan"),
            "best_accuracy": float("nan"),
            "best_threshold": float("nan"),
        }

    num_anomaly = int(y.sum())
    num_normal = int(len(y) - num_anomaly)
    has_both = num_anomaly > 0 and num_normal > 0

    if not has_both:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = roc_auc_score_np(y, s)
        ap = average_precision_score_np(y, s)

    best_f1, best_acc, best_thr = best_f1_acc_threshold(y, s)

    return {
        "num_images": int(len(y)),
        "num_normal": num_normal,
        "num_anomaly": num_anomaly,
        "has_both_classes": bool(has_both),
        "auroc": auc,
        "ap": ap,
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "best_threshold": best_thr,
    }


def build_base_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Stage 9 prediction table has many duplicated image rows because fusion weights/methods vary.
    # Base signals are image-level and should be identical across those rows, so we deduplicate.
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

    base["M_crop_vlm"] = base["vlm_score_norm"]
    base["D_detector"] = base["detector_score_norm"]
    base["Q_quality"] = base["candidate_quality_norm"].fillna(0.0)
    base["K_consistency"] = base["high_high_consistency"].fillna(0.0)

    # Fixed, non-tuned ablation formulas.
    base["score_detector_only"] = base["D_detector"]
    base["score_crop_topk_vlm"] = base["M_crop_vlm"]
    base["score_naive_detector_crop_fusion"] = 0.5 * base["D_detector"] + 0.5 * base["M_crop_vlm"]

    # Quality should modulate whether the crop VLM signal is trusted, not replace the detector score.
    base["score_quality_weighted_crop_raw"] = (
        0.5 * base["D_detector"]
        + 0.5 * (base["M_crop_vlm"] * (0.5 + 0.5 * base["Q_quality"]))
    )

    # Consistency gets a small fixed weight. This is not tuned on test labels.
    base["score_quality_consistency_fusion_raw"] = (
        0.40 * base["D_detector"]
        + 0.40 * base["M_crop_vlm"]
        + 0.10 * base["Q_quality"]
        + 0.10 * base["K_consistency"]
    )

    # Normalize raw variants within each protocol group so scores are comparable for threshold metrics.
    group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
    for raw_col, out_col in [
        ("score_quality_weighted_crop_raw", "score_quality_weighted_crop"),
        ("score_quality_consistency_fusion_raw", "score_quality_consistency_fusion"),
    ]:
        base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)

    return base


def make_variant_long(base: pd.DataFrame) -> pd.DataFrame:
    variants = [
        ("V0", "detector_only", "score_detector_only", False, False),
        ("V2", "crop_topk_vlm", "score_crop_topk_vlm", False, False),
        ("V3", "naive_detector_crop_fusion", "score_naive_detector_crop_fusion", False, False),
        ("V4", "quality_weighted_crop", "score_quality_weighted_crop", True, False),
        ("V5", "quality_consistency_fusion", "score_quality_consistency_fusion", True, True),
    ]

    rows = []
    id_cols = [
        "backbone",
        "dataset",
        "category",
        "strategy",
        "eval_mode",
        "image_key",
        "is_anomaly_final",
    ]
    optional_cols = ["has_candidate", "num_candidates"]
    id_cols += [c for c in optional_cols if c in base.columns]

    for variant_id, variant, score_col, uses_q, uses_k in variants:
        tmp = base[id_cols].copy()
        tmp["variant_id"] = variant_id
        tmp["variant"] = variant
        tmp["score"] = base[score_col].astype(float)
        tmp["uses_quality"] = uses_q
        tmp["uses_consistency"] = uses_k
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def summarize(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["backbone", "dataset", "strategy", "eval_mode", "variant_id", "variant"]

    per_config_rows = []
    per_category_rows = []

    for keys, g in long_df.groupby(group_cols, dropna=False):
        metric = eval_binary(g["is_anomaly_final"], g["score"])
        row = dict(zip(group_cols, keys))
        row.update(metric)
        per_config_rows.append(row)

        for cat, cg in g.groupby("category", dropna=False):
            cm = eval_binary(cg["is_anomaly_final"], cg["score"])
            crow = dict(zip(group_cols, keys))
            crow["category"] = cat
            crow.update(cm)
            per_category_rows.append(crow)

    per_config = pd.DataFrame(per_config_rows)
    per_category = pd.DataFrame(per_category_rows)

    # Choose best fixed-protocol row for each backbone/dataset/strategy/eval_mode by V5 AUROC,
    # but report every variant in that same protocol. This avoids cherry-picking variant formulas.
    best_rows = []
    for keys, g in per_config.groupby(["backbone", "dataset", "strategy", "eval_mode"], dropna=False):
        v5 = g[g["variant"] == "quality_consistency_fusion"]
        if v5.empty:
            continue
        row = v5.sort_values("auroc", ascending=False).iloc[0].copy()
        best_rows.append(row)

    best_protocol = pd.DataFrame(best_rows)
    if not best_protocol.empty:
        best_protocol = best_protocol.sort_values("auroc", ascending=False).reset_index(drop=True)
        best_protocol["rank_by_v5_auroc"] = range(1, len(best_protocol) + 1)

    return per_config, per_category, best_protocol


def write_report(
    base: pd.DataFrame,
    per_config: pd.DataFrame,
    per_category: pd.DataFrame,
    best_protocol: pd.DataFrame,
) -> None:
    lines = []
    lines += [
        "# Stage 16-A1 QCR-U Fixed-Protocol Ablation",
        "",
        "## 1. Purpose",
        "",
        "This stage evaluates fixed, non-tuned QCR-U ablation variants using the existing Stage 9 prediction table.",
        "",
        "It does not train models, rerun VLM inference, or tune weights on the test set.",
        "",
        "## 2. Input",
        "",
        f"- source: `{IN_PRED.relative_to(ROOT)}`",
        f"- deduplicated base rows: `{len(base)}`",
        "",
        "The base table contains detector score, crop VLM score, candidate quality, and detector-VLM consistency.",
        "",
        "## 3. Fixed Ablation Variants",
        "",
        "| Variant | Formula | Meaning |",
        "|---|---|---|",
        "| detector_only | `D` | detector score only |",
        "| crop_topk_vlm | `M` | crop VLM score only |",
        "| naive_detector_crop_fusion | `0.5D + 0.5M` | naive fusion baseline |",
        "| quality_weighted_crop | `0.5D + 0.5(M * (0.5 + 0.5Q))` | candidate quality modulates VLM evidence |",
        "| quality_consistency_fusion | `0.4D + 0.4M + 0.1Q + 0.1K` | fixed Q+C fusion variant |",
        "",
        "Where `D` is detector score, `M` is crop VLM abnormal score, `Q` is candidate quality, and `K` is detector-VLM high-high consistency.",
        "",
        "## 4. Best Protocols by Q+C Fusion AUROC",
        "",
        "| Rank | Backbone | Dataset | Strategy | Eval Mode | V5 AUROC | V5 AP | V5 Best F1 |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]

    if best_protocol.empty:
        lines.append("| - | - | - | - | - | - | - | - |")
    else:
        for _, r in best_protocol.head(20).iterrows():
            lines.append(
                f"| {int(r['rank_by_v5_auroc'])} | {r['backbone']} | {r['dataset']} | "
                f"{r['strategy']} | {r['eval_mode']} | "
                f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} |"
            )

    lines += [
        "",
        "## 5. Variant Comparison Within the Best Protocol",
        "",
    ]

    if not best_protocol.empty:
        best = best_protocol.iloc[0]
        mask = (
            (per_config["backbone"] == best["backbone"])
            & (per_config["dataset"] == best["dataset"])
            & (per_config["strategy"] == best["strategy"])
            & (per_config["eval_mode"] == best["eval_mode"])
        )
        comp = per_config[mask].sort_values("variant_id")

        lines += [
            f"Best protocol by V5 AUROC: `{best['backbone']} / {best['dataset']} / {best['strategy']} / {best['eval_mode']}`.",
            "",
            "| Variant | AUROC | AP | Best F1 | Best Accuracy |",
            "|---|---:|---:|---:|---:|",
        ]

        for _, r in comp.iterrows():
            lines.append(
                f"| {r['variant']} | {r['auroc']:.4f} | {r['ap']:.4f} | "
                f"{r['best_f1']:.4f} | {r['best_accuracy']:.4f} |"
            )

        v = dict(zip(comp["variant"], comp["auroc"]))
        naive = v.get("naive_detector_crop_fusion")
        qc = v.get("quality_consistency_fusion")
        q = v.get("quality_weighted_crop")
        det = v.get("detector_only")
        crop = v.get("crop_topk_vlm")

        lines += [
            "",
            "Key AUROC deltas in the best protocol:",
            "",
        ]

        if qc is not None and naive is not None:
            lines.append(f"- Q+C fusion minus naive fusion: `{qc - naive:+.4f}`.")
        if q is not None and naive is not None:
            lines.append(f"- Quality-weighted crop minus naive fusion: `{q - naive:+.4f}`.")
        if qc is not None and det is not None:
            lines.append(f"- Q+C fusion minus detector-only: `{qc - det:+.4f}`.")
        if qc is not None and crop is not None:
            lines.append(f"- Q+C fusion minus crop VLM only: `{qc - crop:+.4f}`.")

    lines += [
        "",
        "## 6. Interpretation Rules",
        "",
        "This stage is diagnostic. A positive result only means fixed Q+C evidence is useful under the existing Stage 9 signals.",
        "",
        "It is not yet the final QCR-U method unless:",
        "",
        "1. Q+C improves over naive fusion consistently, not only in one protocol.",
        "2. The selected protocol is justified without test-set tuning.",
        "3. Per-category results do not collapse on one or more primary categories.",
        "",
        "## 7. Outputs",
        "",
        f"- `{OUT_PER_CONFIG.relative_to(ROOT)}`",
        f"- `{OUT_PER_CATEGORY.relative_to(ROOT)}`",
        f"- `{OUT_BEST.relative_to(ROOT)}`",
        f"- `{OUT_REPORT.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    df = read_csv_strict(IN_PRED)
    base = build_base_table(df)
    long_df = make_variant_long(base)
    per_config, per_category, best_protocol = summarize(long_df)

    per_config.to_csv(OUT_PER_CONFIG, index=False, lineterminator="\n")
    per_category.to_csv(OUT_PER_CATEGORY, index=False, lineterminator="\n")
    best_protocol.to_csv(OUT_BEST, index=False, lineterminator="\n")

    write_report(base, per_config, per_category, best_protocol)

    print("[DONE]", OUT_PER_CONFIG)
    print("[DONE]", OUT_PER_CATEGORY)
    print("[DONE]", OUT_BEST)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== top protocols by V5 quality_consistency_fusion AUROC =====")
    if best_protocol.empty:
        print("EMPTY")
    else:
        print(
            best_protocol[
                [
                    "rank_by_v5_auroc",
                    "backbone",
                    "dataset",
                    "strategy",
                    "eval_mode",
                    "auroc",
                    "ap",
                    "best_f1",
                    "best_accuracy",
                ]
            ].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()
