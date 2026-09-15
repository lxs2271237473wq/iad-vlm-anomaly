from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

IN_PRED = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_predictions.csv"

OUT_DIR = ROOT / "results" / "stage9_qcr_u"
OUT_PRED = OUT_DIR / "stage9_a3_qcr_u_debiased_predictions.csv"
OUT_SUMMARY = OUT_DIR / "stage9_a3_qcr_u_debiased_summary.csv"
OUT_PERCAT = OUT_DIR / "stage9_a3_qcr_u_debiased_per_category.csv"
OUT_REPORT = OUT_DIR / "stage9_a3_qcr_u_debias_report.md"


def to_binary_series(series: pd.Series) -> pd.Series:
    def convert(x: object) -> int:
        if pd.isna(x):
            return 0
        if isinstance(x, bool):
            return int(x)
        text = str(x).strip().lower()
        if text in {"1", "true", "yes", "anomaly", "defect", "bad"}:
            return 1
        if text in {"0", "false", "no", "normal", "good"}:
            return 0
        try:
            return int(float(text) > 0)
        except Exception:
            return 0

    return series.map(convert).astype(int)


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(s).rank(method="average").to_numpy()
    pos_rank_sum = ranks[y == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]

    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    ap = precision[y_sorted == 1].sum() / n_pos
    return float(ap)


def best_f1_accuracy(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")

    thresholds = np.unique(s)

    best_f1 = -1.0
    best_acc = -1.0
    best_thr = float(thresholds[0]) if len(thresholds) else 0.0

    for thr in thresholds:
        pred = (s >= thr).astype(int)

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else 2 * tp / denom
        acc = float((pred == y).mean())

        if f1 > best_f1 or (math.isclose(f1, best_f1) and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_thr = float(thr)

    return float(best_f1), float(best_acc), float(best_thr)


def get_base_predictions() -> pd.DataFrame:
    if not IN_PRED.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PRED}")

    pred = pd.read_csv(IN_PRED)

    required = [
        "backbone",
        "strategy",
        "eval_mode",
        "category",
        "image_key",
        "is_anomaly_final",
        "vlm_score_norm",
        "detector_score_norm",
        "candidate_quality_norm",
        "high_high_consistency",
        "has_candidate",
    ]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"Missing columns in {IN_PRED}: {missing}")

    base = pred[pred["fusion_method"] == "vlm_only"].copy()

    base["is_anomaly_final"] = to_binary_series(base["is_anomaly_final"])
    base["has_candidate"] = to_binary_series(base["has_candidate"])

    for col in [
        "vlm_score_norm",
        "detector_score_norm",
        "candidate_quality_norm",
        "high_high_consistency",
    ]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    base["q_original"] = base["candidate_quality_norm"]

    # Key debias step:
    # If no candidate exists, do not treat Q as 0.
    # Set Q to neutral 0.5 so candidate absence itself is not used as anomaly evidence.
    base["q_neutral"] = np.where(base["has_candidate"] == 1, base["candidate_quality_norm"], 0.5)

    # Also test a harsher version: remove Q completely.
    base["q_removed"] = 0.0

    return base


def build_debiased_scores(base: pd.DataFrame) -> pd.DataFrame:
    rows = []

    methods = {
        "vlm_only": lambda d: d["vlm_score_norm"],
        "detector_only": lambda d: d["detector_score_norm"],
        "candidate_quality_original": lambda d: d["q_original"],
        "candidate_quality_neutral": lambda d: d["q_neutral"],

        # Original QCR-U-like scores.
        "qcr_u_original_fixed": lambda d: (
            0.65 * d["vlm_score_norm"]
            + 0.20 * d["q_original"]
            + 0.15 * d["high_high_consistency"]
        ),
        "qcr_u_original_detector_aware": lambda d: (
            0.55 * d["vlm_score_norm"]
            + 0.20 * d["q_original"]
            + 0.15 * d["high_high_consistency"]
            + 0.10 * d["detector_score_norm"]
        ),

        # Debiased QCR-U: no-candidate images get neutral Q=0.5.
        "qcr_u_neutral_q_fixed": lambda d: (
            0.65 * d["vlm_score_norm"]
            + 0.20 * d["q_neutral"]
            + 0.15 * d["high_high_consistency"]
        ),
        "qcr_u_neutral_q_detector_aware": lambda d: (
            0.55 * d["vlm_score_norm"]
            + 0.20 * d["q_neutral"]
            + 0.15 * d["high_high_consistency"]
            + 0.10 * d["detector_score_norm"]
        ),

        # No-Q QCR-U: tests whether M/K/D alone are sufficient.
        "qcr_u_no_q_fixed": lambda d: (
            0.75 * d["vlm_score_norm"]
            + 0.25 * d["high_high_consistency"]
        ),
        "qcr_u_no_q_detector_aware": lambda d: (
            0.65 * d["vlm_score_norm"]
            + 0.20 * d["high_high_consistency"]
            + 0.15 * d["detector_score_norm"]
        ),
    }

    base_cols = [
        "backbone",
        "strategy",
        "eval_mode",
        "used_mode",
        "category",
        "image_key",
        "is_anomaly_final",
        "has_candidate",
        "vlm_score_norm",
        "detector_score_norm",
        "candidate_quality_norm",
        "high_high_consistency",
        "q_original",
        "q_neutral",
    ]
    base_cols = [c for c in base_cols if c in base.columns]

    for method_name, score_fn in methods.items():
        part = base[base_cols].copy()
        part["debiased_method"] = method_name
        part["score"] = score_fn(base).astype(float)
        rows.append(part)

    return pd.concat(rows, ignore_index=True)


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["backbone", "strategy", "eval_mode", "debiased_method"]

    for keys, part in pred.groupby(group_cols, dropna=False, sort=False):
        backbone, strategy, eval_mode, method = keys

        y = to_binary_series(part["is_anomaly_final"]).to_numpy()
        s = pd.to_numeric(part["score"], errors="coerce").fillna(0.0).to_numpy()

        f1, acc, thr = best_f1_accuracy(y, s)

        rows.append(
            {
                "backbone": backbone,
                "strategy": strategy,
                "eval_mode": eval_mode,
                "debiased_method": method,
                "num_images": int(len(part)),
                "num_normal": int((y == 0).sum()),
                "num_anomaly": int((y == 1).sum()),
                "candidate_coverage": float(to_binary_series(part["has_candidate"]).mean()),
                "auroc": binary_auroc(y, s),
                "ap": average_precision(y, s),
                "best_f1": f1,
                "best_accuracy": acc,
                "best_threshold": thr,
            }
        )

    out = pd.DataFrame(rows)

    base = out[out["debiased_method"] == "vlm_only"][
        ["backbone", "strategy", "eval_mode", "auroc", "ap", "best_f1", "best_accuracy"]
    ].rename(
        columns={
            "auroc": "vlm_only_auroc",
            "ap": "vlm_only_ap",
            "best_f1": "vlm_only_best_f1",
            "best_accuracy": "vlm_only_best_accuracy",
        }
    )

    out = out.merge(base, on=["backbone", "strategy", "eval_mode"], how="left")
    out["delta_auroc_vs_vlm"] = out["auroc"] - out["vlm_only_auroc"]
    out["delta_ap_vs_vlm"] = out["ap"] - out["vlm_only_ap"]
    out["delta_best_f1_vs_vlm"] = out["best_f1"] - out["vlm_only_best_f1"]
    out["delta_accuracy_vs_vlm"] = out["best_accuracy"] - out["vlm_only_best_accuracy"]

    return out.sort_values(
        ["backbone", "strategy", "eval_mode", "auroc"],
        ascending=[True, True, True, False],
    )


def per_category_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["backbone", "strategy", "eval_mode", "debiased_method", "category"]

    for keys, part in pred.groupby(group_cols, dropna=False, sort=False):
        backbone, strategy, eval_mode, method, category = keys

        y = to_binary_series(part["is_anomaly_final"]).to_numpy()
        s = pd.to_numeric(part["score"], errors="coerce").fillna(0.0).to_numpy()

        f1, acc, thr = best_f1_accuracy(y, s)

        rows.append(
            {
                "backbone": backbone,
                "strategy": strategy,
                "eval_mode": eval_mode,
                "debiased_method": method,
                "category": category,
                "num_images": int(len(part)),
                "num_normal": int((y == 0).sum()),
                "num_anomaly": int((y == 1).sum()),
                "has_both_classes": bool((y == 0).any() and (y == 1).any()),
                "candidate_coverage": float(to_binary_series(part["has_candidate"]).mean()),
                "auroc": binary_auroc(y, s),
                "ap": average_precision(y, s),
                "best_f1": f1,
                "best_accuracy": acc,
                "best_threshold": thr,
            }
        )

    out = pd.DataFrame(rows)

    base = out[out["debiased_method"] == "vlm_only"][
        [
            "backbone",
            "strategy",
            "eval_mode",
            "category",
            "auroc",
            "ap",
            "best_f1",
            "best_accuracy",
        ]
    ].rename(
        columns={
            "auroc": "vlm_only_auroc",
            "ap": "vlm_only_ap",
            "best_f1": "vlm_only_best_f1",
            "best_accuracy": "vlm_only_best_accuracy",
        }
    )

    out = out.merge(base, on=["backbone", "strategy", "eval_mode", "category"], how="left")
    out["delta_auroc_vs_vlm"] = out["auroc"] - out["vlm_only_auroc"]
    out["delta_ap_vs_vlm"] = out["ap"] - out["vlm_only_ap"]
    out["delta_best_f1_vs_vlm"] = out["best_f1"] - out["vlm_only_best_f1"]

    return out.sort_values(
        ["backbone", "strategy", "eval_mode", "category", "auroc"],
        ascending=[True, True, True, True, False],
    )


def write_report(summary: pd.DataFrame, percat: pd.DataFrame) -> None:
    qcr_methods = [
        "qcr_u_original_fixed",
        "qcr_u_original_detector_aware",
        "qcr_u_neutral_q_fixed",
        "qcr_u_neutral_q_detector_aware",
        "qcr_u_no_q_fixed",
        "qcr_u_no_q_detector_aware",
    ]

    best_debiased = (
        summary[summary["debiased_method"].isin(qcr_methods)]
        .sort_values(["auroc", "delta_auroc_vs_vlm"], ascending=False)
        .head(20)
    )

    best_candidate = (
        summary[
            summary["debiased_method"].isin(
                ["candidate_quality_original", "candidate_quality_neutral"]
            )
        ]
        .sort_values("auroc", ascending=False)
        .head(20)
    )

    neutral_q = summary[
        summary["debiased_method"].isin(
            ["qcr_u_neutral_q_fixed", "qcr_u_neutral_q_detector_aware"]
        )
    ].copy()

    no_q = summary[
        summary["debiased_method"].isin(
            ["qcr_u_no_q_fixed", "qcr_u_no_q_detector_aware"]
        )
    ].copy()

    neutral_positive = int((neutral_q["delta_auroc_vs_vlm"] > 0).sum())
    neutral_total = int(len(neutral_q))
    no_q_positive = int((no_q["delta_auroc_vs_vlm"] > 0).sum())
    no_q_total = int(len(no_q))

    lines: List[str] = []

    lines.append("# Stage 9-A3 QCR-U Debias Check Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("Stage 9-A2 showed that candidate_quality_only is extremely strong.")
    lines.append("This stage removes candidate-existence bias by assigning a neutral Q value to images without candidates.")
    lines.append("It reads Stage 9-A1 predictions only and does not train models or regenerate anomaly maps.")
    lines.append("")
    lines.append("## 2. Debias Setting")
    lines.append("")
    lines.append("```text")
    lines.append("q_original = candidate_quality_norm")
    lines.append("q_neutral = candidate_quality_norm if candidate exists else 0.5")
    lines.append("```")
    lines.append("")
    lines.append("The neutral value prevents no-candidate images from being automatically treated as normal through Q=0.")
    lines.append("")
    lines.append("## 3. Output Files")
    lines.append("")
    lines.append(f"- `{OUT_PRED.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_PERCAT.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 4. Best QCR-U / Debiased Rows")
    lines.append("")
    lines.append("| Backbone | Strategy | Eval mode | Method | AUROC | AP | Best F1 | ΔAUROC vs VLM |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")

    for _, r in best_debiased.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['debiased_method']} | "
            f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
        )

    lines.append("")
    lines.append("## 5. Candidate Quality Original vs Neutral")
    lines.append("")
    lines.append("| Backbone | Strategy | Eval mode | Method | AUROC | AP | Best F1 | ΔAUROC vs VLM |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")

    for _, r in best_candidate.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['debiased_method']} | "
            f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
        )

    lines.append("")
    lines.append("## 6. Stability Counts")
    lines.append("")
    lines.append(f"- Neutral-Q QCR-U positive settings: {neutral_positive}/{neutral_total}")
    lines.append(f"- No-Q QCR-U positive settings: {no_q_positive}/{no_q_total}")
    lines.append("")
    lines.append("## 7. Decision Rule")
    lines.append("")
    lines.append("| Observation | Paper Decision |")
    lines.append("|---|---|")
    lines.append("| Neutral-Q QCR-U still improves most settings | Keep QCR-U as a calibration/ablation module |")
    lines.append("| No-Q QCR-U still improves most settings | Emphasize detector-VLM consistency K rather than candidate quality Q |")
    lines.append("| Only original-Q works | Do not use QCR-U as a method claim; keep as leakage/diagnostic analysis |")
    lines.append("")
    lines.append("## 8. Conservative Claim")
    lines.append("")
    lines.append("Until this check is inspected, the safest claim remains:")
    lines.append("")
    lines.append("```text")
    lines.append("QCR-U is a diagnostic calibration mechanism. The paper's core contribution should remain localization-guided VLM reasoning, not candidate-quality fusion.")
    lines.append("```")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = get_base_predictions()
    pred = build_debiased_scores(base)
    summary = summarize(pred)
    percat = per_category_summary(pred)

    pred.to_csv(OUT_PRED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    percat.to_csv(OUT_PERCAT, index=False)
    write_report(summary, percat)

    print("[DONE]", OUT_PRED)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_PERCAT)
    print("[DONE]", OUT_REPORT)
    print("prediction_rows:", len(pred))
    print("summary_rows:", len(summary))
    print("per_category_rows:", len(percat))

    cols = [
        "backbone",
        "strategy",
        "eval_mode",
        "debiased_method",
        "auroc",
        "ap",
        "best_f1",
        "delta_auroc_vs_vlm",
    ]

    print("\nTop debiased QCR-U rows:")
    show = (
        summary[
            summary["debiased_method"].isin(
                [
                    "qcr_u_neutral_q_fixed",
                    "qcr_u_neutral_q_detector_aware",
                    "qcr_u_no_q_fixed",
                    "qcr_u_no_q_detector_aware",
                ]
            )
        ]
        .sort_values(["auroc", "delta_auroc_vs_vlm"], ascending=False)
        .head(20)
    )
    print(show[cols].to_string(index=False))

    print("\nCandidate quality original vs neutral:")
    show_q = (
        summary[
            summary["debiased_method"].isin(
                ["candidate_quality_original", "candidate_quality_neutral"]
            )
        ]
        .sort_values(["backbone", "strategy", "eval_mode", "debiased_method"])
    )
    print(show_q[cols].to_string(index=False))


if __name__ == "__main__":
    main()
