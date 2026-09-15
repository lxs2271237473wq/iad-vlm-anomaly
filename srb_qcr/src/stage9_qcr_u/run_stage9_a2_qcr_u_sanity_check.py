from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()
IN_PRED = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_predictions.csv"
IN_SUMMARY = ROOT / "results" / "stage9_qcr_u" / "stage9_a1_qcr_u_fusion_summary.csv"

OUT_DIR = ROOT / "results" / "stage9_qcr_u"
OUT_PERCAT = OUT_DIR / "stage9_a2_qcr_u_per_category.csv"
OUT_MACRO = OUT_DIR / "stage9_a2_qcr_u_macro_summary.csv"
OUT_DIAG = OUT_DIR / "stage9_a2_qcr_u_signal_diagnostics.csv"
OUT_REPORT = OUT_DIR / "stage9_a2_qcr_u_sanity_report.md"


KEY_METHODS = [
    "vlm_only",
    "detector_only",
    "candidate_quality_only",
    "vlm_detector_avg",
    "qcr_u_fixed",
    "qcr_u_detector_aware",
]


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


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    if math.isclose(float(x[mask].std()), 0.0) or math.isclose(float(y[mask].std()), 0.0):
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def compute_per_category(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["backbone", "strategy", "eval_mode", "fusion_method", "category"]

    for keys, part in pred.groupby(group_cols, dropna=False, sort=False):
        backbone, strategy, eval_mode, method, category = keys

        y = to_binary_series(part["is_anomaly_final"]).to_numpy()
        s = pd.to_numeric(part["fusion_score"], errors="coerce").fillna(0.0).to_numpy()

        f1, acc, thr = best_f1_accuracy(y, s)

        rows.append(
            {
                "backbone": backbone,
                "strategy": strategy,
                "eval_mode": eval_mode,
                "fusion_method": method,
                "category": category,
                "num_images": int(len(part)),
                "num_normal": int((y == 0).sum()),
                "num_anomaly": int((y == 1).sum()),
                "has_both_classes": bool((y == 0).any() and (y == 1).any()),
                "candidate_coverage": float(to_binary_series(part["has_candidate"]).mean())
                if "has_candidate" in part.columns
                else float("nan"),
                "auroc": binary_auroc(y, s),
                "ap": average_precision(y, s),
                "best_f1": f1,
                "best_accuracy": acc,
                "best_threshold": thr,
            }
        )

    out = pd.DataFrame(rows)

    base = out[out["fusion_method"] == "vlm_only"][
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
    out["delta_accuracy_vs_vlm"] = out["best_accuracy"] - out["vlm_only_best_accuracy"]

    return out.sort_values(
        ["backbone", "strategy", "eval_mode", "category", "auroc"],
        ascending=[True, True, True, True, False],
    )


def compute_macro(percat: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["backbone", "strategy", "eval_mode", "fusion_method"]

    valid = percat[percat["has_both_classes"]].copy()

    for keys, part in valid.groupby(group_cols, dropna=False, sort=False):
        backbone, strategy, eval_mode, method = keys

        delta = pd.to_numeric(part["delta_auroc_vs_vlm"], errors="coerce")

        rows.append(
            {
                "backbone": backbone,
                "strategy": strategy,
                "eval_mode": eval_mode,
                "fusion_method": method,
                "num_categories": int(part["category"].nunique()),
                "macro_auroc": float(pd.to_numeric(part["auroc"], errors="coerce").mean()),
                "macro_ap": float(pd.to_numeric(part["ap"], errors="coerce").mean()),
                "macro_best_f1": float(pd.to_numeric(part["best_f1"], errors="coerce").mean()),
                "median_auroc": float(pd.to_numeric(part["auroc"], errors="coerce").median()),
                "min_auroc": float(pd.to_numeric(part["auroc"], errors="coerce").min()),
                "max_auroc": float(pd.to_numeric(part["auroc"], errors="coerce").max()),
                "num_categories_delta_positive": int((delta > 0).sum()),
                "num_categories_delta_negative": int((delta < 0).sum()),
                "mean_delta_auroc_vs_vlm": float(delta.mean()),
                "median_delta_auroc_vs_vlm": float(delta.median()),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(
        ["backbone", "strategy", "eval_mode", "macro_auroc"],
        ascending=[True, True, True, False],
    )


def compute_signal_diagnostics(pred: pd.DataFrame) -> pd.DataFrame:
    base = pred[pred["fusion_method"] == "vlm_only"].copy()

    score_cols = [
        "vlm_score_norm",
        "detector_score_norm",
        "candidate_quality_norm",
        "high_high_consistency",
    ]

    for col in score_cols:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    base["is_anomaly_final"] = to_binary_series(base["is_anomaly_final"])
    base["has_candidate"] = to_binary_series(base["has_candidate"])

    rows = []

    group_cols = ["backbone", "strategy", "eval_mode", "category"]

    for keys, part in base.groupby(group_cols, dropna=False, sort=False):
        backbone, strategy, eval_mode, category = keys

        y = part["is_anomaly_final"]

        normal = part[y == 0]
        anomaly = part[y == 1]

        row = {
            "backbone": backbone,
            "strategy": strategy,
            "eval_mode": eval_mode,
            "category": category,
            "num_images": int(len(part)),
            "num_normal": int(len(normal)),
            "num_anomaly": int(len(anomaly)),
            "candidate_rate_all": float(part["has_candidate"].mean()) if len(part) else float("nan"),
            "candidate_rate_normal": float(normal["has_candidate"].mean()) if len(normal) else float("nan"),
            "candidate_rate_anomaly": float(anomaly["has_candidate"].mean()) if len(anomaly) else float("nan"),
            "q_mean_normal": float(normal["candidate_quality_norm"].mean()) if len(normal) else float("nan"),
            "q_mean_anomaly": float(anomaly["candidate_quality_norm"].mean()) if len(anomaly) else float("nan"),
            "d_mean_normal": float(normal["detector_score_norm"].mean()) if len(normal) else float("nan"),
            "d_mean_anomaly": float(anomaly["detector_score_norm"].mean()) if len(anomaly) else float("nan"),
            "m_mean_normal": float(normal["vlm_score_norm"].mean()) if len(normal) else float("nan"),
            "m_mean_anomaly": float(anomaly["vlm_score_norm"].mean()) if len(anomaly) else float("nan"),
            "k_mean_normal": float(normal["high_high_consistency"].mean()) if len(normal) else float("nan"),
            "k_mean_anomaly": float(anomaly["high_high_consistency"].mean()) if len(anomaly) else float("nan"),
            "corr_q_y": safe_corr(part["candidate_quality_norm"], y),
            "corr_d_y": safe_corr(part["detector_score_norm"], y),
            "corr_m_y": safe_corr(part["vlm_score_norm"], y),
            "corr_k_y": safe_corr(part["high_high_consistency"], y),
            "corr_q_d": safe_corr(part["candidate_quality_norm"], part["detector_score_norm"]),
            "corr_q_m": safe_corr(part["candidate_quality_norm"], part["vlm_score_norm"]),
            "corr_d_m": safe_corr(part["detector_score_norm"], part["vlm_score_norm"]),
        }

        row["candidate_rate_gap_anomaly_minus_normal"] = row["candidate_rate_anomaly"] - row["candidate_rate_normal"]
        row["q_gap_anomaly_minus_normal"] = row["q_mean_anomaly"] - row["q_mean_normal"]
        row["d_gap_anomaly_minus_normal"] = row["d_mean_anomaly"] - row["d_mean_normal"]
        row["m_gap_anomaly_minus_normal"] = row["m_mean_anomaly"] - row["m_mean_normal"]

        rows.append(row)

    out = pd.DataFrame(rows)

    return out.sort_values(
        ["backbone", "strategy", "eval_mode", "category"],
        ascending=[True, True, True, True],
    )


def select_report_tables(macro: pd.DataFrame, percat: pd.DataFrame, diag: pd.DataFrame) -> dict:
    tables = {}

    tables["best_macro_qcr"] = (
        macro[macro["fusion_method"].isin(["qcr_u_fixed", "qcr_u_detector_aware"])]
        .sort_values(["macro_auroc", "mean_delta_auroc_vs_vlm"], ascending=False)
        .head(12)
    )

    tables["candidate_quality_macro"] = (
        macro[macro["fusion_method"] == "candidate_quality_only"]
        .sort_values(["macro_auroc"], ascending=False)
        .head(12)
    )

    tables["qcr_negative_categories"] = (
        percat[
            percat["fusion_method"].isin(["qcr_u_fixed", "qcr_u_detector_aware"])
            & (percat["delta_auroc_vs_vlm"] < 0)
        ]
        .sort_values(["delta_auroc_vs_vlm"])
        .head(20)
    )

    tables["q_dominant_categories"] = (
        diag.sort_values(["corr_q_y", "q_gap_anomaly_minus_normal"], ascending=False)
        .head(20)
    )

    return tables


def write_report(pred: pd.DataFrame, percat: pd.DataFrame, macro: pd.DataFrame, diag: pd.DataFrame) -> None:
    tables = select_report_tables(macro, percat, diag)

    lines: List[str] = []

    lines.append("# Stage 9-A2 QCR-U Sanity Check Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage checks whether Stage 9-A1 QCR-U fusion can be used as a paper-level module.")
    lines.append("It reads existing Stage 9-A1 predictions only. It does not train models, rerun CLIP, or regenerate anomaly maps.")
    lines.append("")
    lines.append("## 2. Why This Check Is Necessary")
    lines.append("")
    lines.append("Stage 9-A1 shows strong QCR-U performance, but candidate_quality_only is also extremely strong.")
    lines.append("Therefore, the current result must be treated carefully: QCR-U may be a useful calibration module, but candidate quality may already encode most anomaly evidence.")
    lines.append("")
    lines.append("## 3. Output Files")
    lines.append("")
    lines.append(f"- `{OUT_PERCAT.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_MACRO.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_DIAG.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 4. Best QCR-U Macro Results")
    lines.append("")
    lines.append("| Backbone | Strategy | Eval mode | Fusion | Macro AUROC | Macro AP | Macro F1 | Positive categories | Negative categories | Mean ΔAUROC vs VLM |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")

    for _, r in tables["best_macro_qcr"].iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['fusion_method']} | "
            f"{r['macro_auroc']:.4f} | {r['macro_ap']:.4f} | {r['macro_best_f1']:.4f} | "
            f"{int(r['num_categories_delta_positive'])} | {int(r['num_categories_delta_negative'])} | "
            f"{r['mean_delta_auroc_vs_vlm']:.4f} |"
        )

    lines.append("")
    lines.append("## 5. Candidate-quality-only Macro Results")
    lines.append("")
    lines.append("| Backbone | Strategy | Eval mode | Macro AUROC | Mean ΔAUROC vs VLM |")
    lines.append("|---|---|---|---:|---:|")

    for _, r in tables["candidate_quality_macro"].iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
            f"{r['macro_auroc']:.4f} | {r['mean_delta_auroc_vs_vlm']:.4f} |"
        )

    lines.append("")
    lines.append("## 6. QCR-U Negative-delta Categories")
    lines.append("")
    lines.append("| Backbone | Strategy | Eval mode | Category | Fusion | AUROC | VLM AUROC | ΔAUROC |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")

    neg = tables["qcr_negative_categories"]
    if neg.empty:
        lines.append("| - | - | - | - | - | - | - | - |")
    else:
        for _, r in neg.iterrows():
            lines.append(
                f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['category']} | {r['fusion_method']} | "
                f"{r['auroc']:.4f} | {r['vlm_only_auroc']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
            )

    lines.append("")
    lines.append("## 7. Strongest Candidate-quality Separation")
    lines.append("")
    lines.append("| Backbone | Strategy | Eval mode | Category | Candidate rate normal | Candidate rate anomaly | Q normal | Q anomaly | corr(Q,y) |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")

    for _, r in tables["q_dominant_categories"].iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['category']} | "
            f"{r['candidate_rate_normal']:.4f} | {r['candidate_rate_anomaly']:.4f} | "
            f"{r['q_mean_normal']:.4f} | {r['q_mean_anomaly']:.4f} | {r['corr_q_y']:.4f} |"
        )

    lines.append("")
    lines.append("## 8. Decision Guidance")
    lines.append("")
    lines.append("Use the following rule for the next paper decision:")
    lines.append("")
    lines.append("| Condition | Decision |")
    lines.append("|---|---|")
    lines.append("| QCR-U improves macro AUROC over vlm_only across most categories and candidate_quality_only is not overwhelmingly dominant | Keep QCR-U as a main module |")
    lines.append("| QCR-U improves some settings, but candidate_quality_only is much stronger | Keep QCR-U as an ablation/calibration module, not the core contribution |")
    lines.append("| QCR-U often hurts per-category AUROC | Keep QCR-U only as diagnostic analysis |")
    lines.append("")
    lines.append("## 9. Current Conservative Interpretation")
    lines.append("")
    lines.append("Before Stage 9-A2 results are inspected, the safest interpretation is:")
    lines.append("")
    lines.append("```text")
    lines.append("QCR-U is a candidate calibration module that combines VLM abnormality, detector-region quality, and detector/VLM consistency.")
    lines.append("However, because candidate_quality_only is very strong in Stage 9-A1, the module should not yet be claimed as the main source of anomaly reasoning improvement.")
    lines.append("```")
    lines.append("")
    lines.append("## 10. Next Step")
    lines.append("")
    lines.append("After this report is committed, inspect whether QCR-U has stable macro-category gains.")
    lines.append("Then decide whether Stage 9-B should write QCR-U into the method section or keep it as an ablation/diagnostic module.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not IN_PRED.exists():
        raise FileNotFoundError(f"Missing Stage 9-A1 prediction file: {IN_PRED}")

    pred = pd.read_csv(IN_PRED)

    required_cols = [
        "backbone",
        "strategy",
        "eval_mode",
        "fusion_method",
        "category",
        "is_anomaly_final",
        "fusion_score",
    ]
    missing = [c for c in required_cols if c not in pred.columns]
    if missing:
        raise ValueError(f"Missing required columns in {IN_PRED}: {missing}")

    pred = pred[pred["fusion_method"].isin(KEY_METHODS)].copy()

    percat = compute_per_category(pred)
    macro = compute_macro(percat)
    diag = compute_signal_diagnostics(pred)

    percat.to_csv(OUT_PERCAT, index=False)
    macro.to_csv(OUT_MACRO, index=False)
    diag.to_csv(OUT_DIAG, index=False)
    write_report(pred, percat, macro, diag)

    print("[DONE]", OUT_PERCAT)
    print("[DONE]", OUT_MACRO)
    print("[DONE]", OUT_DIAG)
    print("[DONE]", OUT_REPORT)
    print("prediction_rows:", len(pred))
    print("per_category_rows:", len(percat))
    print("macro_rows:", len(macro))
    print("diagnostic_rows:", len(diag))

    print("\nTop QCR-U macro rows:")
    show = (
        macro[macro["fusion_method"].isin(["qcr_u_fixed", "qcr_u_detector_aware"])]
        .sort_values(["macro_auroc", "mean_delta_auroc_vs_vlm"], ascending=False)
        .head(12)
    )
    if not show.empty:
        print(
            show[
                [
                    "backbone",
                    "strategy",
                    "eval_mode",
                    "fusion_method",
                    "macro_auroc",
                    "macro_ap",
                    "macro_best_f1",
                    "num_categories_delta_positive",
                    "num_categories_delta_negative",
                    "mean_delta_auroc_vs_vlm",
                ]
            ].to_string(index=False)
        )

    print("\nCandidate-quality-only macro rows:")
    show_q = (
        macro[macro["fusion_method"] == "candidate_quality_only"]
        .sort_values("macro_auroc", ascending=False)
        .head(12)
    )
    if not show_q.empty:
        print(
            show_q[
                [
                    "backbone",
                    "strategy",
                    "eval_mode",
                    "macro_auroc",
                    "macro_ap",
                    "macro_best_f1",
                    "mean_delta_auroc_vs_vlm",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
