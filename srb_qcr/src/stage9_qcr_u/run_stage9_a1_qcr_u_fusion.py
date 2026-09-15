from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()
OUT_DIR = ROOT / "results" / "stage9_qcr_u"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED = OUT_DIR / "stage9_a1_qcr_u_fusion_predictions.csv"
OUT_SUMMARY = OUT_DIR / "stage9_a1_qcr_u_fusion_summary.csv"
OUT_REPORT = OUT_DIR / "stage9_a1_qcr_u_fusion_report.md"


BACKBONE_CONFIGS = [
    {
        "backbone": "PatchCore",
        "detector_root": ROOT / "results" / "stage7_generalization" / "visa_patchcore" / "VisA",
        "vlm_predictions": ROOT / "results" / "stage7_generalization" / "visa_binary_prompt_reasoning" / "visa_binary_prompt_predictions.csv",
        "detector_prediction_name": "patchcore_image_predictions.csv",
    },
    {
        "backbone": "FastFlow",
        "detector_root": ROOT / "results" / "stage7_generalization" / "visa_multibackbone" / "fastflow_12cls" / "VisA",
        "vlm_predictions": ROOT / "results" / "stage7_generalization" / "visa_multibackbone" / "fastflow_binary_prompt_reasoning" / "visa_binary_prompt_predictions.csv",
        "detector_prediction_name": "fastflow_image_predictions.csv",
    },
]


FUSION_WEIGHTS = {
    "vlm_only": {"M": 1.00, "Q": 0.00, "K": 0.00, "D": 0.00},
    "detector_only": {"M": 0.00, "Q": 0.00, "K": 0.00, "D": 1.00},
    "candidate_quality_only": {"M": 0.00, "Q": 1.00, "K": 0.00, "D": 0.00},
    "vlm_detector_avg": {"M": 0.70, "Q": 0.00, "K": 0.00, "D": 0.30},
    "qcr_u_fixed": {"M": 0.65, "Q": 0.20, "K": 0.15, "D": 0.00},
    "qcr_u_detector_aware": {"M": 0.55, "Q": 0.20, "K": 0.15, "D": 0.10},
}


def canonicalize_path(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\\", "/")
    for anchor in ["datasets/VisA/", "datasets/MVTecAD/", "datasets/"]:
        idx = text.find(anchor)
        if idx >= 0:
            return text[idx:]
    return text.lstrip("./")


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


def minmax(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)

    lo = float(valid.min())
    hi = float(valid.max())
    if math.isclose(lo, hi):
        return pd.Series(np.full(len(x), 0.5), index=x.index, dtype=float)

    return ((x - lo) / (hi - lo)).fillna(0.0)


def add_group_minmax(df: pd.DataFrame, value_col: str, out_col: str, group_cols: List[str]) -> pd.DataFrame:
    parts = []
    for _, part in df.groupby(group_cols, dropna=False, sort=False):
        part = part.copy()
        part[out_col] = minmax(part[value_col])
        parts.append(part)

    if not parts:
        df[out_col] = []
        return df

    return pd.concat(parts, ignore_index=True)


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


def read_candidate_quality(detector_root: Path) -> pd.DataFrame:
    files = sorted(detector_root.glob("*/candidate_regions.csv"))
    frames = []

    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue

        df = df.copy()
        df["source_candidate_csv"] = str(path.relative_to(ROOT))
        df["image_key"] = df.get("canonical_image_path", df["image_path"]).map(canonicalize_path)

        for col in ["mean_score", "max_score", "area"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["log_area"] = np.log1p(df["area"].clip(lower=0))

        df = add_group_minmax(df, "mean_score", "mean_score_norm", ["category"])
        df = add_group_minmax(df, "max_score", "max_score_norm", ["category"])
        df = add_group_minmax(df, "log_area", "area_norm", ["category"])

        df["candidate_quality_raw"] = (
            0.45 * df["mean_score_norm"]
            + 0.35 * df["max_score_norm"]
            + 0.20 * df["area_norm"]
        )

        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["category", "image_key", "candidate_quality_norm", "num_candidates"]
        )

    all_cands = pd.concat(frames, ignore_index=True)

    agg = (
        all_cands.groupby(["category", "image_key"], as_index=False)
        .agg(
            candidate_quality=("candidate_quality_raw", "max"),
            candidate_mean_score_max=("mean_score", "max"),
            candidate_max_score_max=("max_score", "max"),
            candidate_area_max=("area", "max"),
            num_candidates=("component_rank", "count"),
        )
    )

    agg = add_group_minmax(agg, "candidate_quality", "candidate_quality_norm", ["category"])
    return agg


def read_detector_predictions(detector_root: Path, prediction_name: str) -> pd.DataFrame:
    files = sorted(detector_root.glob(f"*/{prediction_name}"))
    frames = []

    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue

        df = df.copy()
        df["source_detector_csv"] = str(path.relative_to(ROOT))
        df["image_key"] = df.get("canonical_image_path", df["image_path"]).map(canonicalize_path)

        if "image_score" not in df.columns:
            raise ValueError(f"Missing image_score in {path}")

        df["detector_image_score"] = pd.to_numeric(df["image_score"], errors="coerce").fillna(0.0)

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["category", "image_key", "detector_image_score"])

    det = pd.concat(frames, ignore_index=True)

    keep_cols = [
        "dataset",
        "category",
        "image_key",
        "image_path",
        "canonical_image_path",
        "label",
        "is_anomaly",
        "detector_image_score",
        "image_threshold",
        "pred_is_anomaly",
        "image_correct",
        "source_detector_csv",
    ]
    keep_cols = [c for c in keep_cols if c in det.columns]

    det = det[keep_cols].drop_duplicates(subset=["category", "image_key"])
    det = add_group_minmax(det, "detector_image_score", "detector_score_norm", ["category"])

    return det


def read_vlm_predictions(path: Path, backbone: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty VLM prediction file: {path}")

    df = df.copy()
    df["backbone"] = backbone
    df["source_vlm_csv"] = str(path.relative_to(ROOT))
    df["image_key"] = df.get("canonical_image_path", df["image_path"]).map(canonicalize_path)

    if "vlm_anomaly_score" not in df.columns:
        raise ValueError(f"Missing vlm_anomaly_score in {path}")

    df["vlm_anomaly_score"] = pd.to_numeric(df["vlm_anomaly_score"], errors="coerce").fillna(0.0)

    if "strategy" not in df.columns:
        df["strategy"] = "unknown_strategy"
    if "eval_mode" not in df.columns:
        df["eval_mode"] = "unknown_eval_mode"
    if "used_mode" not in df.columns:
        df["used_mode"] = "unknown_used_mode"

    return df


def build_backbone_table(config: Dict[str, object]) -> pd.DataFrame:
    backbone = str(config["backbone"])
    detector_root = Path(config["detector_root"])
    vlm_path = Path(config["vlm_predictions"])
    prediction_name = str(config["detector_prediction_name"])

    candidates = read_candidate_quality(detector_root)
    detector = read_detector_predictions(detector_root, prediction_name)
    vlm = read_vlm_predictions(vlm_path, backbone)

    merged = vlm.merge(
        detector,
        on=["category", "image_key"],
        how="left",
        suffixes=("_vlm", "_detector"),
    )

    merged = merged.merge(
        candidates,
        on=["category", "image_key"],
        how="left",
    )

    if "dataset" not in merged.columns:
        if "dataset_vlm" in merged.columns:
            merged["dataset"] = merged["dataset_vlm"]
        elif "dataset_detector" in merged.columns:
            merged["dataset"] = merged["dataset_detector"]
        else:
            merged["dataset"] = "VisA"

    if "is_anomaly_vlm" in merged.columns:
        merged["is_anomaly_final"] = to_binary_series(merged["is_anomaly_vlm"])
    elif "is_anomaly" in merged.columns:
        merged["is_anomaly_final"] = to_binary_series(merged["is_anomaly"])
    elif "is_anomaly_detector" in merged.columns:
        merged["is_anomaly_final"] = to_binary_series(merged["is_anomaly_detector"])
    else:
        raise ValueError(f"No is_anomaly column after merge for {backbone}")

    for col in [
        "detector_image_score",
        "detector_score_norm",
        "candidate_quality",
        "candidate_quality_norm",
        "num_candidates",
    ]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["has_candidate"] = merged["num_candidates"] > 0

    merged = add_group_minmax(
        merged,
        "vlm_anomaly_score",
        "vlm_score_norm",
        ["backbone", "strategy", "eval_mode", "category"],
    )

    merged["high_high_consistency"] = (
        (1.0 - (merged["vlm_score_norm"] - merged["detector_score_norm"]).abs()).clip(0.0, 1.0)
        * (0.5 * (merged["vlm_score_norm"] + merged["detector_score_norm"]))
    )

    return merged


def add_fusion_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    base_cols = [
        "backbone",
        "dataset",
        "category",
        "strategy",
        "eval_mode",
        "used_mode",
        "image_key",
        "is_anomaly_final",
        "fallback",
        "has_candidate",
        "num_candidates",
        "vlm_anomaly_score",
        "vlm_score_norm",
        "detector_image_score",
        "detector_score_norm",
        "candidate_quality_norm",
        "high_high_consistency",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    for method, weights in FUSION_WEIGHTS.items():
        part = df[base_cols].copy()
        part["fusion_method"] = method

        part["fusion_score"] = (
            weights["M"] * df["vlm_score_norm"]
            + weights["Q"] * df["candidate_quality_norm"]
            + weights["K"] * df["high_high_consistency"]
            + weights["D"] * df["detector_score_norm"]
        )

        part["weight_M_vlm"] = weights["M"]
        part["weight_Q_candidate"] = weights["Q"]
        part["weight_K_consistency"] = weights["K"]
        part["weight_D_detector"] = weights["D"]

        rows.append(part)

    return pd.concat(rows, ignore_index=True)


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["backbone", "strategy", "eval_mode", "fusion_method"]

    for keys, part in pred.groupby(group_cols, dropna=False, sort=False):
        backbone, strategy, eval_mode, method = keys

        y = to_binary_series(part["is_anomaly_final"]).to_numpy()
        s = pd.to_numeric(part["fusion_score"], errors="coerce").fillna(0.0).to_numpy()

        f1, acc, thr = best_f1_accuracy(y, s)

        rows.append(
            {
                "backbone": backbone,
                "strategy": strategy,
                "eval_mode": eval_mode,
                "fusion_method": method,
                "num_images": int(len(part)),
                "num_normal": int((y == 0).sum()),
                "num_anomaly": int((y == 1).sum()),
                "coverage_ratio": float(to_binary_series(part["has_candidate"]).mean())
                if "has_candidate" in part.columns
                else float("nan"),
                "auroc": binary_auroc(y, s),
                "ap": average_precision(y, s),
                "best_f1": f1,
                "best_accuracy": acc,
                "best_threshold": thr,
            }
        )

    summary = pd.DataFrame(rows)

    baseline = summary[summary["fusion_method"] == "vlm_only"][
        ["backbone", "strategy", "eval_mode", "auroc", "ap", "best_f1", "best_accuracy"]
    ].rename(
        columns={
            "auroc": "vlm_only_auroc",
            "ap": "vlm_only_ap",
            "best_f1": "vlm_only_best_f1",
            "best_accuracy": "vlm_only_best_accuracy",
        }
    )

    summary = summary.merge(baseline, on=["backbone", "strategy", "eval_mode"], how="left")

    summary["delta_auroc_vs_vlm"] = summary["auroc"] - summary["vlm_only_auroc"]
    summary["delta_ap_vs_vlm"] = summary["ap"] - summary["vlm_only_ap"]
    summary["delta_best_f1_vs_vlm"] = summary["best_f1"] - summary["vlm_only_best_f1"]
    summary["delta_accuracy_vs_vlm"] = summary["best_accuracy"] - summary["vlm_only_best_accuracy"]

    return summary.sort_values(
        ["backbone", "strategy", "eval_mode", "auroc"],
        ascending=[True, True, True, False],
    )


def write_report(summary: pd.DataFrame, pred: pd.DataFrame) -> None:
    best = (
        summary.sort_values(["backbone", "auroc"], ascending=[True, False])
        .groupby("backbone", as_index=False)
        .head(5)
    )

    qcr = summary[summary["fusion_method"].str.contains("qcr_u", regex=False)].copy()

    qcr_best = (
        qcr.sort_values(["backbone", "auroc"], ascending=[True, False])
        .groupby("backbone", as_index=False)
        .head(3)
    )

    lines = []
    lines.append("# Stage 9-A1 QCR-U Fusion Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage implements fixed-weight QCR-U fusion on existing VisA predictions.")
    lines.append("It reads existing detector predictions, candidate regions, and VLM binary prompt scores.")
    lines.append("It does not train models, rerun CLIP, or regenerate anomaly maps.")
    lines.append("")
    lines.append("## 2. Score Definition")
    lines.append("")
    lines.append("```text")
    lines.append("M = normalized VLM anomaly score")
    lines.append("Q = normalized candidate-region quality")
    lines.append("D = normalized detector image score")
    lines.append("K = high-high detector/VLM consistency")
    lines.append("F_qcr = alpha * M + beta * Q + gamma * K")
    lines.append("F_qcr_detector_aware = alpha * M + beta * Q + gamma * K + delta * D")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Output Files")
    lines.append("")
    lines.append(f"- `{OUT_PRED.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 4. Best Overall Rows")
    lines.append("")
    lines.append("| Backbone | Strategy | Eval mode | Fusion | AUROC | AP | Best F1 | Delta AUROC vs VLM |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")

    for _, r in best.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['fusion_method']} | "
            f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
        )

    lines.append("")
    lines.append("## 5. Best QCR-U Rows")
    lines.append("")
    lines.append("| Backbone | Strategy | Eval mode | Fusion | AUROC | AP | Best F1 | Delta AUROC vs VLM |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")

    for _, r in qcr_best.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | {r['fusion_method']} | "
            f"{r['auroc']:.4f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['delta_auroc_vs_vlm']:.4f} |"
        )

    lines.append("")
    lines.append("## 6. Interpretation Boundary")
    lines.append("")
    lines.append("- These are fixed-weight fusion results, not supervised grid-search results.")
    lines.append("- Positive delta means Q/K/D adds useful signal beyond VLM score alone.")
    lines.append("- Negative delta means the current QCR-U construction should remain diagnostic evidence rather than a main performance claim.")
    lines.append("- Since labels are not used to tune weights, this is safer than an oracle weight search.")
    lines.append("")
    lines.append("## 7. Next Step")
    lines.append("")
    lines.append("Stage 9-A2 should decide whether QCR-U is kept as a main module, an ablation module, or a diagnostic analysis.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tables = []

    for config in BACKBONE_CONFIGS:
        print(f"[LOAD] {config['backbone']}")
        tables.append(build_backbone_table(config))

    base = pd.concat(tables, ignore_index=True)
    pred = add_fusion_scores(base)
    summary = summarize(pred)

    pred.to_csv(OUT_PRED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(summary, pred)

    print("[DONE]", OUT_PRED)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print("prediction_rows:", len(pred))
    print("summary_rows:", len(summary))

    print("\nTop QCR-U rows:")
    show = (
        summary[summary["fusion_method"].str.contains("qcr_u", regex=False)]
        .sort_values("auroc", ascending=False)
        .head(10)
    )
    print(
        show[
            [
                "backbone",
                "strategy",
                "eval_mode",
                "fusion_method",
                "auroc",
                "ap",
                "best_f1",
                "delta_auroc_vs_vlm",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
