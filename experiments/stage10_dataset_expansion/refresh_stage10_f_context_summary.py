from pathlib import Path
from datetime import datetime
import math
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

PRED_PATH = ROOT / "results/stage10_dataset_expansion/stage10_f_multiscale_context_image_predictions.csv"
SUMMARY_PATH = ROOT / "results/stage10_dataset_expansion/stage10_f_multiscale_context_summary.csv"
REPORT_PATH = ROOT / "docs/stage10_dataset_expansion/stage10_f_multiscale_context_crop_report.md"


def binary_auroc(y_true, scores):
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
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true, scores):
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
    return float(precision[y_sorted == 1].sum() / n_pos)


def best_f1_accuracy(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")

    best_f1 = -1.0
    best_acc = -1.0
    best_thr = float(s[0])

    for thr in np.unique(s):
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


def add_metric_row(rows, pred, method, score_col):
    if score_col not in pred.columns:
        return

    part = pred.drop_duplicates("image_path").copy()
    y = part["gt_binary"].astype(int).to_numpy()
    s = pd.to_numeric(part[score_col], errors="coerce").fillna(0.0).to_numpy()

    f1, acc, thr = best_f1_accuracy(y, s)

    rows.append({
        "dataset": "MVTec AD 2",
        "category": "vial",
        "method": method,
        "num_images": int(len(part)),
        "num_normal": int((y == 0).sum()),
        "num_anomaly": int((y == 1).sum()),
        "auroc": binary_auroc(y, s),
        "ap": average_precision(y, s),
        "best_f1": f1,
        "best_accuracy": acc,
        "best_threshold": thr,
    })


def add_context_metric_rows(rows, pred):
    context_names = sorted(pred["context_name"].dropna().unique().tolist())

    for context_name in context_names:
        part = pred[pred["context_name"] == context_name].drop_duplicates("image_path").copy()

        for method_suffix, score_col in [
            ("top1", "context_top1_score"),
            ("topk_max", "context_topk_max_score"),
            ("topk_mean", "context_topk_mean_score"),
        ]:
            if score_col not in part.columns:
                continue

            y = part["gt_binary"].astype(int).to_numpy()
            s = pd.to_numeric(part[score_col], errors="coerce").fillna(0.0).to_numpy()

            f1, acc, thr = best_f1_accuracy(y, s)

            rows.append({
                "dataset": "MVTec AD 2",
                "category": "vial",
                "method": f"{context_name}_{method_suffix}",
                "num_images": int(len(part)),
                "num_normal": int((y == 0).sum()),
                "num_anomaly": int((y == 1).sum()),
                "auroc": binary_auroc(y, s),
                "ap": average_precision(y, s),
                "best_f1": f1,
                "best_accuracy": acc,
                "best_threshold": thr,
            })


def write_report(summary):
    lines = []
    lines.append("# Stage 10-F Multiscale Context Crop Diagnostic")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("Stage 10-F2 refreshes the multiscale context summary from existing image-level prediction files.")
    lines.append("It does not rerun CLIP, regenerate crops, train models, or modify datasets.")
    lines.append("")
    lines.append("## 2. Output Files")
    lines.append("")
    lines.append(f"- Summary: `{SUMMARY_PATH.relative_to(ROOT)}`")
    lines.append(f"- Report: `{REPORT_PATH.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 3. Summary")
    lines.append("")
    lines.append("| Method | Images | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {int(r['num_images'])} | {float(r['auroc']):.4f} | "
            f"{float(r['ap']):.4f} | {float(r['best_f1']):.4f} | "
            f"{float(r['best_accuracy']):.4f} | {float(r['delta_auroc_vs_full']):.4f} |"
        )

    lines.append("")
    lines.append("## 4. Decision Rule")
    lines.append("")
    lines.append("- If any context-crop method exceeds full_image, keep MVTec AD 2 vial as positive evidence for context-aware localization-guided VLM reasoning.")
    lines.append("- If all context-crop methods remain below full_image, treat AD2/vial as a negative case and test another AD2 category or MVTec LOCO AD.")
    lines.append("- PatchCore score is a detector reference, not VLM reasoning evidence.")
    lines.append("")
    lines.append(f"<!-- stage10_f2_refreshed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')} -->")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(PRED_PATH)

    pred = pd.read_csv(PRED_PATH)

    required = ["image_path", "context_name", "gt_binary", "context_top1_score", "context_topk_max_score", "context_topk_mean_score"]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("[INFO] prediction rows:", len(pred))
    print("[INFO] unique images:", pred["image_path"].nunique())
    print("[INFO] contexts:", sorted(pred["context_name"].dropna().unique().tolist()))

    rows = []

    add_metric_row(rows, pred, "full_image", "full_image_score")
    add_metric_row(rows, pred, "stage10e_crop_top1", "crop_top1_score")
    add_metric_row(rows, pred, "stage10e_crop_topk_max", "crop_topk_max_score")
    add_metric_row(rows, pred, "stage10e_crop_topk_mean", "crop_topk_mean_score")
    add_metric_row(rows, pred, "patchcore_score", "patchcore_pred_score")

    add_context_metric_rows(rows, pred)

    summary = pd.DataFrame(rows)

    if "full_image" not in set(summary["method"]):
        raise RuntimeError("full_image row missing; cannot compute delta.")

    base = summary[summary["method"] == "full_image"].iloc[0]
    summary["delta_auroc_vs_full"] = summary["auroc"] - float(base["auroc"])
    summary["delta_ap_vs_full"] = summary["ap"] - float(base["ap"])
    summary["delta_best_f1_vs_full"] = summary["best_f1"] - float(base["best_f1"])
    summary["delta_accuracy_vs_full"] = summary["best_accuracy"] - float(base["best_accuracy"])

    summary = summary.sort_values("auroc", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_report(summary)

    ctx = summary[
        summary["method"].str.contains("context_", regex=False)
        | summary["method"].str.contains("square_context", regex=False)
    ]

    print("[DONE]", SUMMARY_PATH)
    print("[DONE]", REPORT_PATH)
    print("")
    print(summary.to_string(index=False))
    print("")
    print("===== CONTEXT METHODS =====")
    print(ctx.to_string(index=False))

    if ctx.empty:
        raise RuntimeError("Context methods are still missing after F2 refresh.")


if __name__ == "__main__":
    main()
