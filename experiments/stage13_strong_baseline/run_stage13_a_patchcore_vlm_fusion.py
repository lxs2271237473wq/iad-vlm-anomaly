from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


ROOT = Path(".").resolve()

IN_PRED = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv"

OUT_DIR = ROOT / "results/stage13_strong_baseline"
DOC_DIR = ROOT / "docs/stage13_strong_baseline"

OUT_GRID = OUT_DIR / "stage13_a_patchcore_vlm_fusion_grid.csv"
OUT_SUMMARY = OUT_DIR / "stage13_a_patchcore_vlm_fusion_summary.csv"
OUT_PER_CATEGORY = OUT_DIR / "stage13_a_patchcore_vlm_fusion_per_category.csv"
OUT_LOCO = OUT_DIR / "stage13_a_patchcore_vlm_fusion_loco_category.csv"
OUT_COMPLEMENT = OUT_DIR / "stage13_a_patchcore_vlm_score_complementarity.csv"
OUT_REPORT = DOC_DIR / "stage13_a_patchcore_vlm_fusion_report.md"

PRIMARY_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

PATCHCORE_COL = "patchcore_score"
FULL_COL = "full_image_score"
CONTEXT_COL = "context_topk_mean_score"
LABEL_COL = "gt_binary"


def f4(x):
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def safe_auroc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_ap(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def best_f1_acc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    precision, recall, _ = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_f1 = float(np.nanmax(f1))

    best_acc = -1.0
    best_thr = float(np.unique(s)[0])

    for thr in np.unique(s):
        pred = (s >= thr).astype(int)
        acc = float((pred == y).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    return best_f1, best_acc, best_thr


def metric_row(df, method, score_col, scope, category="", alpha="", fusion_pair="", protocol="direct"):
    y = df["y_true"].to_numpy(dtype=int)
    s = df[score_col].to_numpy(dtype=float)
    best_f1, best_acc, best_thr = best_f1_acc(y, s)

    return {
        "protocol": protocol,
        "scope": scope,
        "category": category,
        "method": method,
        "fusion_pair": fusion_pair,
        "alpha_patchcore": alpha,
        "num_images": int(len(df)),
        "num_normal": int((y == 0).sum()),
        "num_anomaly": int((y == 1).sum()),
        "auroc": safe_auroc(y, s),
        "ap": safe_ap(y, s),
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "best_threshold": best_thr,
    }


def zscore_by_category(df, cols):
    out = df.copy()

    for col in cols:
        zcol = f"{col}_z"

        def z(x):
            mu = x.mean()
            sigma = x.std(ddof=0)
            if sigma == 0 or pd.isna(sigma):
                return x * 0.0
            return (x - mu) / sigma

        out[zcol] = out.groupby("category")[col].transform(z)

    return out


def load_prediction_table():
    if not IN_PRED.exists():
        raise FileNotFoundError(IN_PRED)

    df = pd.read_csv(IN_PRED)

    required = [
        "category",
        "image_path",
        LABEL_COL,
        PATCHCORE_COL,
        FULL_COL,
        CONTEXT_COL,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nCurrent columns:\n"
            + "\n".join(df.columns)
        )

    df = df[df["category"].isin(PRIMARY_CATEGORIES)].copy()

    df["y_true"] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)
    for c in [PATCHCORE_COL, FULL_COL, CONTEXT_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["y_true", PATCHCORE_COL, FULL_COL, CONTEXT_COL]).copy()

    return df


def build_direct_summary(df):
    rows = []

    methods = [
        ("patchcore_score", PATCHCORE_COL),
        ("full_image_score", FULL_COL),
        ("context_topk_mean_score", CONTEXT_COL),
    ]

    for name, col in methods:
        rows.append(metric_row(df, name, col, "ALL_PRIMARY"))

    for category in PRIMARY_CATEGORIES:
        part = df[df["category"] == category].copy()
        for name, col in methods:
            rows.append(metric_row(part, name, col, "category", category=category))

    return pd.DataFrame(rows)


def build_fusion_grid(df):
    rows = []
    work = zscore_by_category(df, [PATCHCORE_COL, FULL_COL, CONTEXT_COL])
    alpha_values = np.round(np.linspace(0.0, 1.0, 101), 2)

    pairs = [
        ("patchcore_plus_full_image", FULL_COL),
        ("patchcore_plus_context", CONTEXT_COL),
    ]

    for pair_name, vlm_col in pairs:
        for alpha in alpha_values:
            fused_col = f"fused_{pair_name}_{alpha:.2f}"
            work[fused_col] = (
                alpha * work[f"{PATCHCORE_COL}_z"]
                + (1.0 - alpha) * work[f"{vlm_col}_z"]
            )

            rows.append(
                metric_row(
                    work,
                    fused_col,
                    fused_col,
                    "ALL_PRIMARY",
                    alpha=float(alpha),
                    fusion_pair=pair_name,
                    protocol="same_set_grid",
                )
            )

            for category in PRIMARY_CATEGORIES:
                part = work[work["category"] == category].copy()
                rows.append(
                    metric_row(
                        part,
                        fused_col,
                        fused_col,
                        "category",
                        category=category,
                        alpha=float(alpha),
                        fusion_pair=pair_name,
                        protocol="same_set_grid",
                    )
                )

    return pd.DataFrame(rows)


def select_best_fusions(df, grid):
    rows = build_direct_summary(df).to_dict("records")

    for pair_name in ["patchcore_plus_full_image", "patchcore_plus_context"]:
        part = grid[
            (grid["scope"] == "ALL_PRIMARY")
            & (grid["fusion_pair"] == pair_name)
            & (grid["protocol"] == "same_set_grid")
        ].copy()

        best = part.sort_values(["auroc", "ap"], ascending=False).iloc[0].to_dict()
        best["method"] = f"{pair_name}_best_same_set"
        rows.append(best)

    return pd.DataFrame(rows)


def build_per_category_best(grid):
    rows = []

    for category in PRIMARY_CATEGORIES:
        for pair_name in ["patchcore_plus_full_image", "patchcore_plus_context"]:
            part = grid[
                (grid["scope"] == "category")
                & (grid["category"] == category)
                & (grid["fusion_pair"] == pair_name)
                & (grid["protocol"] == "same_set_grid")
            ].copy()

            best = part.sort_values(["auroc", "ap"], ascending=False).iloc[0].to_dict()
            best["method"] = f"{pair_name}_best_same_set"
            rows.append(best)

    return pd.DataFrame(rows)


def build_loco(df):
    rows = []
    work = zscore_by_category(df, [PATCHCORE_COL, FULL_COL, CONTEXT_COL])
    alpha_values = np.round(np.linspace(0.0, 1.0, 101), 2)

    pairs = [
        ("patchcore_plus_full_image", FULL_COL),
        ("patchcore_plus_context", CONTEXT_COL),
    ]

    for held_out in PRIMARY_CATEGORIES:
        train = work[work["category"] != held_out].copy()
        test = work[work["category"] == held_out].copy()

        for pair_name, vlm_col in pairs:
            train_rows = []

            for alpha in alpha_values:
                tmp = train[["category", "image_path", "y_true"]].copy()
                tmp["score"] = (
                    alpha * train[f"{PATCHCORE_COL}_z"]
                    + (1.0 - alpha) * train[f"{vlm_col}_z"]
                )

                train_rows.append(
                    metric_row(
                        tmp,
                        f"{pair_name}_train_alpha_{alpha:.2f}",
                        "score",
                        "train_categories",
                        alpha=float(alpha),
                        fusion_pair=pair_name,
                        protocol="loco_train_select",
                    )
                )

            train_df = pd.DataFrame(train_rows)
            best_train = train_df.sort_values(["auroc", "ap"], ascending=False).iloc[0]
            alpha = float(best_train["alpha_patchcore"])

            tmp_test = test[["category", "image_path", "y_true"]].copy()
            tmp_test["score"] = (
                alpha * test[f"{PATCHCORE_COL}_z"]
                + (1.0 - alpha) * test[f"{vlm_col}_z"]
            )

            row = metric_row(
                tmp_test,
                f"{pair_name}_loco_selected_alpha",
                "score",
                "held_out_category",
                category=held_out,
                alpha=alpha,
                fusion_pair=pair_name,
                protocol="loco_test",
            )
            row["train_auroc"] = float(best_train["auroc"])
            row["train_ap"] = float(best_train["ap"])
            rows.append(row)

    loco = pd.DataFrame(rows)

    agg_rows = []

    for pair_name, vlm_col in pairs:
        parts = []

        for held_out in PRIMARY_CATEGORIES:
            selected = loco[
                (loco["fusion_pair"] == pair_name)
                & (loco["category"] == held_out)
            ].iloc[0]

            alpha = float(selected["alpha_patchcore"])
            test = work[work["category"] == held_out].copy()
            test["score"] = (
                alpha * test[f"{PATCHCORE_COL}_z"]
                + (1.0 - alpha) * test[f"{vlm_col}_z"]
            )
            parts.append(test[["category", "image_path", "y_true", "score"]])

        agg = pd.concat(parts, ignore_index=True)
        agg_rows.append(
            metric_row(
                agg,
                f"{pair_name}_loco_aggregate",
                "score",
                "ALL_PRIMARY",
                fusion_pair=pair_name,
                protocol="loco_aggregate",
            )
        )

    return pd.concat([loco, pd.DataFrame(agg_rows)], ignore_index=True)


def build_complementarity(df):
    rows = []

    for category in ["ALL_PRIMARY"] + PRIMARY_CATEGORIES:
        part = df.copy() if category == "ALL_PRIMARY" else df[df["category"] == category].copy()

        rows.append({
            "category": category,
            "num_images": int(len(part)),
            "patchcore_full_spearman": float(part[[PATCHCORE_COL, FULL_COL]].corr(method="spearman").iloc[0, 1]),
            "patchcore_context_spearman": float(part[[PATCHCORE_COL, CONTEXT_COL]].corr(method="spearman").iloc[0, 1]),
            "interpretation": "lower correlation suggests more complementarity; high correlation suggests repeated detector ranking",
        })

    return pd.DataFrame(rows)


def write_report(summary, per_category, loco, complement):
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    patch = summary[summary["method"] == "patchcore_score"].iloc[0]
    full = summary[summary["method"] == "full_image_score"].iloc[0]
    context = summary[summary["method"] == "context_topk_mean_score"].iloc[0]

    best_full = summary[summary["method"] == "patchcore_plus_full_image_best_same_set"].iloc[0]
    best_context = summary[summary["method"] == "patchcore_plus_context_best_same_set"].iloc[0]

    loco_agg = loco[loco["protocol"] == "loco_aggregate"].copy()
    loco_full = loco_agg[loco_agg["fusion_pair"] == "patchcore_plus_full_image"].iloc[0]
    loco_context = loco_agg[loco_agg["fusion_pair"] == "patchcore_plus_context"].iloc[0]

    same_set_delta = float(best_context["auroc"]) - float(patch["auroc"])
    loco_delta = float(loco_context["auroc"]) - float(patch["auroc"])

    lines = []

    lines.append("# Stage 13-A PatchCore and VLM Score Fusion")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage tests whether the context-aware VLM branch provides complementary information to the strong PatchCore detector.")
    lines.append("")
    lines.append("It does not train models, regenerate crops, or rerun VLM inference. It only fuses existing image-level scores from Stage 11-D.")
    lines.append("")
    lines.append("## 2. Direct Baselines on ALL_PRIMARY")
    lines.append("")
    lines.append("| Method | AUROC | AP | Best F1 | Best Acc |")
    lines.append("|---|---:|---:|---:|---:|")

    for _, r in summary[summary["method"].isin(["patchcore_score", "full_image_score", "context_topk_mean_score"])].iterrows():
        lines.append(
            f"| {r['method']} | {f4(r['auroc'])} | {f4(r['ap'])} | "
            f"{f4(r['best_f1'])} | {f4(r['best_accuracy'])} |"
        )

    lines.append("")
    lines.append("## 3. Same-set Fusion Result")
    lines.append("")
    lines.append("Same-set fusion searches the fusion weight on the same evaluation set. It is an upper-bound diagnostic and should not be overclaimed as fair final evidence.")
    lines.append("")
    lines.append("| Fusion | Best alpha for PatchCore | AUROC | Delta vs PatchCore | AP |")
    lines.append("|---|---:|---:|---:|---:|")

    for r in [best_full, best_context]:
        delta = float(r["auroc"]) - float(patch["auroc"])
        lines.append(
            f"| {r['fusion_pair']} | {f4(r['alpha_patchcore'])} | "
            f"{f4(r['auroc'])} | {f4(delta)} | {f4(r['ap'])} |"
        )

    lines.append("")
    lines.append("## 4. Leave-one-category-out Fusion")
    lines.append("")
    lines.append("This protocol selects the fusion weight on three categories and evaluates on the held-out category.")
    lines.append("")
    lines.append("| Category / Aggregate | Fusion | Alpha for PatchCore | AUROC | AP | Train AUROC |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for _, r in loco.iterrows():
        label = r["category"] if str(r["category"]) != "" else r["scope"]
        lines.append(
            f"| {label} | {r['fusion_pair']} | {f4(r['alpha_patchcore'])} | "
            f"{f4(r['auroc'])} | {f4(r['ap'])} | {f4(r.get('train_auroc', ''))} |"
        )

    lines.append("")
    lines.append("## 5. Per-category Same-set Best Fusion")
    lines.append("")
    lines.append("| Category | Fusion | Alpha for PatchCore | AUROC | AP |")
    lines.append("|---|---|---:|---:|---:|")

    for _, r in per_category.iterrows():
        lines.append(
            f"| {r['category']} | {r['fusion_pair']} | {f4(r['alpha_patchcore'])} | "
            f"{f4(r['auroc'])} | {f4(r['ap'])} |"
        )

    lines.append("")
    lines.append("## 6. Score Complementarity")
    lines.append("")
    lines.append("| Category | PatchCore vs full-image correlation | PatchCore vs context correlation |")
    lines.append("|---|---:|---:|")

    for _, r in complement.iterrows():
        lines.append(
            f"| {r['category']} | {f4(r['patchcore_full_spearman'])} | "
            f"{f4(r['patchcore_context_spearman'])} |"
        )

    lines.append("")
    lines.append("## 7. Decision")
    lines.append("")

    if same_set_delta > 0:
        lines.append(f"Same-set PatchCore+context fusion improves over PatchCore by {f4(same_set_delta)} AUROC.")
    else:
        lines.append(f"Same-set PatchCore+context fusion does not improve over PatchCore; delta is {f4(same_set_delta)} AUROC.")

    if loco_delta > 0:
        lines.append(f"Leave-one-category-out PatchCore+context fusion improves over PatchCore by {f4(loco_delta)} AUROC.")
    else:
        lines.append(f"Leave-one-category-out PatchCore+context fusion does not improve over PatchCore; delta is {f4(loco_delta)} AUROC.")

    lines.append("")
    lines.append("If fusion does not improve over PatchCore, the VLM branch should not be claimed as a detector-strength improvement. The next analysis should move to PatchCore error cases and uncertain samples.")
    lines.append("")
    lines.append("## 8. Output Files")
    lines.append("")
    lines.append(f"- Fusion grid: `{OUT_GRID.relative_to(ROOT)}`")
    lines.append(f"- Fusion summary: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Per-category fusion: `{OUT_PER_CATEGORY.relative_to(ROOT)}`")
    lines.append(f"- Leave-one-category-out fusion: `{OUT_LOCO.relative_to(ROOT)}`")
    lines.append(f"- Complementarity table: `{OUT_COMPLEMENT.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    df = load_prediction_table()

    grid = build_fusion_grid(df)
    summary = select_best_fusions(df, grid)
    per_category = build_per_category_best(grid)
    loco = build_loco(df)
    complement = build_complementarity(df)

    grid.to_csv(OUT_GRID, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    per_category.to_csv(OUT_PER_CATEGORY, index=False)
    loco.to_csv(OUT_LOCO, index=False)
    complement.to_csv(OUT_COMPLEMENT, index=False)

    write_report(summary, per_category, loco, complement)

    print("[DONE]", OUT_GRID)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_PER_CATEGORY)
    print("[DONE]", OUT_LOCO)
    print("[DONE]", OUT_COMPLEMENT)
    print("[DONE]", OUT_REPORT)

    print("\n===== Fusion summary =====")
    print(summary.to_string(index=False))

    print("\n===== Leave-one-category-out =====")
    print(loco.to_string(index=False))

    print("\n===== Complementarity =====")
    print(complement.to_string(index=False))


if __name__ == "__main__":
    main()
