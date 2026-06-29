from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


ROOT = Path(".").resolve()

IN_PRED = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv"
IN_STAGE11_MAIN = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv"
IN_STAGE11_METHOD = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_method_table.csv"

OUT_DIR = ROOT / "results/stage13_strong_baseline"
DOC_DIR = ROOT / "docs/stage13_strong_baseline"

OUT_GRID = OUT_DIR / "stage13_a_patchcore_vlm_fusion_grid.csv"
OUT_SUMMARY = OUT_DIR / "stage13_a_patchcore_vlm_fusion_summary.csv"
OUT_PER_CATEGORY = OUT_DIR / "stage13_a_patchcore_vlm_fusion_per_category.csv"
OUT_LOCO = OUT_DIR / "stage13_a_patchcore_vlm_fusion_loco_category.csv"
OUT_COMPLEMENT = OUT_DIR / "stage13_a_patchcore_vlm_score_complementarity.csv"
OUT_REPORT = DOC_DIR / "stage13_a_patchcore_vlm_fusion_report.md"


PRIMARY_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

PATCHCORE_METHOD = "patchcore_score"
FULL_METHOD = "full_image"
CONTEXT_METHOD = "context_1.50_topk_mean"


def f4(x) -> str:
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def infer_score_column(df: pd.DataFrame) -> str:
    candidates = [
        "score",
        "anomaly_score",
        "image_score",
        "prediction_score",
        "pred_score",
        "vlm_score",
        "method_score",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in {"gt_label", "label", "y_true", "target", "is_anomaly"}
    ]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        "Cannot infer score column. Columns are:\n"
        + "\n".join(df.columns)
    )


def infer_label_column(df: pd.DataFrame) -> str:
    candidates = ["gt_label", "label", "y_true", "target", "is_anomaly"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Cannot infer label column. Columns are:\n"
        + "\n".join(df.columns)
    )


def normalize_label(v) -> int:
    if pd.isna(v):
        raise ValueError("NaN label found.")

    if isinstance(v, (bool, np.bool_)):
        return int(v)

    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(v > 0)

    s = str(v).strip().lower()

    normal_values = {
        "0",
        "false",
        "normal",
        "good",
        "ok",
        "negative",
        "neg",
    }
    anomaly_values = {
        "1",
        "true",
        "anomaly",
        "abnormal",
        "bad",
        "defect",
        "defective",
        "positive",
        "pos",
    }

    if s in normal_values:
        return 0
    if s in anomaly_values:
        return 1

    raise ValueError(f"Cannot normalize label value: {v!r}")


def safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def best_f1_acc(y_true: np.ndarray, score: np.ndarray) -> Tuple[float, float, float]:
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_f1 = float(np.nanmax(f1))

    candidate_thresholds = np.unique(score)
    best_acc = -1.0
    best_thr = float(candidate_thresholds[0])

    for thr in candidate_thresholds:
        pred = (score >= thr).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    return best_f1, best_acc, best_thr


def metric_row(
    df: pd.DataFrame,
    method: str,
    score_col: str,
    scope: str,
    category: str | None = None,
    alpha: float | None = None,
    fusion_pair: str | None = None,
    protocol: str = "direct",
) -> Dict[str, object]:
    y = df["y_true"].to_numpy(dtype=int)
    s = df[score_col].to_numpy(dtype=float)

    best_f1, best_acc, best_thr = best_f1_acc(y, s)

    return {
        "protocol": protocol,
        "scope": scope,
        "category": "" if category is None else category,
        "method": method,
        "fusion_pair": "" if fusion_pair is None else fusion_pair,
        "alpha_patchcore": "" if alpha is None else float(alpha),
        "num_images": int(len(df)),
        "num_normal": int((y == 0).sum()),
        "num_anomaly": int((y == 1).sum()),
        "auroc": safe_auroc(y, s),
        "ap": safe_ap(y, s),
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "best_threshold": best_thr,
    }


def zscore_by_category(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for col in score_cols:
        zcol = f"{col}_z"

        def zscore(x: pd.Series) -> pd.Series:
            mu = x.mean()
            sigma = x.std(ddof=0)
            if sigma == 0 or pd.isna(sigma):
                return x * 0.0
            return (x - mu) / sigma

        out[zcol] = out.groupby("category")[col].transform(zscore)

    return out


def load_prediction_table() -> pd.DataFrame:
    require_file(IN_PRED)

    raw = pd.read_csv(IN_PRED)

    required = ["category", "method", "image_path"]
    for c in required:
        if c not in raw.columns:
            raise ValueError(f"Required column `{c}` not found. Columns: {list(raw.columns)}")

    score_col = infer_score_column(raw)
    label_col = infer_label_column(raw)

    raw = raw.copy()
    raw["y_true"] = raw[label_col].map(normalize_label)
    raw[score_col] = pd.to_numeric(raw[score_col], errors="coerce")

    raw = raw[raw["category"].isin(PRIMARY_CATEGORIES)].copy()

    id_cols = ["category", "image_path", "y_true"]
    pivot = raw.pivot_table(
        index=id_cols,
        columns="method",
        values=score_col,
        aggfunc="mean",
    ).reset_index()

    missing_methods = [
        m for m in [PATCHCORE_METHOD, FULL_METHOD, CONTEXT_METHOD]
        if m not in pivot.columns
    ]
    if missing_methods:
        available = sorted([c for c in pivot.columns if c not in id_cols])
        raise ValueError(
            "Missing required methods: "
            + ", ".join(missing_methods)
            + "\nAvailable methods:\n"
            + "\n".join(available)
        )

    pivot = pivot.dropna(subset=[PATCHCORE_METHOD, FULL_METHOD, CONTEXT_METHOD]).copy()

    return pivot


def build_direct_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    direct_methods = [
        PATCHCORE_METHOD,
        FULL_METHOD,
        CONTEXT_METHOD,
    ]

    for method in direct_methods:
        rows.append(metric_row(df, method, method, "ALL_PRIMARY"))

    for category in PRIMARY_CATEGORIES:
        part = df[df["category"] == category].copy()
        for method in direct_methods:
            rows.append(metric_row(part, method, method, "category", category=category))

    return pd.DataFrame(rows)


def build_fusion_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    work = zscore_by_category(
        df,
        [PATCHCORE_METHOD, FULL_METHOD, CONTEXT_METHOD],
    )

    alpha_values = np.round(np.linspace(0.0, 1.0, 101), 2)

    fusion_pairs = [
        ("patchcore_plus_full_image", FULL_METHOD),
        ("patchcore_plus_context", CONTEXT_METHOD),
    ]

    for pair_name, vlm_col in fusion_pairs:
        for alpha in alpha_values:
            fused_col = f"fused_{pair_name}_{alpha:.2f}"
            work[fused_col] = (
                alpha * work[f"{PATCHCORE_METHOD}_z"]
                + (1.0 - alpha) * work[f"{vlm_col}_z"]
            )

            rows.append(
                metric_row(
                    work,
                    method=fused_col,
                    score_col=fused_col,
                    scope="ALL_PRIMARY",
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
                        method=fused_col,
                        score_col=fused_col,
                        scope="category",
                        category=category,
                        alpha=float(alpha),
                        fusion_pair=pair_name,
                        protocol="same_set_grid",
                    )
                )

    return pd.DataFrame(rows)


def select_best_fusions(grid: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    direct = build_direct_summary(load_prediction_table())
    for _, r in direct.iterrows():
        rows.append(r.to_dict())

    for pair_name in ["patchcore_plus_full_image", "patchcore_plus_context"]:
        part = grid[
            (grid["protocol"] == "same_set_grid")
            & (grid["scope"] == "ALL_PRIMARY")
            & (grid["fusion_pair"] == pair_name)
        ].copy()

        best = part.sort_values(["auroc", "ap"], ascending=False).iloc[0].to_dict()
        best["method"] = f"{pair_name}_best_same_set"
        rows.append(best)

    return pd.DataFrame(rows)


def build_per_category_best(grid: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for category in PRIMARY_CATEGORIES:
        for pair_name in ["patchcore_plus_full_image", "patchcore_plus_context"]:
            part = grid[
                (grid["protocol"] == "same_set_grid")
                & (grid["scope"] == "category")
                & (grid["category"] == category)
                & (grid["fusion_pair"] == pair_name)
            ].copy()

            if part.empty:
                continue

            best = part.sort_values(["auroc", "ap"], ascending=False).iloc[0].to_dict()
            best["method"] = f"{pair_name}_best_same_set"
            rows.append(best)

    return pd.DataFrame(rows)


def build_loco(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    work = zscore_by_category(
        df,
        [PATCHCORE_METHOD, FULL_METHOD, CONTEXT_METHOD],
    )

    alpha_values = np.round(np.linspace(0.0, 1.0, 101), 2)

    fusion_pairs = [
        ("patchcore_plus_full_image", FULL_METHOD),
        ("patchcore_plus_context", CONTEXT_METHOD),
    ]

    for held_out in PRIMARY_CATEGORIES:
        train = work[work["category"] != held_out].copy()
        test = work[work["category"] == held_out].copy()

        for pair_name, vlm_col in fusion_pairs:
            train_rows = []

            for alpha in alpha_values:
                score = (
                    alpha * train[f"{PATCHCORE_METHOD}_z"]
                    + (1.0 - alpha) * train[f"{vlm_col}_z"]
                )

                tmp = train[["category", "image_path", "y_true"]].copy()
                tmp["score"] = score.to_numpy()

                r = metric_row(
                    tmp,
                    method=f"{pair_name}_train_alpha_{alpha:.2f}",
                    score_col="score",
                    scope="train_categories",
                    alpha=float(alpha),
                    fusion_pair=pair_name,
                    protocol="loco_train_select",
                )
                train_rows.append(r)

            train_df = pd.DataFrame(train_rows)
            best_train = train_df.sort_values(["auroc", "ap"], ascending=False).iloc[0]
            alpha = float(best_train["alpha_patchcore"])

            test_score = (
                alpha * test[f"{PATCHCORE_METHOD}_z"]
                + (1.0 - alpha) * test[f"{vlm_col}_z"]
            )

            tmp_test = test[["category", "image_path", "y_true"]].copy()
            tmp_test["score"] = test_score.to_numpy()

            test_row = metric_row(
                tmp_test,
                method=f"{pair_name}_loco_selected_alpha",
                score_col="score",
                scope="held_out_category",
                category=held_out,
                alpha=alpha,
                fusion_pair=pair_name,
                protocol="loco_test",
            )

            test_row["train_auroc"] = float(best_train["auroc"])
            test_row["train_ap"] = float(best_train["ap"])
            rows.append(test_row)

    loco = pd.DataFrame(rows)

    # Aggregate held-out predictions using their selected alphas.
    aggregate_rows = []
    for pair_name in ["patchcore_plus_full_image", "patchcore_plus_context"]:
        parts = []
        for held_out in PRIMARY_CATEGORIES:
            selected = loco[
                (loco["fusion_pair"] == pair_name)
                & (loco["category"] == held_out)
            ].iloc[0]
            alpha = float(selected["alpha_patchcore"])

            test = work[work["category"] == held_out].copy()
            vlm_col = FULL_METHOD if pair_name == "patchcore_plus_full_image" else CONTEXT_METHOD
            test["score"] = (
                alpha * test[f"{PATCHCORE_METHOD}_z"]
                + (1.0 - alpha) * test[f"{vlm_col}_z"]
            )
            parts.append(test[["category", "image_path", "y_true", "score"]])

        agg = pd.concat(parts, ignore_index=True)
        r = metric_row(
            agg,
            method=f"{pair_name}_loco_aggregate",
            score_col="score",
            scope="ALL_PRIMARY",
            alpha=None,
            fusion_pair=pair_name,
            protocol="loco_aggregate",
        )
        aggregate_rows.append(r)

    return pd.concat([loco, pd.DataFrame(aggregate_rows)], ignore_index=True)


def build_complementarity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category in ["ALL_PRIMARY"] + PRIMARY_CATEGORIES:
        part = df.copy() if category == "ALL_PRIMARY" else df[df["category"] == category].copy()

        corr_full = part[[PATCHCORE_METHOD, FULL_METHOD]].corr(method="spearman").iloc[0, 1]
        corr_context = part[[PATCHCORE_METHOD, CONTEXT_METHOD]].corr(method="spearman").iloc[0, 1]

        rows.append({
            "category": category,
            "num_images": int(len(part)),
            "patchcore_full_spearman": float(corr_full),
            "patchcore_context_spearman": float(corr_context),
            "interpretation": (
                "lower correlation suggests more potential complementarity; "
                "high correlation suggests VLM score mostly repeats detector ranking"
            ),
        })

    return pd.DataFrame(rows)


def write_report(
    direct_summary: pd.DataFrame,
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    per_category: pd.DataFrame,
    loco: pd.DataFrame,
    complement: pd.DataFrame,
) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    patch = summary[summary["method"] == PATCHCORE_METHOD].iloc[0]
    full = summary[summary["method"] == FULL_METHOD].iloc[0]
    context = summary[summary["method"] == CONTEXT_METHOD].iloc[0]

    best_full_fusion = summary[summary["method"] == "patchcore_plus_full_image_best_same_set"].iloc[0]
    best_context_fusion = summary[summary["method"] == "patchcore_plus_context_best_same_set"].iloc[0]

    loco_agg = loco[loco["protocol"] == "loco_aggregate"].copy()
    loco_context = loco_agg[loco_agg["fusion_pair"] == "patchcore_plus_context"].iloc[0]
    loco_full = loco_agg[loco_agg["fusion_pair"] == "patchcore_plus_full_image"].iloc[0]

    context_delta_vs_patch = float(best_context_fusion["auroc"]) - float(patch["auroc"])
    loco_context_delta_vs_patch = float(loco_context["auroc"]) - float(patch["auroc"])

    if context_delta_vs_patch > 1e-6:
        same_set_decision = "same-set fusion exceeds PatchCore, indicating possible complementary detection information."
    else:
        same_set_decision = "same-set fusion does not exceed PatchCore; VLM branch has not yet shown detection-performance gain over the strong detector."

    if loco_context_delta_vs_patch > 1e-6:
        loco_decision = "leave-one-category-out fusion exceeds PatchCore, giving stronger evidence of robust complementarity."
    else:
        loco_decision = "leave-one-category-out fusion does not exceed PatchCore; any same-set gain should be treated cautiously."

    lines = []
    lines.append("# Stage 13-A PatchCore and VLM Score Fusion")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage directly tests whether the context-aware VLM branch provides useful complementary information to the strong PatchCore detector.")
    lines.append("")
    lines.append("It does not train models, regenerate crops, or rerun VLM inference. It only fuses existing image-level scores from Stage 11-D.")
    lines.append("")
    lines.append("## 2. Why this stage is necessary")
    lines.append("")
    lines.append("The previous Stage 11 result showed that context-aware VLM scoring improves over full-image VLM scoring, but it is still weaker than PatchCore.")
    lines.append("")
    lines.append("Therefore, the key question is no longer whether context crops beat full-image VLM. The stronger question is whether the context-aware VLM branch can improve or complement PatchCore.")
    lines.append("")
    lines.append("## 3. Direct Baselines on ALL_PRIMARY")
    lines.append("")
    lines.append("| Method | AUROC | AP | Best F1 | Best Acc |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in summary[
        summary["method"].isin([PATCHCORE_METHOD, FULL_METHOD, CONTEXT_METHOD])
    ].iterrows():
        lines.append(
            f"| {r['method']} | {f4(r['auroc'])} | {f4(r['ap'])} | "
            f"{f4(r['best_f1'])} | {f4(r['best_accuracy'])} |"
        )

    lines.append("")
    lines.append("## 4. Same-set Fusion Result")
    lines.append("")
    lines.append("Same-set fusion searches the fusion weight on the same evaluation set. It is an upper-bound diagnostic and should not be overclaimed as final fair evidence.")
    lines.append("")
    lines.append("| Fusion | Best alpha for PatchCore | AUROC | ΔAUROC vs PatchCore | AP |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in [best_full_fusion, best_context_fusion]:
        delta = float(r["auroc"]) - float(patch["auroc"])
        lines.append(
            f"| {r['fusion_pair']} | {f4(r['alpha_patchcore'])} | "
            f"{f4(r['auroc'])} | {f4(delta)} | {f4(r['ap'])} |"
        )

    lines.append("")
    lines.append(f"Same-set decision: {same_set_decision}")
    lines.append("")
    lines.append("## 5. Leave-one-category-out Fusion")
    lines.append("")
    lines.append("Leave-one-category-out fusion selects the fusion weight on three categories and evaluates on the held-out category. This is more conservative than same-set fusion.")
    lines.append("")
    lines.append("| Held-out category / aggregate | Fusion | Alpha for PatchCore | AUROC | AP | Train AUROC |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for _, r in loco.iterrows():
        label = r["category"] if r["category"] != "" else r["scope"]
        lines.append(
            f"| {label} | {r['fusion_pair']} | {f4(r['alpha_patchcore'])} | "
            f"{f4(r['auroc'])} | {f4(r['ap'])} | {f4(r.get('train_auroc', ''))} |"
        )

    lines.append("")
    lines.append(f"Leave-one-category-out decision: {loco_decision}")
    lines.append("")
    lines.append("## 6. Per-category Same-set Best Fusion")
    lines.append("")
    lines.append("| Category | Fusion | Alpha for PatchCore | AUROC | AP |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in per_category.iterrows():
        lines.append(
            f"| {r['category']} | {r['fusion_pair']} | {f4(r['alpha_patchcore'])} | "
            f"{f4(r['auroc'])} | {f4(r['ap'])} |"
        )

    lines.append("")
    lines.append("## 7. Score Complementarity")
    lines.append("")
    lines.append("Lower rank correlation between PatchCore and a VLM score suggests more potential complementarity. High correlation suggests the VLM branch mostly repeats the detector ranking.")
    lines.append("")
    lines.append("| Category | PatchCore vs full-image correlation | PatchCore vs context correlation |")
    lines.append("|---|---:|---:|")
    for _, r in complement.iterrows():
        lines.append(
            f"| {r['category']} | {f4(r['patchcore_full_spearman'])} | "
            f"{f4(r['patchcore_context_spearman'])} |"
        )

    lines.append("")
    lines.append("## 8. Interpretation Policy")
    lines.append("")
    if float(best_context_fusion["auroc"]) > float(patch["auroc"]):
        lines.append("The same-set result suggests that the context-aware VLM branch may provide complementary evidence to PatchCore.")
    else:
        lines.append("The same-set result does not show detection-performance gain over PatchCore. In this case, the context-aware VLM branch should not be claimed as a stronger detector.")
    lines.append("")
    if float(loco_context["auroc"]) > float(patch["auroc"]):
        lines.append("The leave-one-category-out result supports a stronger complementarity claim.")
    else:
        lines.append("The leave-one-category-out result does not support a robust detector-improvement claim. The safer interpretation is anomaly understanding / auxiliary review rather than detector replacement.")
    lines.append("")
    lines.append("## 9. Output Files")
    lines.append("")
    lines.append(f"- Fusion grid: `{OUT_GRID.relative_to(ROOT)}`")
    lines.append(f"- Fusion summary: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Per-category fusion: `{OUT_PER_CATEGORY.relative_to(ROOT)}`")
    lines.append(f"- Leave-one-category-out fusion: `{OUT_LOCO.relative_to(ROOT)}`")
    lines.append(f"- Complementarity table: `{OUT_COMPLEMENT.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 10. Next Step")
    lines.append("")
    lines.append("If fusion improves over PatchCore, Stage 13-B should analyze which categories and samples benefit. If fusion does not improve over PatchCore, Stage 13-B should switch to error-case and uncertain-sample analysis.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    df = load_prediction_table()

    direct_summary = build_direct_summary(df)
    grid = build_fusion_grid(df)
    summary = select_best_fusions(grid)
    per_category = build_per_category_best(grid)
    loco = build_loco(df)
    complement = build_complementarity(df)

    grid.to_csv(OUT_GRID, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    per_category.to_csv(OUT_PER_CATEGORY, index=False)
    loco.to_csv(OUT_LOCO, index=False)
    complement.to_csv(OUT_COMPLEMENT, index=False)

    write_report(
        direct_summary=direct_summary,
        grid=grid,
        summary=summary,
        per_category=per_category,
        loco=loco,
        complement=complement,
    )

    print("[DONE]", OUT_GRID)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_PER_CATEGORY)
    print("[DONE]", OUT_LOCO)
    print("[DONE]", OUT_COMPLEMENT)
    print("[DONE]", OUT_REPORT)

    print("\n===== Fusion summary =====")
    print(summary.to_string(index=False))

    print("\n===== LOCO fusion =====")
    print(loco.to_string(index=False))

    print("\n===== Complementarity =====")
    print(complement.to_string(index=False))


if __name__ == "__main__":
    main()
