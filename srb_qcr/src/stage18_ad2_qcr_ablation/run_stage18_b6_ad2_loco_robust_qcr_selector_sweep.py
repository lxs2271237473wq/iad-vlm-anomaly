from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

AD2_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

IN_ALL_CONFIGS = ROOT / "results/stage18_ad2_qcr_ablation/stage18_b5_ad2_loco_qcr_all_configs_per_category.csv"

OUT_DIR = ROOT / "results/stage18_ad2_qcr_ablation"
DOC_DIR = ROOT / "docs/stage18_ad2_qcr_ablation"

OUT_FOLDS = OUT_DIR / "stage18_b6_ad2_loco_robust_selector_folds.csv"
OUT_SUMMARY = OUT_DIR / "stage18_b6_ad2_loco_robust_selector_summary.csv"
OUT_REPORT = DOC_DIR / "stage18_b6_ad2_loco_robust_qcr_selector_sweep_report.md"


GROUP_COLS = ["q_source", "q_direction", "eta", "gamma"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"Bad CSV format: {path}")
    return df


def is_valid_candidate_quality_source(q_source: str) -> bool:
    q = str(q_source).lower()

    # Hard invalid: these are VLM/evidence scores or label-like terms, not candidate-quality proxies.
    invalid_tokens = [
        "full_image",
        "context_top",
        "tight_top",
        "vlm",
        "clip",
        "gt",
        "label",
        "target",
        "anomaly_binary",
    ]

    if any(t in q for t in invalid_tokens):
        return False

    valid_prefixes = [
        "candidate_score_",
        "tight_candidate_mask_density",
        "context_candidate_mask_density",
        "map_area",
        "num_candidates",
    ]

    return any(q.startswith(p) for p in valid_prefixes)


def train_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = (
        train.groupby(GROUP_COLS, as_index=False)
        .agg(
            train_mean_V3=("auroc_V3_naive", "mean"),
            train_mean_quality=("auroc_quality_qcr", "mean"),
            train_mean_adaptive=("auroc_adaptive_qcr", "mean"),
            train_delta_quality=("delta_quality_minus_V3", "mean"),
            train_delta_adaptive=("delta_adaptive_minus_V3", "mean"),
            train_wins_quality=("delta_quality_minus_V3", lambda x: int((x > 0).sum())),
            train_wins_adaptive=("delta_adaptive_minus_V3", lambda x: int((x > 0).sum())),
            train_worst_delta_quality=("delta_quality_minus_V3", "min"),
            train_worst_delta_adaptive=("delta_adaptive_minus_V3", "min"),
            train_std_delta_adaptive=("delta_adaptive_minus_V3", "std"),
        )
        .fillna({"train_std_delta_adaptive": 0.0})
    )

    # A conservative robust objective: prefer positive average delta, many wins, and avoid catastrophic categories.
    g["robust_delta_score"] = (
        g["train_delta_adaptive"]
        + 0.50 * g["train_worst_delta_adaptive"]
        + 0.01 * g["train_wins_adaptive"]
        - 0.05 * g["train_std_delta_adaptive"]
    )

    # Another conservative objective focused on quality-only, because gamma often does not help.
    g["robust_quality_score"] = (
        g["train_delta_quality"]
        + 0.50 * g["train_worst_delta_quality"]
        + 0.01 * g["train_wins_quality"]
    )

    return g


def select_config(ts: pd.DataFrame, selector: str) -> pd.Series:
    work = ts.copy()

    if selector == "B5_baseline_max_train_adaptive_auroc":
        return work.sort_values(
            ["train_mean_adaptive", "train_delta_adaptive", "train_wins_adaptive", "train_worst_delta_adaptive"],
            ascending=[False, False, False, False],
        ).iloc[0]

    if selector == "max_train_delta_adaptive":
        return work.sort_values(
            ["train_delta_adaptive", "train_wins_adaptive", "train_worst_delta_adaptive", "train_mean_adaptive"],
            ascending=[False, False, False, False],
        ).iloc[0]

    if selector == "wins_then_delta_adaptive":
        return work.sort_values(
            ["train_wins_adaptive", "train_delta_adaptive", "train_worst_delta_adaptive", "train_mean_adaptive"],
            ascending=[False, False, False, False],
        ).iloc[0]

    if selector == "worst_delta_then_mean_delta_adaptive":
        return work.sort_values(
            ["train_worst_delta_adaptive", "train_delta_adaptive", "train_wins_adaptive", "train_mean_adaptive"],
            ascending=[False, False, False, False],
        ).iloc[0]

    if selector == "robust_delta_score":
        return work.sort_values(
            ["robust_delta_score", "train_wins_adaptive", "train_delta_adaptive", "train_worst_delta_adaptive"],
            ascending=[False, False, False, False],
        ).iloc[0]

    if selector == "robust_quality_score":
        return work.sort_values(
            ["robust_quality_score", "train_wins_quality", "train_delta_quality", "train_worst_delta_quality"],
            ascending=[False, False, False, False],
        ).iloc[0]

    if selector == "semantic_candidate_score_max_min_inverted":
        sub = work[
            (work["q_source"] == "candidate_score_max_min")
            & (work["q_direction"] == "inverted")
        ].copy()
        if sub.empty:
            sub = work.copy()
        return sub.sort_values(
            ["robust_delta_score", "train_wins_adaptive", "train_delta_adaptive", "train_worst_delta_adaptive"],
            ascending=[False, False, False, False],
        ).iloc[0]

    if selector == "semantic_candidate_score_max_mean_inverted":
        sub = work[
            (work["q_source"] == "candidate_score_max_mean")
            & (work["q_direction"] == "inverted")
        ].copy()
        if sub.empty:
            sub = work.copy()
        return sub.sort_values(
            ["robust_delta_score", "train_wins_adaptive", "train_delta_adaptive", "train_worst_delta_adaptive"],
            ascending=[False, False, False, False],
        ).iloc[0]

    raise ValueError(f"Unknown selector: {selector}")


def run_selector(all_configs: pd.DataFrame, selector: str) -> pd.DataFrame:
    rows = []

    for heldout in AD2_CATEGORIES:
        train_cats = [c for c in AD2_CATEGORIES if c != heldout]

        train = all_configs[all_configs["category"].isin(train_cats)].copy()
        test = all_configs[all_configs["category"] == heldout].copy()

        ts = train_summary(train)
        chosen = select_config(ts, selector)

        match = np.ones(len(test), dtype=bool)
        for c in GROUP_COLS:
            match &= test[c].values == chosen[c]

        if match.sum() != 1:
            raise RuntimeError(f"Selector {selector}, heldout {heldout}: expected one test config, got {match.sum()}")

        tr = test[match].iloc[0]

        rows.append(
            {
                "selector": selector,
                "heldout_category": heldout,
                "train_categories": ";".join(train_cats),
                "selected_q_source": chosen["q_source"],
                "selected_q_direction": chosen["q_direction"],
                "selected_eta": float(chosen["eta"]),
                "selected_gamma": float(chosen["gamma"]),
                "train_mean_V3": float(chosen["train_mean_V3"]),
                "train_mean_quality": float(chosen["train_mean_quality"]),
                "train_mean_adaptive": float(chosen["train_mean_adaptive"]),
                "train_delta_quality": float(chosen["train_delta_quality"]),
                "train_delta_adaptive": float(chosen["train_delta_adaptive"]),
                "train_wins_quality": int(chosen["train_wins_quality"]),
                "train_wins_adaptive": int(chosen["train_wins_adaptive"]),
                "train_worst_delta_quality": float(chosen["train_worst_delta_quality"]),
                "train_worst_delta_adaptive": float(chosen["train_worst_delta_adaptive"]),
                "test_V3": float(tr["auroc_V3_naive"]),
                "test_quality_qcr": float(tr["auroc_quality_qcr"]),
                "test_adaptive_qcr": float(tr["auroc_adaptive_qcr"]),
                "test_delta_quality_minus_V3": float(tr["delta_quality_minus_V3"]),
                "test_delta_adaptive_minus_V3": float(tr["delta_adaptive_minus_V3"]),
                "test_delta_adaptive_minus_quality": float(tr["delta_adaptive_minus_quality"]),
                "test_Q_alone": float(tr["auroc_Q_alone"]),
            }
        )

    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for selector, sub in folds.groupby("selector"):
        rows.append(
            {
                "selector": selector,
                "num_folds": len(sub),
                "mean_test_V3": float(sub["test_V3"].mean()),
                "mean_test_quality_qcr": float(sub["test_quality_qcr"].mean()),
                "mean_test_adaptive_qcr": float(sub["test_adaptive_qcr"].mean()),
                "mean_delta_quality_minus_V3": float(sub["test_delta_quality_minus_V3"].mean()),
                "mean_delta_adaptive_minus_V3": float(sub["test_delta_adaptive_minus_V3"].mean()),
                "wins_quality_over_V3": int((sub["test_delta_quality_minus_V3"] > 0).sum()),
                "wins_adaptive_over_V3": int((sub["test_delta_adaptive_minus_V3"] > 0).sum()),
                "wins_adaptive_over_quality": int((sub["test_delta_adaptive_minus_quality"] > 0).sum()),
                "worst_quality_delta": float(sub["test_delta_quality_minus_V3"].min()),
                "worst_adaptive_delta": float(sub["test_delta_adaptive_minus_V3"].min()),
                "worst_adaptive_category": str(
                    sub.sort_values("test_delta_adaptive_minus_V3").iloc[0]["heldout_category"]
                ),
            }
        )

    out = pd.DataFrame(rows)

    out["claim_status"] = np.where(
        (out["mean_delta_adaptive_minus_V3"] > 0) & (out["wins_adaptive_over_V3"] >= 3),
        "promote_ad2_qcr_support",
        np.where(
            out["mean_delta_adaptive_minus_V3"] > 0,
            "weak_positive_boundary_support",
            "do_not_promote",
        ),
    )

    out = out.sort_values(
        ["claim_status", "mean_delta_adaptive_minus_V3", "wins_adaptive_over_V3", "worst_adaptive_delta"],
        ascending=[True, False, False, False],
    )

    return out.reset_index(drop=True)


def fmt(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.4f}"


def signed(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):+.4f}"


def write_report(valid_sources: list[str], folds: pd.DataFrame, summary: pd.DataFrame) -> None:
    best = summary.sort_values(
        ["mean_delta_adaptive_minus_V3", "wins_adaptive_over_V3", "worst_adaptive_delta"],
        ascending=[False, False, False],
    ).iloc[0]

    lines = [
        "# Stage 18-B6 AD2 LOCO Robust QCR Selector Sweep",
        "",
        "## Purpose",
        "",
        "Test whether AD2 QCR can be rescued by a more robust train-category selector rather than the B5 selector that maximizes train adaptive AUROC.",
        "",
        "No held-out category labels are used for selecting Q source, direction, eta, or gamma.",
        "",
        "## Valid candidate-quality sources",
        "",
        "```text",
        *valid_sources,
        "```",
        "",
        "## Best selector",
        "",
        f"- selector: `{best['selector']}`",
        f"- claim_status: `{best['claim_status']}`",
        f"- mean test V3: `{fmt(best['mean_test_V3'])}`",
        f"- mean test adaptive QCR: `{fmt(best['mean_test_adaptive_qcr'])}`",
        f"- adaptive QCR minus V3: `{signed(best['mean_delta_adaptive_minus_V3'])}`",
        f"- adaptive wins over V3: `{int(best['wins_adaptive_over_V3'])}/4`",
        f"- worst adaptive category: `{best['worst_adaptive_category']}`",
        f"- worst adaptive delta: `{signed(best['worst_adaptive_delta'])}`",
        "",
        "## Selector summary",
        "",
        "| Selector | Status | V3 | Adaptive | Delta | Wins | Worst category | Worst delta |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['selector']} | {r['claim_status']} | "
            f"{fmt(r['mean_test_V3'])} | {fmt(r['mean_test_adaptive_qcr'])} | "
            f"{signed(r['mean_delta_adaptive_minus_V3'])} | "
            f"{int(r['wins_adaptive_over_V3'])}/4 | "
            f"{r['worst_adaptive_category']} | {signed(r['worst_adaptive_delta'])} |"
        )

    lines += [
        "",
        "## Decision rule",
        "",
        "- If at least one selector has positive mean adaptive delta and wins at least 3/4 held-out categories, AD2 QCR can be used as supporting cross-category evidence.",
        "- If all selectors have negative mean delta, stop optimizing AD2 QCR and report AD2 as boundary/sensitivity evidence.",
        "",
        "## Outputs",
        "",
        f"- `{OUT_FOLDS.relative_to(ROOT)}`",
        f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    all_configs = read_csv(IN_ALL_CONFIGS)

    all_configs["valid_candidate_quality_source"] = all_configs["q_source"].map(is_valid_candidate_quality_source)
    all_configs = all_configs[all_configs["valid_candidate_quality_source"]].copy()

    valid_sources = sorted(all_configs["q_source"].unique().tolist())

    selectors = [
        "B5_baseline_max_train_adaptive_auroc",
        "max_train_delta_adaptive",
        "wins_then_delta_adaptive",
        "worst_delta_then_mean_delta_adaptive",
        "robust_delta_score",
        "robust_quality_score",
        "semantic_candidate_score_max_min_inverted",
        "semantic_candidate_score_max_mean_inverted",
    ]

    fold_frames = []
    for selector in selectors:
        fold_frames.append(run_selector(all_configs, selector))

    folds = pd.concat(fold_frames, ignore_index=True)
    summary = summarize(folds)

    folds.to_csv(OUT_FOLDS, index=False, lineterminator="\n")
    summary.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")

    write_report(valid_sources, folds, summary)

    print("[DONE]", OUT_FOLDS)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== selector summary =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
