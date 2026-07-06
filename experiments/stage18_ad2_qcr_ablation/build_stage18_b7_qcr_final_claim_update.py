from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(".").resolve()

IN_B6_SUMMARY = ROOT / "results/stage18_ad2_qcr_ablation/stage18_b6_ad2_loco_robust_selector_summary.csv"
IN_B6_FOLDS = ROOT / "results/stage18_ad2_qcr_ablation/stage18_b6_ad2_loco_robust_selector_folds.csv"

OUT_DIR = ROOT / "results/stage18_ad2_qcr_ablation"
DOC_DIR = ROOT / "docs/stage18_ad2_qcr_ablation"

OUT_FINAL_TABLE = OUT_DIR / "stage18_b7_ad2_qcr_final_paper_facing_table.csv"
OUT_FOLD_TABLE = OUT_DIR / "stage18_b7_ad2_qcr_final_loco_folds.csv"
OUT_CLAIM_UPDATE = OUT_DIR / "stage18_b7_qcr_final_claim_update.csv"
OUT_REPORT = DOC_DIR / "stage18_b7_qcr_final_claim_update_report.md"

LOCKED_SELECTOR = "semantic_candidate_score_max_mean_inverted"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"Bad CSV format: {path}")
    return df


def fmt(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.4f}"


def signed(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):+.4f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    summary = read_csv(IN_B6_SUMMARY)
    folds = read_csv(IN_B6_FOLDS)

    s = summary[summary["selector"] == LOCKED_SELECTOR]
    if s.empty:
        raise RuntimeError(f"Missing locked selector in B6 summary: {LOCKED_SELECTOR}")
    s = s.iloc[0]

    f = folds[folds["selector"] == LOCKED_SELECTOR].copy()
    if f.empty:
        raise RuntimeError(f"Missing locked selector folds: {LOCKED_SELECTOR}")

    # Decide final AD2 score variant.
    quality_mean = float(s["mean_test_quality_qcr"])
    adaptive_mean = float(s["mean_test_adaptive_qcr"])

    if quality_mean >= adaptive_mean:
        final_variant = "Quality-Calibrated QCR"
        final_score = quality_mean
        final_delta = float(s["mean_delta_quality_minus_V3"])
        final_wins = int(s["wins_quality_over_V3"])
        final_worst_delta = float(s["worst_quality_delta"])
        final_note = "Quality-only calibration is selected because it is slightly stronger than adaptive refinement on AD2."
    else:
        final_variant = "Quality-Calibrated QCR + adaptive refinement"
        final_score = adaptive_mean
        final_delta = float(s["mean_delta_adaptive_minus_V3"])
        final_wins = int(s["wins_adaptive_over_V3"])
        final_worst_delta = float(s["worst_adaptive_delta"])
        final_note = "Adaptive refinement is selected because it is stronger than quality-only calibration on AD2."

    final_table = pd.DataFrame(
        [
            {
                "setting": "AD2 four-category LOCO policy",
                "method": "Naive detector-crop fusion",
                "mean_image_auroc": float(s["mean_test_V3"]),
                "delta_vs_naive": 0.0,
                "wins_vs_naive": "",
                "paper_role": "baseline",
            },
            {
                "setting": "AD2 four-category LOCO policy",
                "method": "Quality-Calibrated QCR",
                "mean_image_auroc": quality_mean,
                "delta_vs_naive": float(s["mean_delta_quality_minus_V3"]),
                "wins_vs_naive": f"{int(s['wins_quality_over_V3'])}/4",
                "paper_role": "main_qcr_support",
            },
            {
                "setting": "AD2 four-category LOCO policy",
                "method": "Quality-Calibrated QCR + adaptive refinement",
                "mean_image_auroc": adaptive_mean,
                "delta_vs_naive": float(s["mean_delta_adaptive_minus_V3"]),
                "wins_vs_naive": f"{int(s['wins_adaptive_over_V3'])}/4",
                "paper_role": "auxiliary_refinement",
            },
        ]
    )

    final_table.to_csv(OUT_FINAL_TABLE, index=False, lineterminator="\n")

    fold_table = f[
        [
            "heldout_category",
            "selected_q_source",
            "selected_q_direction",
            "selected_eta",
            "selected_gamma",
            "test_V3",
            "test_quality_qcr",
            "test_adaptive_qcr",
            "test_delta_quality_minus_V3",
            "test_delta_adaptive_minus_V3",
        ]
    ].copy()

    fold_table.to_csv(OUT_FOLD_TABLE, index=False, lineterminator="\n")

    claim_rows = [
        {
            "claim_item": "main_innovation",
            "decision": "keep",
            "final_wording": "Quality-Calibrated QCR: candidate-quality calibration for localization-guided VLM anomaly evidence.",
            "evidence": "VisA primary ablation plus AD2 four-category LOCO policy support.",
        },
        {
            "claim_item": "adaptive_refinement",
            "decision": "downgrade_to_auxiliary",
            "final_wording": "Adaptive consistency refinement is an optional auxiliary refinement, not the primary source of the gain.",
            "evidence": "On AD2, quality-only QCR slightly exceeds adaptive QCR.",
        },
        {
            "claim_item": "ad2_qcr_support",
            "decision": "supporting_cross_category_evidence",
            "final_wording": (
                "With a fixed semantic candidate-quality source and LOCO-selected weights, "
                "QCR improves over naive detector-crop fusion on AD2 four-category evaluation."
            ),
            "evidence": f"{final_variant}: {fmt(final_score)} AUROC, {signed(final_delta)} over naive, wins {final_wins}/4.",
        },
        {
            "claim_item": "overclaim_to_avoid",
            "decision": "avoid",
            "final_wording": "Do not claim that all Q sources or adaptive consistency are universally beneficial.",
            "evidence": "Default transferred Q source and several robust selectors did not improve over naive on AD2.",
        },
    ]

    claim_update = pd.DataFrame(claim_rows)
    claim_update.to_csv(OUT_CLAIM_UPDATE, index=False, lineterminator="\n")

    lines = [
        "# Stage 18-B7 QCR Final Claim Update",
        "",
        "## Decision",
        "",
        "QCR remains viable as the main innovation, but the claim must focus on candidate-quality calibration rather than adaptive refinement.",
        "",
        "## Locked AD2 support setting",
        "",
        f"- locked selector: `{LOCKED_SELECTOR}`",
        "- Q source: `candidate_score_max_mean`",
        "- Q direction: `inverted`",
        "- protocol: AD2 four-category leave-one-category-out policy selection",
        "",
        "## Final AD2 paper-facing result",
        "",
        "| Method | Mean AUROC | Delta vs naive | Wins | Paper role |",
        "|---|---:|---:|---:|---|",
    ]

    for _, r in final_table.iterrows():
        lines.append(
            f"| {r['method']} | {fmt(r['mean_image_auroc'])} | "
            f"{signed(r['delta_vs_naive'])} | {r['wins_vs_naive']} | {r['paper_role']} |"
        )

    lines += [
        "",
        "## Final method choice",
        "",
        f"- selected AD2-facing QCR variant: `{final_variant}`",
        f"- selected AUROC: `{fmt(final_score)}`",
        f"- selected delta vs naive: `{signed(final_delta)}`",
        f"- selected wins vs naive: `{final_wins}/4`",
        f"- worst category delta: `{signed(final_worst_delta)}`",
        f"- note: {final_note}",
        "",
        "## Paper wording",
        "",
        "Use this wording:",
        "",
        "```text",
        "The main contribution is Quality-Calibrated QCR, which calibrates localization-guided crop-level VLM evidence using candidate-region quality. On VisA, the controlled ablation shows consistent gains over naive detector-crop fusion. On the AD2 four-category setting, a fixed semantic candidate-quality source with leave-one-category-out policy selection improves over naive fusion, providing supporting cross-category evidence. Adaptive consistency refinement is retained only as an auxiliary analysis rather than the primary source of improvement.",
        "```",
        "",
        "Avoid this wording:",
        "",
        "```text",
        "Adaptive QCR is universally beneficial across all datasets and all candidate-quality sources.",
        "```",
        "",
        "## Fold details",
        "",
        "| Held-out | Q source | Direction | eta | gamma | V3 | Quality QCR | Adaptive QCR | Quality-V3 | Adaptive-V3 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in fold_table.iterrows():
        lines.append(
            f"| {r['heldout_category']} | {r['selected_q_source']} | {r['selected_q_direction']} | "
            f"{float(r['selected_eta']):.2f} | {float(r['selected_gamma']):.2f} | "
            f"{fmt(r['test_V3'])} | {fmt(r['test_quality_qcr'])} | {fmt(r['test_adaptive_qcr'])} | "
            f"{signed(r['test_delta_quality_minus_V3'])} | {signed(r['test_delta_adaptive_minus_V3'])} |"
        )

    lines += [
        "",
        "## Outputs",
        "",
        f"- `{OUT_FINAL_TABLE.relative_to(ROOT)}`",
        f"- `{OUT_FOLD_TABLE.relative_to(ROOT)}`",
        f"- `{OUT_CLAIM_UPDATE.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_FINAL_TABLE)
    print("[DONE]", OUT_FOLD_TABLE)
    print("[DONE]", OUT_CLAIM_UPDATE)
    print("[DONE]", OUT_REPORT)
    print()
    print(final_table.to_string(index=False))
    print()
    print(claim_update.to_string(index=False))


if __name__ == "__main__":
    main()
