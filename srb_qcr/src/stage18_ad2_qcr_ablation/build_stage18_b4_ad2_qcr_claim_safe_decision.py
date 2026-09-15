from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(".").resolve()

IN_B2_SUMMARY = ROOT / "results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_ablation_summary.csv"
IN_B2_DELTAS = ROOT / "results/stage18_ad2_qcr_ablation/stage18_b2_ad2_qcr_claim_ready_deltas.csv"
IN_B3_RANKED = ROOT / "results/stage18_ad2_qcr_ablation/stage18_b3_ad2_q_source_sweep_ranked.csv"

OUT_DIR = ROOT / "results/stage18_ad2_qcr_ablation"
DOC_DIR = ROOT / "docs/stage18_ad2_qcr_ablation"

OUT_VALID = OUT_DIR / "stage18_b4_ad2_qcr_valid_q_sources_ranked.csv"
OUT_DECISION = OUT_DIR / "stage18_b4_ad2_qcr_claim_safe_decision.csv"
OUT_TABLE = OUT_DIR / "stage18_b4_ad2_qcr_paper_facing_table.csv"
OUT_REPORT = DOC_DIR / "stage18_b4_ad2_qcr_claim_safe_decision_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"Bad CSV format: {path}")
    return df


def is_invalid_q_source(q_source: str) -> bool:
    q = str(q_source).lower()

    invalid_tokens = [
        "full_image",
        "context_top",
        "tight_top",
        "normal_score",
        "anomaly_score",
        "vlm",
        "clip",
    ]

    return any(t in q for t in invalid_tokens)


def quality_family(q_source: str) -> str:
    q = str(q_source).lower()

    if q.startswith("candidate_score_"):
        return "candidate_region_score"
    if q.startswith("map_area_"):
        return "region_geometry"
    if q.startswith("num_candidates"):
        return "candidate_count"

    return "unsupported_or_vlm_evidence"


def fmt(x: float | int | str) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def signed(x: float | int | str) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):+.4f}"
    except Exception:
        return str(x)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    b2_summary = read_csv(IN_B2_SUMMARY)
    b2_deltas = read_csv(IN_B2_DELTAS)
    b3 = read_csv(IN_B3_RANKED)

    b3["quality_family"] = b3["q_source"].map(quality_family)
    b3["invalid_as_candidate_quality"] = b3["q_source"].map(is_invalid_q_source)
    b3["passes_mean_positive"] = b3["mean_delta_V4_minus_V3"] > 0
    b3["passes_3_of_4_wins"] = b3["wins_V4_over_V3"] >= 3
    b3["claim_safe_candidate"] = (
        (~b3["invalid_as_candidate_quality"])
        & (b3["quality_family"] != "unsupported_or_vlm_evidence")
        & b3["passes_mean_positive"]
        & b3["passes_3_of_4_wins"]
    )

    valid = b3[b3["claim_safe_candidate"]].copy()

    valid_perf = valid.sort_values(
        ["mean_delta_V4_minus_V3", "wins_V4_over_V3", "mean_auroc_V4_quality"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    valid_stable = valid.sort_values(
        ["worst_category_delta_V4_minus_V3", "wins_V4_over_V3", "mean_delta_V4_minus_V3"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    all_valid_ranked = valid.sort_values(
        [
            "wins_V4_over_V3",
            "mean_delta_V4_minus_V3",
            "worst_category_delta_V4_minus_V3",
            "mean_auroc_V4_quality",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    all_valid_ranked.to_csv(OUT_VALID, index=False, lineterminator="\n")

    b2_v3 = b2_summary[b2_summary["variant_id"] == "V3"].iloc[0]
    b2_v4 = b2_summary[b2_summary["variant_id"] == "V4"].iloc[0]
    b2_v6 = b2_summary[b2_summary["variant_id"] == "V6"].iloc[0]

    b2_v4_delta = b2_deltas[
        b2_deltas["comparison"] == "Quality-Calibrated QCR vs naive fusion"
    ].iloc[0]["delta_a_minus_b"]

    b2_v6_delta = b2_deltas[
        b2_deltas["comparison"] == "Adaptive refinement vs naive fusion"
    ].iloc[0]["delta_a_minus_b"]

    decision_rows = []

    # 1. Raw B2 default result.
    decision_rows.append(
        {
            "case_id": "B2_default_q_source",
            "paper_status": "boundary_result_not_main_claim",
            "q_source": "candidate_score_mean_max",
            "q_direction": "direct",
            "quality_family": "candidate_region_score",
            "mean_auroc_V3_naive": float(b2_v3["mean_image_auroc"]),
            "mean_auroc_V4_quality": float(b2_v4["mean_image_auroc"]),
            "mean_auroc_V6_adaptive": float(b2_v6["mean_image_auroc"]),
            "delta_V4_minus_V3": float(b2_v4_delta),
            "delta_V6_minus_V3": float(b2_v6_delta),
            "wins_V4_over_V3": np.nan,
            "worst_category": "",
            "worst_category_delta_V4_minus_V3": np.nan,
            "decision": "Do not use as main AD2 QCR evidence because V4 is below V3.",
        }
    )

    # 2. Invalid best source.
    invalid_best = b3.iloc[0]
    decision_rows.append(
        {
            "case_id": "B3_best_overall_invalid_as_Q",
            "paper_status": "exclude_from_qcr_claim",
            "q_source": invalid_best["q_source"],
            "q_direction": invalid_best["q_direction"],
            "quality_family": invalid_best["quality_family"],
            "mean_auroc_V3_naive": float(invalid_best["mean_auroc_V3_naive"]),
            "mean_auroc_V4_quality": float(invalid_best["mean_auroc_V4_quality"]),
            "mean_auroc_V6_adaptive": float(invalid_best["mean_auroc_V6_adaptive"]),
            "delta_V4_minus_V3": float(invalid_best["mean_delta_V4_minus_V3"]),
            "delta_V6_minus_V3": float(invalid_best["mean_delta_V6_minus_V3"]),
            "wins_V4_over_V3": int(invalid_best["wins_V4_over_V3"]),
            "worst_category": invalid_best["worst_category"],
            "worst_category_delta_V4_minus_V3": float(invalid_best["worst_category_delta_V4_minus_V3"]),
            "decision": "Exclude because this is VLM evidence, not candidate quality.",
        }
    )

    # 3. Performance-best valid source.
    if not valid_perf.empty:
        r = valid_perf.iloc[0]
        decision_rows.append(
            {
                "case_id": "B3_performance_best_valid_candidate_Q",
                "paper_status": "supporting_source_sensitivity",
                "q_source": r["q_source"],
                "q_direction": r["q_direction"],
                "quality_family": r["quality_family"],
                "mean_auroc_V3_naive": float(r["mean_auroc_V3_naive"]),
                "mean_auroc_V4_quality": float(r["mean_auroc_V4_quality"]),
                "mean_auroc_V6_adaptive": float(r["mean_auroc_V6_adaptive"]),
                "delta_V4_minus_V3": float(r["mean_delta_V4_minus_V3"]),
                "delta_V6_minus_V3": float(r["mean_delta_V6_minus_V3"]),
                "wins_V4_over_V3": int(r["wins_V4_over_V3"]),
                "worst_category": r["worst_category"],
                "worst_category_delta_V4_minus_V3": float(r["worst_category_delta_V4_minus_V3"]),
                "decision": "Valid non-GT candidate-quality source; use as sensitivity, not final main claim unless Q definition is locked and rerun consistently.",
            }
        )

    # 4. Stability-preferred valid source.
    if not valid_stable.empty:
        r = valid_stable.iloc[0]
        decision_rows.append(
            {
                "case_id": "B3_stability_preferred_valid_candidate_Q",
                "paper_status": "recommended_if_formalizing_new_Q_definition",
                "q_source": r["q_source"],
                "q_direction": r["q_direction"],
                "quality_family": r["quality_family"],
                "mean_auroc_V3_naive": float(r["mean_auroc_V3_naive"]),
                "mean_auroc_V4_quality": float(r["mean_auroc_V4_quality"]),
                "mean_auroc_V6_adaptive": float(r["mean_auroc_V6_adaptive"]),
                "delta_V4_minus_V3": float(r["mean_delta_V4_minus_V3"]),
                "delta_V6_minus_V3": float(r["mean_delta_V6_minus_V3"]),
                "wins_V4_over_V3": int(r["wins_V4_over_V3"]),
                "worst_category": r["worst_category"],
                "worst_category_delta_V4_minus_V3": float(r["worst_category_delta_V4_minus_V3"]),
                "decision": "Best conservative valid candidate-quality source by worst-category stability.",
            }
        )

    decision = pd.DataFrame(decision_rows)
    decision.to_csv(OUT_DECISION, index=False, lineterminator="\n")

    table = decision[
        [
            "case_id",
            "paper_status",
            "q_source",
            "q_direction",
            "mean_auroc_V3_naive",
            "mean_auroc_V4_quality",
            "mean_auroc_V6_adaptive",
            "delta_V4_minus_V3",
            "wins_V4_over_V3",
            "worst_category",
            "decision",
        ]
    ].copy()

    table.to_csv(OUT_TABLE, index=False, lineterminator="\n")

    if valid.empty:
        final_status = "ad2_qcr_boundary_only_no_valid_q_source_passed"
        recommended_q = "none"
    else:
        stable = valid_stable.iloc[0]
        final_status = "ad2_qcr_supporting_sensitivity_not_main_claim_yet"
        recommended_q = f"{stable['q_source']} / {stable['q_direction']}"

    lines = [
        "# Stage 18-B4 AD2 QCR Claim-safe Decision",
        "",
        "## Purpose",
        "",
        "Convert the AD2 Q-source sweep into a claim-safe decision table.",
        "",
        "## Key decision",
        "",
        f"- final_status: `{final_status}`",
        f"- recommended_q_if_formalizing_new_definition: `{recommended_q}`",
        "",
        "## Why the best overall source is excluded",
        "",
        "The top overall source in B3 is `full_image_score/direct`, but this is full-image VLM evidence, not candidate quality. It must not be used as Q in a candidate-quality calibration claim.",
        "",
        "## Claim-safe cases",
        "",
        "| Case | Status | Q source | Direction | V3 | V4 | V6 | V4-V3 | Wins | Worst category |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for _, r in decision.iterrows():
        wins = "NA" if pd.isna(r["wins_V4_over_V3"]) else str(int(r["wins_V4_over_V3"]))
        lines.append(
            f"| {r['case_id']} | {r['paper_status']} | {r['q_source']} | {r['q_direction']} | "
            f"{fmt(r['mean_auroc_V3_naive'])} | {fmt(r['mean_auroc_V4_quality'])} | "
            f"{fmt(r['mean_auroc_V6_adaptive'])} | {signed(r['delta_V4_minus_V3'])} | "
            f"{wins} | {r['worst_category']} |"
        )

    lines += [
        "",
        "## Paper recommendation",
        "",
        "Use AD2 four-category QCR as a source-sensitivity/boundary-supporting result unless the new Q definition is formally locked and rerun consistently across the main VisA ablation.",
        "",
        "Recommended wording:",
        "",
        "```text",
        "On the AD2 four-category setting, the default transferred Q source is not uniformly beneficial. A non-GT candidate-region score source recovers a positive mean gain over naive detector-crop fusion, but we report this as a candidate-quality source sensitivity rather than as the primary QCR claim.",
        "```",
        "",
        "## Outputs",
        "",
        f"- `{OUT_VALID.relative_to(ROOT)}`",
        f"- `{OUT_DECISION.relative_to(ROOT)}`",
        f"- `{OUT_TABLE.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_VALID)
    print("[DONE]", OUT_DECISION)
    print("[DONE]", OUT_TABLE)
    print("[DONE]", OUT_REPORT)
    print()
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
