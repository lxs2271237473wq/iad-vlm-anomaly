from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_PRIMARY = ROOT / "results/stage16_qcru_ablation/stage16_b_adaptive_qcru_primary_protocol_table.csv"
IN_DECISION = ROOT / "results/stage16_qcru_ablation/stage16_b_adaptive_qcru_final_method_decision.csv"

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_CSV = OUT_DIR / "stage16_c_final_method_claims.csv"
OUT_DOC = DOC_DIR / "stage16_c_final_method_claims_report.md"


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
    return df


def get_primary_delta(primary: pd.DataFrame, left: str, right: str) -> dict:
    idx = ["backbone", "dataset", "strategy", "eval_mode"]
    piv = primary.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
    piv.columns.name = None

    if left not in piv.columns or right not in piv.columns:
        return {
            "left": left,
            "right": right,
            "num_protocols": 0,
            "wins": 0,
            "win_rate": 0.0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "min_delta": float("nan"),
            "max_delta": float("nan"),
        }

    d = piv[left] - piv[right]
    return {
        "left": left,
        "right": right,
        "num_protocols": int(len(d)),
        "wins": int((d > 0).sum()),
        "win_rate": float((d > 0).mean()) if len(d) else 0.0,
        "mean_delta": float(d.mean()),
        "median_delta": float(d.median()),
        "min_delta": float(d.min()),
        "max_delta": float(d.max()),
    }


def lookup_decision(decision: pd.DataFrame, scope: str, comparison: str) -> dict:
    row = decision[
        (decision["scope"] == scope)
        & (decision["comparison"] == comparison)
    ]

    if row.empty:
        return {
            "num_protocols": "",
            "wins": "",
            "win_rate": "",
            "mean_delta": "",
            "median_delta": "",
            "min_delta": "",
            "max_delta": "",
        }

    r = row.iloc[0]
    return {
        "num_protocols": int(r["num_protocols"]),
        "wins": int(r["wins"]),
        "win_rate": float(r["win_rate"]),
        "mean_delta": float(r["mean_delta"]),
        "median_delta": float(r["median_delta"]),
        "min_delta": float(r["min_delta"]),
        "max_delta": float(r["max_delta"]),
    }


def fmt(x) -> str:
    if x == "":
        return ""
    try:
        return f"{float(x):+.4f}"
    except Exception:
        return str(x)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    primary = read_csv_strict(IN_PRIMARY)
    decision = read_csv_strict(IN_DECISION)

    # Primary-protocol deltas.
    d_v4_v3 = get_primary_delta(primary, "V4", "V3")
    d_v6_v3 = get_primary_delta(primary, "V6", "V3")
    d_v6_v4 = get_primary_delta(primary, "V6", "V4")
    d_v5_v4 = get_primary_delta(primary, "V5", "V4")
    d_v6_v5 = get_primary_delta(primary, "V6", "V5")

    # All-protocol summaries from Stage 16-B decision file.
    all_v4_v3 = lookup_decision(decision, "all_protocols", "quality_minus_naive")
    all_v6_v3 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_naive")
    all_v6_v4 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_quality")
    all_v6_v5 = lookup_decision(decision, "all_protocols", "adaptive_qcru_minus_fixed_qc")

    # Recommendation from Stage 16-B.
    recommended_name = ""
    final_recommendation = ""
    if "recommended_method_name" in decision.columns:
        vals = [v for v in decision["recommended_method_name"].dropna().astype(str).tolist() if v.strip()]
        if vals:
            recommended_name = vals[0]
    if "final_recommendation" in decision.columns:
        vals = [v for v in decision["final_recommendation"].dropna().astype(str).tolist() if v.strip()]
        if vals:
            final_recommendation = vals[0]

    if not recommended_name:
        if d_v6_v4["mean_delta"] >= 0.005:
            recommended_name = "Adaptive QCR-U"
        elif d_v6_v4["mean_delta"] > 0:
            recommended_name = "Quality-Calibrated QCR with adaptive consistency refinement"
        else:
            recommended_name = "Quality-Calibrated Localization-Guided Fusion"

    if not final_recommendation:
        final_recommendation = (
            "Use the quality-calibrated method as the main paper-facing method; "
            "treat adaptive consistency as a conservative refinement."
        )

    rows = [
        {
            "claim_id": "C1",
            "claim_type": "final_method_name",
            "claim": "Use Quality-Calibrated QCR as the main paper-facing method family.",
            "evidence": (
                f"Stage 16-B recommends `{recommended_name}`. "
                f"Primary adaptive-minus-quality mean delta is {fmt(d_v6_v4['mean_delta'])} AUROC."
            ),
            "paper_status": "use",
        },
        {
            "claim_id": "C2",
            "claim_type": "main_effective_component",
            "claim": "Candidate quality calibration is the main effective component.",
            "evidence": (
                f"Primary quality-minus-naive mean delta is {fmt(d_v4_v3['mean_delta'])} AUROC; "
                f"all-protocol quality-minus-naive mean delta is {fmt(all_v4_v3['mean_delta'])} AUROC."
            ),
            "paper_status": "use",
        },
        {
            "claim_id": "C3",
            "claim_type": "auxiliary_component",
            "claim": "Adaptive consistency is a conservative refinement, not the main source of improvement.",
            "evidence": (
                f"Primary adaptive-minus-quality mean delta is only {fmt(d_v6_v4['mean_delta'])} AUROC; "
                f"all-protocol adaptive-minus-quality mean delta is {fmt(all_v6_v4['mean_delta'])} AUROC."
            ),
            "paper_status": "use_with_caution",
        },
        {
            "claim_id": "C4",
            "claim_type": "rejected_claim",
            "claim": "Do not claim fixed quality-consistency fusion as the final method.",
            "evidence": (
                f"Primary fixed-QC-minus-quality mean delta is {fmt(d_v5_v4['mean_delta'])} AUROC, "
                "but earlier robustness checks showed fixed consistency is not stable across protocols."
            ),
            "paper_status": "reject",
        },
        {
            "claim_id": "C5",
            "claim_type": "rejected_claim",
            "claim": "Do not claim consistency is universally beneficial.",
            "evidence": (
                "Fixed consistency was not robust across all protocols, and adaptive consistency only gives a small safe bonus."
            ),
            "paper_status": "reject",
        },
        {
            "claim_id": "C6",
            "claim_type": "safe_paper_claim",
            "claim": "Localization-guided VLM evidence becomes more reliable when crop evidence is calibrated by candidate quality.",
            "evidence": (
                f"Primary quality-minus-naive mean delta is {fmt(d_v4_v3['mean_delta'])} AUROC "
                f"with win rate {d_v4_v3['win_rate']:.4f}."
            ),
            "paper_status": "use",
        },
        {
            "claim_id": "C7",
            "claim_type": "safe_paper_claim",
            "claim": "Adaptive consistency can be retained as a reliability-gated refinement that avoids overcommitting to unstable fixed consistency.",
            "evidence": (
                f"Primary adaptive-minus-fixed-QC mean delta is {fmt(d_v6_v5['mean_delta'])} AUROC; "
                f"all-protocol adaptive-minus-fixed-QC mean delta is {fmt(all_v6_v5['mean_delta'])} AUROC."
            ),
            "paper_status": "use_with_caution",
        },
        {
            "claim_id": "C8",
            "claim_type": "final_recommendation",
            "claim": final_recommendation,
            "evidence": "This recommendation comes from Stage 16-B paper-facing comparison.",
            "paper_status": "use",
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, lineterminator="\n")

    lines = []
    lines += [
        "# Stage 16-C Final Method Claims",
        "",
        "## 1. Decision",
        "",
        "The final method should not be written as fixed QCR-U or as a consistency-driven method.",
        "",
        "The paper-facing method family is:",
        "",
        "```text",
        "Quality-Calibrated QCR",
        "```",
        "",
        "A more descriptive paper title/method phrase is:",
        "",
        "```text",
        "Quality-Calibrated Localization-Guided VLM Reasoning",
        "```",
        "",
        "The safest extended method name is:",
        "",
        "```text",
        f"{recommended_name}",
        "```",
        "",
        "## 2. Why this decision is necessary",
        "",
        "Stage 16-B shows that the adaptive variant is consistently better than naive fusion, but almost all useful gain comes from quality calibration.",
        "",
        "Adaptive consistency is retained only as a conservative gated refinement. It should not be described as the main performance source.",
        "",
        "## 3. Primary Protocol Evidence",
        "",
        "| Comparison | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    evidence_rows = [
        ("quality minus naive", d_v4_v3),
        ("adaptive minus naive", d_v6_v3),
        ("adaptive minus quality", d_v6_v4),
        ("fixed Q+C minus quality", d_v5_v4),
        ("adaptive minus fixed Q+C", d_v6_v5),
    ]

    for name, d in evidence_rows:
        lines.append(
            f"| {name} | {d['wins']} | {d['num_protocols']} | {d['win_rate']:.4f} | "
            f"{fmt(d['mean_delta'])} | {fmt(d['median_delta'])} | "
            f"{fmt(d['min_delta'])} | {fmt(d['max_delta'])} |"
        )

    lines += [
        "",
        "## 4. All-Protocol Evidence From Stage 16-B",
        "",
        "| Comparison | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    all_rows = [
        ("quality minus naive", all_v4_v3),
        ("adaptive minus naive", all_v6_v3),
        ("adaptive minus quality", all_v6_v4),
        ("adaptive minus fixed Q+C", all_v6_v5),
    ]

    for name, d in all_rows:
        lines.append(
            f"| {name} | {d['wins']} | {d['num_protocols']} | {float(d['win_rate']):.4f} | "
            f"{fmt(d['mean_delta'])} | {fmt(d['median_delta'])} | "
            f"{fmt(d['min_delta'])} | {fmt(d['max_delta'])} |"
        )

    lines += [
        "",
        "## 5. Final Claims Table",
        "",
        "| Claim ID | Type | Claim | Paper Status |",
        "|---|---|---|---|",
    ]

    for _, r in out.iterrows():
        lines.append(
            f"| {r['claim_id']} | {r['claim_type']} | {r['claim']} | {r['paper_status']} |"
        )

    lines += [
        "",
        "## 6. Safe Contribution Wording",
        "",
        "Use this wording in the paper:",
        "",
        "```text",
        "We propose a quality-calibrated localization-guided VLM reasoning framework for industrial anomaly recognition. Instead of directly fusing detector and VLM scores, the method calibrates crop-level VLM evidence using candidate quality derived from anomaly localization. We further analyze detector-VLM consistency and find that fixed consistency is not robust; therefore, consistency is used only as a conservative adaptive refinement.",
        "```",
        "",
        "## 7. Claims to Avoid",
        "",
        "- Do not claim fixed Q+C fusion is the final method.",
        "- Do not claim consistency is universally beneficial.",
        "- Do not claim adaptive consistency is the main source of improvement.",
        "- Do not claim full industrial anomaly understanding.",
        "- Do not claim manufacturing-cause reasoning.",
        "- Do not claim pixel-level segmentation SOTA.",
        "",
        "## 8. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Stage 16-D: Paper-facing final comparison table",
        "```",
        "",
        "Stage 16-D should compare the final method family against strong baselines and earlier fusion variants in one table.",
        "",
        "## 9. Outputs",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_DOC)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
