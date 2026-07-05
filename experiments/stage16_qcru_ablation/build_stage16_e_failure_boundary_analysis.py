from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

IN_PRED = ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv"
IN_STAGE16D_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_CASES = OUT_DIR / "stage16_e_failure_boundary_case_inventory.csv"
OUT_CATEGORY = OUT_DIR / "stage16_e_category_boundary_summary.csv"
OUT_DECISION = OUT_DIR / "stage16_e_boundary_decision_summary.csv"
OUT_DOC = DOC_DIR / "stage16_e_failure_boundary_analysis_report.md"


REQUIRED_COLUMNS = [
    "backbone",
    "dataset",
    "category",
    "strategy",
    "eval_mode",
    "image_key",
    "is_anomaly_final",
    "vlm_score_norm",
    "detector_score_norm",
    "candidate_quality_norm",
    "high_high_consistency",
]


PRIMARY_FILTER = {
    "dataset": "VisA",
    "strategy": "inspection_binary",
    "eval_mode": "crop_topk_ensemble",
}


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
    return df


def minmax_safe(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    lo = x.min()
    hi = x.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)

    i = 0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    positives = y.sum()
    if positives <= 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / positives)


def eval_binary(y_true: pd.Series, y_score: pd.Series) -> dict:
    y = pd.to_numeric(y_true, errors="coerce").astype(int).to_numpy()
    s = pd.to_numeric(y_score, errors="coerce").astype(float).to_numpy()

    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]

    num_anomaly = int(y.sum())
    num_normal = int(len(y) - num_anomaly)
    has_both = num_anomaly > 0 and num_normal > 0

    if not has_both:
        return {
            "num_images": int(len(y)),
            "num_normal": num_normal,
            "num_anomaly": num_anomaly,
            "auroc": float("nan"),
            "ap": float("nan"),
        }

    return {
        "num_images": int(len(y)),
        "num_normal": num_normal,
        "num_anomaly": num_anomaly,
        "auroc": roc_auc_score_np(y, s),
        "ap": average_precision_score_np(y, s),
    }


def build_base_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    optional_cols = [
        "fallback",
        "has_candidate",
        "num_candidates",
        "image_path",
        "gt_label",
        "defect_type",
    ]

    base_cols = REQUIRED_COLUMNS + [c for c in optional_cols if c in df.columns]
    base = df[base_cols].copy()

    base = base.drop_duplicates(
        subset=["backbone", "dataset", "category", "strategy", "eval_mode", "image_key"]
    ).reset_index(drop=True)

    for c in [
        "is_anomaly_final",
        "vlm_score_norm",
        "detector_score_norm",
        "candidate_quality_norm",
        "high_high_consistency",
        "num_candidates",
    ]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    base["D"] = base["detector_score_norm"].fillna(0.0)
    base["M"] = base["vlm_score_norm"].fillna(0.0)
    base["Q"] = base["candidate_quality_norm"].fillna(0.0)
    base["K"] = base["high_high_consistency"].fillna(0.0)

    base["score_naive"] = 0.5 * base["D"] + 0.5 * base["M"]

    base["score_quality_raw"] = (
        0.5 * base["D"]
        + 0.5 * (base["M"] * (0.5 + 0.5 * base["Q"]))
    )

    base["score_fixed_qc_raw"] = (
        0.40 * base["D"]
        + 0.40 * base["M"]
        + 0.10 * base["Q"]
        + 0.10 * base["K"]
    )

    agreement = (1.0 - (base["D"] - base["M"]).abs()).clip(lower=0.0, upper=1.0)
    mutual_anomaly_evidence = np.minimum(base["D"], base["M"])
    adaptive_gate = base["Q"] * base["K"] * agreement * mutual_anomaly_evidence

    base["agreement"] = agreement
    base["mutual_anomaly_evidence"] = mutual_anomaly_evidence
    base["adaptive_gate"] = adaptive_gate
    base["score_adaptive_raw"] = base["score_quality_raw"] + 0.05 * adaptive_gate

    group_cols = ["backbone", "dataset", "strategy", "eval_mode"]
    for raw_col, out_col in [
        ("score_quality_raw", "score_quality"),
        ("score_fixed_qc_raw", "score_fixed_qc"),
        ("score_adaptive_raw", "score_adaptive"),
    ]:
        base[out_col] = base.groupby(group_cols, dropna=False)[raw_col].transform(minmax_safe)

    base["delta_quality_minus_naive"] = base["score_quality"] - base["score_naive"]
    base["delta_fixed_minus_quality"] = base["score_fixed_qc"] - base["score_quality"]
    base["delta_adaptive_minus_quality"] = base["score_adaptive"] - base["score_quality"]
    base["delta_adaptive_minus_fixed"] = base["score_adaptive"] - base["score_fixed_qc"]
    base["detector_vlm_disagreement"] = (base["D"] - base["M"]).abs()

    return base


def filter_primary(base: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=base.index)
    for k, v in PRIMARY_FILTER.items():
        mask &= base[k] == v
    primary = base[mask].copy()
    if primary.empty:
        print("[WARN] primary filter returned empty; using full base table.")
        return base.copy()
    return primary


def select_top_cases(g: pd.DataFrame, case_type: str, sort_col: str, ascending: bool, n: int = 5) -> pd.DataFrame:
    cols = [
        "backbone",
        "dataset",
        "strategy",
        "eval_mode",
        "category",
        "image_key",
        "is_anomaly_final",
        "D",
        "M",
        "Q",
        "K",
        "agreement",
        "mutual_anomaly_evidence",
        "adaptive_gate",
        "score_naive",
        "score_quality",
        "score_fixed_qc",
        "score_adaptive",
        "delta_quality_minus_naive",
        "delta_fixed_minus_quality",
        "delta_adaptive_minus_quality",
        "delta_adaptive_minus_fixed",
        "detector_vlm_disagreement",
    ]
    cols += [c for c in ["image_path", "gt_label", "defect_type", "has_candidate", "num_candidates", "fallback"] if c in g.columns]
    cols = [c for c in cols if c in g.columns]

    out = g.sort_values(sort_col, ascending=ascending).head(n)[cols].copy()
    out.insert(0, "case_type", case_type)
    out.insert(1, "selection_metric", sort_col)
    out.insert(2, "selection_order", "ascending" if ascending else "descending")
    return out


def build_case_inventory(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for keys, g in primary.groupby(["backbone", "dataset", "strategy", "eval_mode"], dropna=False):
        anomaly = g[g["is_anomaly_final"] == 1].copy()
        normal = g[g["is_anomaly_final"] == 0].copy()

        if not anomaly.empty:
            rows.append(
                select_top_cases(
                    anomaly,
                    "quality_helps_anomaly_boost",
                    "delta_quality_minus_naive",
                    ascending=False,
                )
            )
            rows.append(
                select_top_cases(
                    anomaly,
                    "quality_boundary_anomaly_suppression",
                    "delta_quality_minus_naive",
                    ascending=True,
                )
            )
            rows.append(
                select_top_cases(
                    anomaly,
                    "fixed_consistency_boundary_anomaly_suppression",
                    "delta_fixed_minus_quality",
                    ascending=True,
                )
            )

        if not normal.empty:
            rows.append(
                select_top_cases(
                    normal,
                    "quality_helps_normal_suppression",
                    "delta_quality_minus_naive",
                    ascending=True,
                )
            )
            rows.append(
                select_top_cases(
                    normal,
                    "quality_boundary_normal_boost",
                    "delta_quality_minus_naive",
                    ascending=False,
                )
            )
            rows.append(
                select_top_cases(
                    normal,
                    "fixed_consistency_boundary_normal_boost",
                    "delta_fixed_minus_quality",
                    ascending=False,
                )
            )

        rows.append(
            select_top_cases(
                g,
                "adaptive_refinement_high_gate",
                "adaptive_gate",
                ascending=False,
            )
        )
        rows.append(
            select_top_cases(
                g,
                "detector_vlm_disagreement_boundary",
                "detector_vlm_disagreement",
                ascending=False,
            )
        )

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out


def build_category_summary(primary: pd.DataFrame) -> pd.DataFrame:
    variants = [
        ("V3", "naive_detector_crop_fusion", "score_naive"),
        ("V4", "Quality-Calibrated QCR", "score_quality"),
        ("V5", "Fixed Q+C fusion", "score_fixed_qc"),
        ("V6", "Quality-Calibrated QCR + adaptive consistency refinement", "score_adaptive"),
    ]

    rows = []
    group_cols = ["backbone", "dataset", "strategy", "eval_mode", "category"]

    for keys, g in primary.groupby(group_cols, dropna=False):
        base_row = dict(zip(group_cols, keys))

        for variant_id, method, score_col in variants:
            m = eval_binary(g["is_anomaly_final"], g[score_col])
            row = base_row.copy()
            row.update(
                {
                    "variant_id": variant_id,
                    "method": method,
                    "score_col": score_col,
                    **m,
                }
            )
            rows.append(row)

    long = pd.DataFrame(rows)

    idx = group_cols
    piv = long.pivot_table(index=idx, columns="variant_id", values="auroc", aggfunc="first").reset_index()
    piv.columns.name = None

    for col in ["V3", "V4", "V5", "V6"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv["delta_v4_quality_minus_v3_naive"] = piv["V4"] - piv["V3"]
    piv["delta_v6_adaptive_minus_v4_quality"] = piv["V6"] - piv["V4"]
    piv["delta_v5_fixed_minus_v4_quality"] = piv["V5"] - piv["V4"]
    piv["delta_v6_adaptive_minus_v5_fixed"] = piv["V6"] - piv["V5"]

    def boundary_label(r):
        labels = []
        if pd.notna(r["delta_v4_quality_minus_v3_naive"]) and r["delta_v4_quality_minus_v3_naive"] <= 0:
            labels.append("quality_not_helpful")
        if pd.notna(r["delta_v6_adaptive_minus_v4_quality"]) and abs(r["delta_v6_adaptive_minus_v4_quality"]) < 0.001:
            labels.append("adaptive_gain_negligible")
        if pd.notna(r["delta_v5_fixed_minus_v4_quality"]) and r["delta_v5_fixed_minus_v4_quality"] > 0:
            labels.append("fixed_consistency_can_peak_but_diagnostic")
        if pd.notna(r["V6"]) and r["V6"] < 0.90:
            labels.append("low_absolute_qcr_auc")
        return ";".join(labels) if labels else "no_major_boundary"

    piv["boundary_label"] = piv.apply(boundary_label, axis=1)

    return piv.sort_values(
        ["backbone", "delta_v4_quality_minus_v3_naive", "delta_v6_adaptive_minus_v4_quality"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_decision_summary(category_summary: pd.DataFrame, case_inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(decision_id, topic, decision, evidence, paper_action):
        rows.append(
            {
                "decision_id": decision_id,
                "topic": topic,
                "decision": decision,
                "evidence": evidence,
                "paper_action": paper_action,
            }
        )

    q_delta = category_summary["delta_v4_quality_minus_v3_naive"].dropna()
    a_delta = category_summary["delta_v6_adaptive_minus_v4_quality"].dropna()
    f_delta = category_summary["delta_v5_fixed_minus_v4_quality"].dropna()

    add(
        "E1",
        "quality_calibration",
        "Keep candidate quality calibration as the main method core.",
        f"Per-category mean V4-V3 AUROC delta={q_delta.mean():+.4f}; wins={(q_delta > 0).sum()}/{len(q_delta)}.",
        "Use as main contribution.",
    )

    add(
        "E2",
        "adaptive_consistency",
        "Keep adaptive consistency only as a refinement.",
        f"Per-category mean V6-V4 AUROC delta={a_delta.mean():+.4f}; wins={(a_delta > 0).sum()}/{len(a_delta)}.",
        "Use with caution; do not call it the main source of improvement.",
    )

    add(
        "E3",
        "fixed_consistency",
        "Do not use fixed Q+C as the final method even if it peaks on some categories.",
        f"Per-category mean V5-V4 AUROC delta={f_delta.mean():+.4f}; positive cases={(f_delta > 0).sum()}/{len(f_delta)}.",
        "Mention as diagnostic only.",
    )

    if not case_inventory.empty:
        counts = case_inventory["case_type"].value_counts().to_dict()
        add(
            "E4",
            "case_inventory",
            "Use selected cases for qualitative boundary analysis.",
            "; ".join([f"{k}={v}" for k, v in counts.items()]),
            "Inspect representative cases manually before paper figures.",
        )

    add(
        "E5",
        "paper_boundary",
        "The method should be claimed as reliability calibration, not full anomaly understanding.",
        "The case taxonomy explicitly includes detector-VLM disagreement and candidate-quality boundary cases.",
        "Use boundary-aware wording in paper.",
    )

    return pd.DataFrame(rows)


def write_report(category_summary: pd.DataFrame, case_inventory: pd.DataFrame, decision: pd.DataFrame) -> None:
    lines = []
    lines += [
        "# Stage 16-E Failure Cases and Boundary Analysis",
        "",
        "## 1. Purpose",
        "",
        "Stage 16-D created the paper-facing main comparison. Stage 16-E explains method boundaries.",
        "",
        "This stage does not train models or rerun VLM inference. It mines the existing Stage 9 prediction table for representative boundary cases.",
        "",
        "## 2. Primary Scope",
        "",
        "The case inventory uses the QCR primary protocol:",
        "",
        "```text",
        "dataset = VisA",
        "strategy = inspection_binary",
        "eval_mode = crop_topk_ensemble",
        "```",
        "",
        "## 3. Category-level Boundary Summary",
        "",
        "| Backbone | Category | V3 Naive | V4 Quality | V5 Fixed Q+C | V6 Adaptive | V4-V3 | V6-V4 | V5-V4 | Boundary Label |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for _, r in category_summary.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['category']} | "
            f"{r['V3']:.4f} | {r['V4']:.4f} | {r['V5']:.4f} | {r['V6']:.4f} | "
            f"{r['delta_v4_quality_minus_v3_naive']:+.4f} | "
            f"{r['delta_v6_adaptive_minus_v4_quality']:+.4f} | "
            f"{r['delta_v5_fixed_minus_v4_quality']:+.4f} | "
            f"{r['boundary_label']} |"
        )

    lines += [
        "",
        "## 4. Case Types Extracted",
        "",
        "| Case Type | Meaning | Paper Use |",
        "|---|---|---|",
        "| quality_helps_anomaly_boost | anomaly images whose score is boosted by quality calibration | positive qualitative example |",
        "| quality_helps_normal_suppression | normal images suppressed by quality calibration | false-positive reduction example |",
        "| quality_boundary_anomaly_suppression | anomaly images suppressed by quality calibration | boundary / failure case |",
        "| quality_boundary_normal_boost | normal images boosted by quality calibration | boundary / failure case |",
        "| fixed_consistency_boundary_anomaly_suppression | anomaly images where fixed consistency hurts | explains why fixed Q+C is not final |",
        "| fixed_consistency_boundary_normal_boost | normal images where fixed consistency increases risk | explains false-positive boundary |",
        "| adaptive_refinement_high_gate | images with strongest adaptive gate | explains refinement behavior |",
        "| detector_vlm_disagreement_boundary | images with high detector/VLM disagreement | explains detector-VLM conflict |",
        "",
    ]

    if case_inventory.empty:
        lines.append("No case inventory generated.")
    else:
        counts = case_inventory["case_type"].value_counts().reset_index()
        counts.columns = ["case_type", "count"]
        lines += [
            "Case counts:",
            "",
            "| Case Type | Count |",
            "|---|---:|",
        ]
        for _, r in counts.iterrows():
            lines.append(f"| {r['case_type']} | {int(r['count'])} |")

    lines += [
        "",
        "## 5. Boundary Decisions",
        "",
        "| Decision ID | Topic | Decision | Paper Action |",
        "|---|---|---|---|",
    ]

    for _, r in decision.iterrows():
        lines.append(
            f"| {r['decision_id']} | {r['topic']} | {r['decision']} | {r['paper_action']} |"
        )

    lines += [
        "",
        "## 6. Paper Interpretation",
        "",
        "The correct interpretation is:",
        "",
        "```text",
        "Quality calibration is the main reliability mechanism. It helps when candidate quality aligns with true localized anomaly evidence, but it can still fail when localization quality is misleading or when the VLM and detector disagree. Fixed consistency can produce high peak AUROC in the primary protocol, but it is not robust enough to be the final method. Adaptive consistency is retained only as a conservative refinement.",
        "```",
        "",
        "## 7. Claims to Avoid",
        "",
        "- Do not claim the method solves all detector localization errors.",
        "- Do not claim consistency is universally beneficial.",
        "- Do not claim adaptive consistency is the main source of improvement.",
        "- Do not claim pixel-level segmentation SOTA.",
        "- Do not claim manufacturing-cause understanding.",
        "",
        "## 8. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Stage 16-F: final claim-evidence map",
        "```",
        "",
        "Stage 16-F should map every paper claim to the exact table/result that supports it.",
        "",
        "## 9. Outputs",
        "",
        f"- `{OUT_CASES.relative_to(ROOT)}`",
        f"- `{OUT_CATEGORY.relative_to(ROOT)}`",
        f"- `{OUT_DECISION.relative_to(ROOT)}`",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    pred = read_csv_strict(IN_PRED)
    _ = read_csv_strict(IN_STAGE16D_DELTAS) if IN_STAGE16D_DELTAS.exists() else None

    base = build_base_table(pred)
    primary = filter_primary(base)

    case_inventory = build_case_inventory(primary)
    category_summary = build_category_summary(primary)
    decision = build_decision_summary(category_summary, case_inventory)

    case_inventory.to_csv(OUT_CASES, index=False, lineterminator="\n")
    category_summary.to_csv(OUT_CATEGORY, index=False, lineterminator="\n")
    decision.to_csv(OUT_DECISION, index=False, lineterminator="\n")

    write_report(category_summary, case_inventory, decision)

    print("[DONE]", OUT_CASES)
    print("[DONE]", OUT_CATEGORY)
    print("[DONE]", OUT_DECISION)
    print("[DONE]", OUT_DOC)
    print()
    print("===== category summary =====")
    print(category_summary.to_string(index=False))
    print()
    print("===== decision summary =====")
    print(decision.to_string(index=False))
    print()
    print("===== case type counts =====")
    if case_inventory.empty:
        print("EMPTY")
    else:
        print(case_inventory["case_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
