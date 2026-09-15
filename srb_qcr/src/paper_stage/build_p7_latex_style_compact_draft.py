from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(".").resolve()

IN_SYSTEM = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv"
IN_QCR = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv"
IN_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"
IN_BOUNDARY = ROOT / "results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv"
IN_E17 = ROOT / "results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv"

OUT_DIR = ROOT / "results/paper_p7"
DOC_DIR = ROOT / "docs/paper_p7"
TABLE_DIR = DOC_DIR / "tables"

OUT_DRAFT = DOC_DIR / "paper_p7_latex_style_compact_draft.md"
OUT_TABLE_INV = OUT_DIR / "paper_p7_compact_table_inventory.csv"
OUT_CHECKLIST = OUT_DIR / "paper_p7_latex_draft_checklist.csv"

OUT_TABLE1 = TABLE_DIR / "table1_system_baselines.md"
OUT_TABLE2 = TABLE_DIR / "table2_qcr_ablation.md"
OUT_TABLE3 = TABLE_DIR / "table3_boundary_summary.md"
OUT_TABLE4 = TABLE_DIR / "table4_efficientad_sensitivity.md"


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def fmt(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def signed(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):+.4f}"
    except Exception:
        return str(x)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_EMPTY TABLE_"

    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(fmt(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def write_table(path: Path, title: str, df: pd.DataFrame, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = []
    text += [f"# {title}", "", md_table(df), "", f"**Note.** {note}", ""]
    path.write_text("\n".join(text), encoding="utf-8", newline="\n")


def get_system_score(system: pd.DataFrame, method: str):
    rows = system[system["method"] == method]
    if rows.empty:
        return None
    return float(rows.iloc[0]["mean_image_auroc"])


def get_delta(deltas: pd.DataFrame, contains: str):
    rows = deltas[deltas["comparison"].astype(str).str.contains(contains, regex=False, na=False)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["delta"])


def build_table1(system: pd.DataFrame) -> pd.DataFrame:
    df = system.copy()
    df["mean_image_auroc"] = pd.to_numeric(df["mean_image_auroc"], errors="coerce")

    cols = [
        "rank_by_mean_image_auroc",
        "method",
        "mean_image_auroc",
        "paper_role",
        "fairness_tag",
    ]
    cols = [c for c in cols if c in df.columns]

    out = df[cols].copy()
    out = out.rename(
        columns={
            "rank_by_mean_image_auroc": "Rank",
            "method": "Method",
            "mean_image_auroc": "Mean AUROC",
            "paper_role": "Paper role",
            "fairness_tag": "Protocol tag",
        }
    )
    out["Mean AUROC"] = out["Mean AUROC"].map(fmt)
    return out


def build_table2(qcr: pd.DataFrame) -> pd.DataFrame:
    df = qcr.copy()
    df["image_auroc"] = pd.to_numeric(df["image_auroc"], errors="coerce")

    piv = df.pivot_table(
        index=["variant_id", "method", "paper_role"],
        columns="backbone",
        values="image_auroc",
        aggfunc="first",
    ).reset_index()

    piv.columns.name = None

    backbone_cols = [c for c in piv.columns if c not in ["variant_id", "method", "paper_role"]]
    piv["Mean"] = piv[backbone_cols].mean(axis=1)

    order = {"V0": 0, "V2": 1, "V3": 2, "V4": 3, "V5": 4, "V6": 5}
    piv["_order"] = piv["variant_id"].map(order).fillna(99)
    piv = piv.sort_values("_order").drop(columns=["_order"])

    rename = {
        "variant_id": "ID",
        "method": "Method",
        "paper_role": "Role",
        "Mean": "Mean AUROC",
    }
    out = piv.rename(columns=rename)

    for c in out.columns:
        if c not in ["ID", "Method", "Role"]:
            out[c] = out[c].map(fmt)

    return out


def build_table3(boundary: pd.DataFrame) -> pd.DataFrame:
    df = boundary.copy()

    checks = [
        (
            "Quality-Calibrated QCR - Naive fusion",
            "delta_v4_quality_minus_v3_naive",
            "Quality calibration effect",
        ),
        (
            "Adaptive refinement - Quality-Calibrated QCR",
            "delta_v6_adaptive_minus_v4_quality",
            "Adaptive consistency refinement effect",
        ),
        (
            "Fixed Q+C - Quality-Calibrated QCR",
            "delta_v5_fixed_minus_v4_quality",
            "Diagnostic fixed consistency effect",
        ),
        (
            "Adaptive refinement - Fixed Q+C",
            "delta_v6_adaptive_minus_v5_fixed",
            "Robustness tradeoff",
        ),
    ]

    rows = []
    for name, col, interp in checks:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append(
            {
                "Comparison": name,
                "Mean delta": signed(s.mean()) if len(s) else "",
                "Wins": f"{int((s > 0).sum())}/{len(s)}" if len(s) else "0/0",
                "Min": signed(s.min()) if len(s) else "",
                "Max": signed(s.max()) if len(s) else "",
                "Interpretation": interp,
            }
        )

    return pd.DataFrame(rows)


def build_table4(e17: pd.DataFrame) -> pd.DataFrame:
    df = e17.copy()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Metric": r["metric"],
                "EfficientAD-30": fmt(r["efficientad30_value"]),
                "EfficientAD-100": fmt(r["efficientad100_value"]),
                "Delta 100-30": signed(r["delta_100_minus_30"]),
            }
        )
    return pd.DataFrame(rows)


def make_inventory() -> pd.DataFrame:
    rows = [
        {
            "table_id": "Table 1",
            "title": "System-level strong baseline comparison",
            "path": str(OUT_TABLE1.relative_to(ROOT)),
            "paper_location": "Main Results",
            "status": "generated",
        },
        {
            "table_id": "Table 2",
            "title": "QCR primary-protocol ablation",
            "path": str(OUT_TABLE2.relative_to(ROOT)),
            "paper_location": "Ablation Study",
            "status": "generated",
        },
        {
            "table_id": "Table 3",
            "title": "Boundary and robustness summary",
            "path": str(OUT_TABLE3.relative_to(ROOT)),
            "paper_location": "Failure / Boundary Analysis",
            "status": "generated",
        },
        {
            "table_id": "Appendix Table A1",
            "title": "EfficientAD-100 fruit_jelly sensitivity",
            "path": str(OUT_TABLE4.relative_to(ROOT)),
            "paper_location": "Appendix / Baseline Budget Sensitivity",
            "status": "generated",
        },
    ]
    return pd.DataFrame(rows)


def make_checklist() -> pd.DataFrame:
    rows = [
        {
            "item_id": "P7-C1",
            "item": "Compact draft generated",
            "status": "done",
            "next_action": "Manual polish after figures and references are added.",
        },
        {
            "item_id": "P7-C2",
            "item": "Paper-facing compact tables generated",
            "status": "done",
            "next_action": "Convert Markdown tables to LaTeX tabular/booktabs.",
        },
        {
            "item_id": "P7-C3",
            "item": "Framework figure missing",
            "status": "missing",
            "next_action": "Create Figure 1 pipeline schematic.",
        },
        {
            "item_id": "P7-C4",
            "item": "Boundary case figure missing",
            "status": "missing",
            "next_action": "Inspect Stage 16-E case inventory and select representative images.",
        },
        {
            "item_id": "P7-C5",
            "item": "BibTeX missing",
            "status": "missing",
            "next_action": "Prepare references for MVTec AD, VisA, PatchCore, FastFlow, EfficientAD, CLIP, WinCLIP, AnomalyCLIP.",
        },
        {
            "item_id": "P7-C6",
            "item": "AnomalyCLIP risk unresolved",
            "status": "open",
            "next_action": "Either run it later or keep it explicitly as limitation.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    system = read_csv_strict(IN_SYSTEM)
    qcr = read_csv_strict(IN_QCR)
    deltas = read_csv_strict(IN_DELTAS)
    boundary = read_csv_strict(IN_BOUNDARY)
    e17 = read_csv_strict(IN_E17)

    table1 = build_table1(system)
    table2 = build_table2(qcr)
    table3 = build_table3(boundary)
    table4 = build_table4(e17)

    write_table(
        OUT_TABLE1,
        "Table 1. System-level strong baseline comparison",
        table1,
        "LOCO is the fair system-level result. Same-set is an upper-bound diagnostic only.",
    )
    write_table(
        OUT_TABLE2,
        "Table 2. QCR primary-protocol ablation",
        table2,
        "Quality-Calibrated QCR is the main method core. Fixed Q+C is diagnostic only.",
    )
    write_table(
        OUT_TABLE3,
        "Table 3. Boundary and robustness summary",
        table3,
        "Per-category deltas show quality calibration is useful but not universal; adaptive consistency is only a refinement.",
    )
    write_table(
        OUT_TABLE4,
        "Appendix Table A1. EfficientAD-100 fruit_jelly sensitivity",
        table4,
        "This is a defensive fixed-budget sensitivity check, not a full EfficientAD sweep.",
    )

    inventory = make_inventory()
    checklist = make_checklist()
    inventory.to_csv(OUT_TABLE_INV, index=False, lineterminator="\n")
    checklist.to_csv(OUT_CHECKLIST, index=False, lineterminator="\n")

    full_vlm = get_system_score(system, "full-image VLM")
    context_vlm = get_system_score(system, "context-aware VLM")
    winclip = get_system_score(system, "WinCLIP fixed protocol")
    efficientad30 = get_system_score(system, "EfficientAD-30 fixed-budget")
    patchcore = get_system_score(system, "PatchCore")
    loco = get_system_score(system, "PatchCore + context VLM, LOCO")
    same_set = get_system_score(system, "PatchCore + context VLM, same-set")

    d_loco_patch = get_delta(deltas, "LOCO fusion vs PatchCore")
    d_loco_ead = get_delta(deltas, "LOCO fusion vs EfficientAD-30")
    d_quality_naive = get_delta(deltas, "Quality-Calibrated QCR vs naive fusion")
    d_adaptive_quality = get_delta(deltas, "Adaptive refinement vs Quality-Calibrated QCR")
    d_adaptive_naive = get_delta(deltas, "Adaptive refinement vs naive fusion")

    qcr_mean = qcr.groupby("variant_id")["image_auroc"].mean().to_dict()
    e17_image = e17[e17["metric"] == "image_AUROC"].iloc[0].to_dict()

    lines = []
    lines += [
        "# Paper Stage P7: LaTeX-style Compact Draft",
        "",
        "## Title",
        "",
        "**Quality-Calibrated Localization-Guided VLM Reasoning for Industrial Anomaly Recognition**",
        "",
        "## Abstract",
        "",
        "Industrial anomaly recognition with general-purpose vision-language models remains unreliable when defects are small, localized, and visually subtle. "
        "We propose Quality-Calibrated QCR, a localization-guided VLM reasoning framework that converts detector localization evidence into candidate-level visual-language evidence and calibrates crop-level VLM anomaly scores using candidate quality. "
        "The method treats candidate quality as the main reliability mechanism and uses detector-VLM consistency only as a conservative adaptive refinement. "
        "Experiments with detector, VLM, and external anomaly baselines show that localization-guided VLM evidence complements detector scores. "
        "Ablations further show that quality calibration provides the main gain over naive detector-crop fusion, while adaptive consistency yields only a small refinement and fixed consistency should remain diagnostic. "
        "Failure and boundary analyses clarify when quality calibration helps, when it fails, and why broad claims about segmentation or manufacturing-cause reasoning are unsupported.",
        "",
        "## 1. Introduction",
        "",
        "Industrial anomalies often occupy small localized regions and may not dominate global image semantics. "
        "This makes direct full-image VLM reasoning unreliable for industrial inspection. "
        "A detector can provide localization evidence, while a VLM can provide visual-language abnormality evidence; however, naive fusion does not model whether the localized crop is reliable. "
        "We address this gap by calibrating crop-level VLM scores with candidate quality derived from localization evidence.",
        "",
        "Our method, **Quality-Calibrated QCR**, uses detector localization to generate candidate crops, obtains crop-level VLM anomaly evidence, and modulates that evidence using candidate quality. "
        "We also study detector-VLM consistency, but robustness analysis shows that fixed consistency is not reliable enough to be the final method. "
        "Thus, consistency is retained only as a small adaptive refinement.",
        "",
        "Our contributions are:",
        "",
        "1. We formulate industrial anomaly recognition as localization-guided VLM evidence reasoning.",
        "2. We propose candidate-quality calibration as the main reliability mechanism for crop-level VLM evidence.",
        "3. We analyze fixed and adaptive detector-VLM consistency and show why consistency must be treated conservatively.",
        "4. We provide strong baseline comparisons, ablations, boundary analysis, and fixed-budget sensitivity checks with explicit claim restrictions.",
        "",
        "## 2. Related Work",
        "",
        "**Industrial anomaly detection.** PatchCore, FastFlow-style detectors, and EfficientAD provide strong anomaly detection and localization evidence. Our method does not replace them; it uses localization evidence to produce candidate-level VLM evidence.",
        "",
        "**Vision-language anomaly detection.** CLIP-based anomaly methods such as WinCLIP and AnomalyCLIP show the value of language-supervised representations. Our work is narrower: it studies how detector-guided crop evidence should be calibrated before being trusted by a VLM-based anomaly scorer.",
        "",
        "**Reliability calibration.** Naive score fusion assumes detector and VLM scores are equally reliable. We show that candidate quality is the stable calibration signal, whereas fixed consistency is not robust enough to be the final method.",
        "",
        "## 3. Method",
        "",
        "Given image $x$, a detector produces localization evidence $A$ and normalized anomaly score $D$. Candidate crops $C=\\{c_i\\}$ are generated from $A$. The VLM scores each crop and the crop-level scores are aggregated into $M$. Candidate quality $Q$ measures the reliability of the localized crop evidence.",
        "",
        "The naive detector-crop fusion baseline is:",
        "",
        "$$S_{\\mathrm{naive}} = 0.5D + 0.5M.$$",
        "",
        "Quality-Calibrated QCR modulates the VLM contribution by candidate quality:",
        "",
        "$$S_{\\mathrm{quality}} = 0.5D + 0.5M(0.5 + 0.5Q).$$",
        "",
        "A diagnostic fixed Q+C variant is:",
        "",
        "$$S_{\\mathrm{fixed}} = 0.4D + 0.4M + 0.1Q + 0.1K,$$",
        "",
        "where $K$ is detector-VLM high-high consistency. This variant is diagnostic only because fixed consistency is not robust across protocols.",
        "",
        "The adaptive refinement uses a conservative gate:",
        "",
        "$$g = QK(1-|D-M|)\\min(D,M),$$",
        "",
        "$$S_{\\mathrm{adaptive}} = S_{\\mathrm{quality}} + 0.05g.$$",
        "",
        "The main method is $S_{\\mathrm{quality}}$. The adaptive variant is reported as a refinement, not as the main performance source.",
        "",
        "## 4. Experiments",
        "",
        "We evaluate two complementary views. The system-level view compares VLM baselines, detector baselines, and localization-guided fusion. The QCR view isolates detector-only scoring, crop VLM scoring, naive fusion, quality calibration, fixed Q+C, and adaptive refinement under the QCR primary protocol. Image AUROC is the main metric; pixel metrics are auxiliary only.",
        "",
        "## 5. Main Results",
        "",
        f"Full-image VLM obtains mean AUROC `{fmt(full_vlm)}`, while context-aware VLM obtains `{fmt(context_vlm)}`. "
        f"PatchCore obtains `{fmt(patchcore)}` and EfficientAD-30 fixed-budget obtains `{fmt(efficientad30)}`. "
        f"The fair LOCO fusion reaches `{fmt(loco)}`, improving over PatchCore by `{signed(d_loco_patch)}` and over EfficientAD-30 by `{signed(d_loco_ead)}`. "
        f"The same-set result `{fmt(same_set)}` is reported only as an upper-bound diagnostic.",
        "",
        OUT_TABLE1.read_text(encoding="utf-8"),
        "",
        "## 6. Ablation Study",
        "",
        f"Under the QCR primary protocol, naive detector-crop fusion obtains mean AUROC `{fmt(qcr_mean.get('V3'))}`. "
        f"Quality-Calibrated QCR obtains `{fmt(qcr_mean.get('V4'))}`, a gain of `{signed(d_quality_naive)}`. "
        f"The adaptive refinement obtains `{fmt(qcr_mean.get('V6'))}`, improving over naive fusion by `{signed(d_adaptive_naive)}` but over the quality core by only `{signed(d_adaptive_quality)}`. "
        "Therefore, the main effective component is candidate quality calibration, not adaptive consistency.",
        "",
        OUT_TABLE2.read_text(encoding="utf-8"),
        "",
        "## 7. Failure and Boundary Analysis",
        "",
        "Boundary analysis shows that quality calibration is useful but not universal. It can help by boosting true localized anomalies or suppressing normal false positives, but it can also fail when candidate quality is misleading or detector-VLM evidence disagrees. Fixed Q+C can peak in some cases but is not robust enough to be final.",
        "",
        OUT_TABLE3.read_text(encoding="utf-8"),
        "",
        "## 8. EfficientAD Budget Sensitivity",
        "",
        f"A 100-epoch EfficientAD sensitivity check on fruit_jelly gives image-AUROC delta `{signed(e17_image['delta_100_minus_30'])}` relative to EfficientAD-30. "
        "This does not indicate severe image-level underestimation of EfficientAD-30, but it remains a fixed-budget baseline and should not be described as full EfficientAD defeat.",
        "",
        OUT_TABLE4.read_text(encoding="utf-8"),
        "",
        "## 9. Limitations",
        "",
        "The method is image-level anomaly recognition and candidate-level VLM evidence calibration. It does not claim pixel-level segmentation SOTA, manufacturing-cause reasoning, or full anomaly understanding. EfficientAD is fixed-budget, AnomalyCLIP is not included, and adaptive consistency has only a small gain over quality-only.",
        "",
        "## 10. Conclusion",
        "",
        "Quality-Calibrated QCR provides a reliability-calibrated bridge between industrial anomaly localization and VLM anomaly evidence. Candidate quality is the main effective component, while adaptive consistency is a conservative refinement. The evidence supports a cautious but coherent claim: localization-guided VLM anomaly recognition becomes more reliable when crop-level evidence is calibrated by candidate quality.",
        "",
        "## Appendix A. Compact Table Inventory",
        "",
        md_table(inventory),
        "",
        "## Appendix B. Draft Checklist",
        "",
        md_table(checklist),
        "",
    ]

    OUT_DRAFT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_DRAFT)
    print("[DONE]", OUT_TABLE1)
    print("[DONE]", OUT_TABLE2)
    print("[DONE]", OUT_TABLE3)
    print("[DONE]", OUT_TABLE4)
    print("[DONE]", OUT_TABLE_INV)
    print("[DONE]", OUT_CHECKLIST)


if __name__ == "__main__":
    main()
