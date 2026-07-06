from __future__ import annotations

from pathlib import Path
import re
import shutil
import pandas as pd


ROOT = Path(".").resolve()

IN_SYSTEM = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_system_baseline_table.csv"
IN_QCR = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv"
IN_DELTAS = ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_claim_ready_deltas.csv"
IN_BOUNDARY = ROOT / "results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv"
IN_E17 = ROOT / "results/stage17_defensive_sensitivity/stage17_a_efficientad100_vs_30_delta.csv"

IN_P9_BIB = ROOT / "docs/paper_p9/references.bib"
IN_FIG1_SVG = ROOT / "docs/paper_p8/figures/figure1_framework_schematic.svg"

OUT_DIR = ROOT / "results/paper_p10"
DOC_DIR = ROOT / "docs/paper_p10"

PAPER_DIR = ROOT / "paper/quality_calibrated_qcr"
TABLE_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"

OUT_MAIN_TEX = PAPER_DIR / "main.tex"
OUT_BIB = PAPER_DIR / "references.bib"

OUT_TABLE1 = TABLE_DIR / "table1_system_baselines.tex"
OUT_TABLE2 = TABLE_DIR / "table2_qcr_ablation.tex"
OUT_TABLE3 = TABLE_DIR / "table3_boundary_summary.tex"
OUT_TABLE4 = TABLE_DIR / "table4_efficientad_sensitivity.tex"

OUT_FIG1_COPY = FIG_DIR / "figure1_framework_schematic.svg"

OUT_ARTIFACTS = OUT_DIR / "paper_p10_latex_artifact_inventory.csv"
OUT_CHECKLIST = OUT_DIR / "paper_p10_compile_and_submission_checklist.csv"
OUT_REPORT = DOC_DIR / "paper_p10_latex_scaffold_report.md"


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


def latex_escape(x) -> str:
    s = "" if pd.isna(x) else str(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


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


def write_booktabs_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    note: str,
    align: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if align is None:
        align = "l" * len(df.columns)

    lines = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            vals.append(latex_escape(row[c]))
        lines.append(" & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{2pt}",
        rf"\footnotesize{{\emph{{Note.}} {latex_escape(note)}}}",
        r"\end{table}",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build_table1(system: pd.DataFrame) -> pd.DataFrame:
    df = system.copy()
    df["mean_image_auroc"] = pd.to_numeric(df["mean_image_auroc"], errors="coerce")

    if "rank_by_mean_image_auroc" in df.columns:
        df = df.sort_values("rank_by_mean_image_auroc")

    out = pd.DataFrame()
    out["Method"] = df["method"]
    out["Mean AUROC"] = df["mean_image_auroc"].map(fmt)
    out["Role"] = df["paper_role"] if "paper_role" in df.columns else ""
    out["Protocol"] = df["fairness_tag"] if "fairness_tag" in df.columns else ""
    return out


def build_table2(qcr: pd.DataFrame) -> pd.DataFrame:
    df = qcr.copy()
    df["image_auroc"] = pd.to_numeric(df["image_auroc"], errors="coerce")

    index_cols = [c for c in ["variant_id", "method", "paper_role"] if c in df.columns]
    if "backbone" not in df.columns:
        raise RuntimeError("QCR table missing backbone column.")

    piv = df.pivot_table(
        index=index_cols,
        columns="backbone",
        values="image_auroc",
        aggfunc="first",
    ).reset_index()

    piv.columns.name = None
    backbone_cols = [c for c in piv.columns if c not in index_cols]
    piv["Mean"] = piv[backbone_cols].mean(axis=1)

    order = {"V0": 0, "V1": 1, "V2": 2, "V3": 3, "V4": 4, "V5": 5, "V6": 6}
    if "variant_id" in piv.columns:
        piv["_order"] = piv["variant_id"].map(order).fillna(99)
        piv = piv.sort_values("_order").drop(columns=["_order"])

    out = pd.DataFrame()
    out["ID"] = piv["variant_id"] if "variant_id" in piv.columns else ""
    out["Method"] = piv["method"] if "method" in piv.columns else ""
    for c in backbone_cols:
        out[c] = piv[c].map(fmt)
    out["Mean AUROC"] = piv["Mean"].map(fmt)
    out["Role"] = piv["paper_role"] if "paper_role" in piv.columns else ""
    return out


def build_table3(boundary: pd.DataFrame) -> pd.DataFrame:
    df = boundary.copy()

    checks = [
        (
            "Quality-Calibrated QCR - Naive",
            "delta_v4_quality_minus_v3_naive",
            "quality calibration",
        ),
        (
            "Adaptive - Quality-Calibrated QCR",
            "delta_v6_adaptive_minus_v4_quality",
            "adaptive refinement",
        ),
        (
            "Fixed Q+C - Quality-Calibrated QCR",
            "delta_v5_fixed_minus_v4_quality",
            "diagnostic fixed consistency",
        ),
        (
            "Adaptive - Fixed Q+C",
            "delta_v6_adaptive_minus_v5_fixed",
            "robustness tradeoff",
        ),
    ]

    rows = []
    for name, col, interpretation in checks:
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
                "Interpretation": interpretation,
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
                "Delta": signed(r["delta_100_minus_30"]),
            }
        )
    return pd.DataFrame(rows)


def normalize_bib() -> str:
    if not IN_P9_BIB.exists():
        raise FileNotFoundError(IN_P9_BIB)

    text = IN_P9_BIB.read_text(encoding="utf-8").strip()

    # If the file was flattened into one line, make it readable again.
    text = re.sub(r"}\s+@", "}\n\n@", text)
    text = re.sub(r"(@\w+\{[^,]+),\s+", r"\1,\n  ", text)
    text = re.sub(
        r",\s+(title|author|booktitle|journal|year)\s*=",
        r",\n  \1 =",
        text,
    )
    text = re.sub(r"\s+}\s*(?=\n\n@|\Z)", r"\n}", text)
    text = text.strip() + "\n"
    return text


def make_artifact_inventory() -> pd.DataFrame:
    rows = [
        {
            "artifact_id": "P10-A1",
            "artifact": "main.tex",
            "path": str(OUT_MAIN_TEX.relative_to(ROOT)),
            "purpose": "Main LaTeX manuscript scaffold.",
            "status": "generated",
        },
        {
            "artifact_id": "P10-A2",
            "artifact": "references.bib",
            "path": str(OUT_BIB.relative_to(ROOT)),
            "purpose": "Normalized BibTeX file copied from P9 references.",
            "status": "generated",
        },
        {
            "artifact_id": "P10-A3",
            "artifact": "Table 1",
            "path": str(OUT_TABLE1.relative_to(ROOT)),
            "purpose": "System-level baseline comparison.",
            "status": "generated",
        },
        {
            "artifact_id": "P10-A4",
            "artifact": "Table 2",
            "path": str(OUT_TABLE2.relative_to(ROOT)),
            "purpose": "QCR ablation table.",
            "status": "generated",
        },
        {
            "artifact_id": "P10-A5",
            "artifact": "Table 3",
            "path": str(OUT_TABLE3.relative_to(ROOT)),
            "purpose": "Boundary and robustness summary.",
            "status": "generated",
        },
        {
            "artifact_id": "P10-A6",
            "artifact": "Table 4",
            "path": str(OUT_TABLE4.relative_to(ROOT)),
            "purpose": "EfficientAD-100 sensitivity table.",
            "status": "generated",
        },
        {
            "artifact_id": "P10-A7",
            "artifact": "Figure 1 SVG copy",
            "path": str(OUT_FIG1_COPY.relative_to(ROOT)),
            "purpose": "Framework figure source copied from P8; main.tex uses placeholder for compile safety.",
            "status": "generated" if OUT_FIG1_COPY.exists() else "missing_source",
        },
    ]
    return pd.DataFrame(rows)


def make_checklist() -> pd.DataFrame:
    rows = [
        {
            "check_id": "P10-C1",
            "check": "LaTeX scaffold generated",
            "status": "done",
            "next_action": "Open paper/quality_calibrated_qcr/main.tex and inspect section order.",
        },
        {
            "check_id": "P10-C2",
            "check": "Booktabs tables generated",
            "status": "done",
            "next_action": "Manually compress table text if venue page budget is tight.",
        },
        {
            "check_id": "P10-C3",
            "check": "References normalized",
            "status": "done",
            "next_action": "Later replace with official BibTeX if venue requires exact pages/proceedings fields.",
        },
        {
            "check_id": "P10-C4",
            "check": "Figure 1 placeholder in main.tex",
            "status": "done",
            "next_action": "Convert SVG to PDF/PNG and replace placeholder before final submission.",
        },
        {
            "check_id": "P10-C5",
            "check": "Figure 2 still missing",
            "status": "open",
            "next_action": "Manually inspect P8 selected cases before creating boundary montage.",
        },
        {
            "check_id": "P10-C6",
            "check": "AnomalyCLIP not experimentally included",
            "status": "open",
            "next_action": "Keep as limitation unless later run under matched protocol.",
        },
        {
            "check_id": "P10-C7",
            "check": "Compile test",
            "status": "not_run",
            "next_action": "Run pdflatex/bibtex locally if TeX is installed.",
        },
    ]
    return pd.DataFrame(rows)


def write_main_tex(system: pd.DataFrame, qcr: pd.DataFrame, deltas: pd.DataFrame, e17: pd.DataFrame) -> None:
    full_vlm = get_system_score(system, "full-image VLM")
    context_vlm = get_system_score(system, "context-aware VLM")
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

    template = r"""\documentclass[10pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{xcolor}

\title{Quality-Calibrated Localization-Guided VLM Reasoning for Industrial Anomaly Recognition}
\author{Anonymous Authors}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Industrial anomaly recognition with general-purpose vision-language models remains unreliable when defects are small, localized, and visually subtle.
We propose Quality-Calibrated QCR, a localization-guided VLM reasoning framework that converts detector localization evidence into candidate-level visual-language evidence and calibrates crop-level VLM anomaly scores using candidate quality.
The method treats candidate quality as the main reliability mechanism and uses detector-VLM consistency only as a conservative adaptive refinement.
Experiments with detector, VLM, and external anomaly baselines show that localization-guided VLM evidence complements detector scores.
Ablations further show that quality calibration provides the main gain over naive detector-crop fusion, while adaptive consistency yields only a small refinement and fixed consistency should remain diagnostic.
Failure and boundary analyses clarify when quality calibration helps, when it fails, and why broad claims about segmentation or manufacturing-cause reasoning are unsupported.
\end{abstract}

\section{Introduction}

Industrial anomalies often occupy small localized regions and may not dominate global image semantics~\cite{Bergmann2019MVTecAD,Zou2022VisA}.
This makes direct full-image VLM reasoning unreliable for industrial inspection.
A detector can provide localization evidence, while a VLM can provide visual-language abnormality evidence; however, naive fusion does not model whether the localized crop is reliable.
We address this gap by calibrating crop-level VLM scores with candidate quality derived from localization evidence.

Our method, \emph{Quality-Calibrated QCR}, uses detector localization to generate candidate crops, obtains crop-level VLM anomaly evidence, and modulates that evidence using candidate quality.
We also study detector-VLM consistency, but robustness analysis shows that fixed consistency is not reliable enough to be the final method.
Thus, consistency is retained only as a small adaptive refinement.

Our contributions are:
\begin{itemize}
    \item We formulate industrial anomaly recognition as localization-guided VLM evidence reasoning.
    \item We propose candidate-quality calibration as the main reliability mechanism for crop-level VLM evidence.
    \item We analyze fixed and adaptive detector-VLM consistency and show why consistency must be treated conservatively.
    \item We provide strong baseline comparisons, ablations, boundary analysis, and fixed-budget sensitivity checks with explicit claim restrictions.
\end{itemize}

\section{Related Work}

\paragraph{Industrial anomaly detection.}
PatchCore~\cite{Roth2022PatchCore}, FastFlow-style detectors~\cite{Yu2021FastFlow}, and EfficientAD~\cite{Batzner2024EfficientAD} provide strong anomaly detection and localization evidence.
Our method does not replace these detectors; it uses localization evidence to produce candidate-level VLM evidence.

\paragraph{Vision-language anomaly detection.}
CLIP~\cite{Radford2021CLIP}-based anomaly methods such as WinCLIP~\cite{Jeong2023WinCLIP} and AnomalyCLIP~\cite{Zhou2024AnomalyCLIP} show the value of language-supervised representations.
Our work is narrower: it studies how detector-guided crop evidence should be calibrated before being trusted by a VLM-based anomaly scorer.

\paragraph{Reliability calibration.}
Naive score fusion assumes detector and VLM scores are equally reliable.
We show that candidate quality is the stable calibration signal, whereas fixed consistency is not robust enough to be the final method.

\section{Method}

Given image $x$, a detector produces localization evidence $A$ and normalized anomaly score $D$.
Candidate crops $C=\{c_i\}$ are generated from $A$.
The VLM scores each crop and the crop-level scores are aggregated into $M$.
Candidate quality $Q$ measures the reliability of the localized crop evidence.

The naive detector-crop fusion baseline is:
\begin{equation}
S_{\mathrm{naive}} = 0.5D + 0.5M.
\end{equation}

Quality-Calibrated QCR modulates the VLM contribution by candidate quality:
\begin{equation}
S_{\mathrm{quality}} = 0.5D + 0.5M(0.5 + 0.5Q).
\end{equation}

A diagnostic fixed Q+C variant is:
\begin{equation}
S_{\mathrm{fixed}} = 0.4D + 0.4M + 0.1Q + 0.1K,
\end{equation}
where $K$ is detector-VLM high-high consistency.
This variant is diagnostic only because fixed consistency is not robust across protocols.

The adaptive refinement uses a conservative gate:
\begin{equation}
g = QK(1-|D-M|)\min(D,M),
\end{equation}
\begin{equation}
S_{\mathrm{adaptive}} = S_{\mathrm{quality}} + 0.05g.
\end{equation}

The main method is $S_{\mathrm{quality}}$.
The adaptive variant is reported as a refinement, not as the main performance source.

\begin{figure}[t]
\centering
\fbox{
\begin{minipage}{0.92\linewidth}
\centering
\vspace{0.8em}
\textbf{Figure 1 placeholder.}\\
Detector localization $\rightarrow$ candidate crops $\rightarrow$ crop VLM score $\rightarrow$ candidate quality calibration $\rightarrow$ adaptive refinement.\\
The SVG source is stored at \texttt{figures/figure1\_framework\_schematic.svg}.
\vspace{0.8em}
\end{minipage}
}
\caption{Overview of Quality-Calibrated QCR. Candidate quality is the main method core; adaptive consistency is a conservative refinement; fixed Q+C is diagnostic only.}
\label{fig:framework}
\end{figure}

\section{Experiments}

We evaluate two complementary views.
The system-level view compares VLM baselines, detector baselines, and localization-guided fusion, using VisA-style industrial anomaly evaluation context~\cite{Zou2022VisA}.
The QCR view isolates detector-only scoring, crop VLM scoring, naive fusion, quality calibration, fixed Q+C, and adaptive refinement under the QCR primary protocol.
Image AUROC is the main metric; pixel metrics are auxiliary only.

\section{Main Results}

Full-image VLM obtains mean AUROC __FULL_VLM__, while context-aware VLM obtains __CONTEXT_VLM__.
PatchCore~\cite{Roth2022PatchCore} obtains __PATCHCORE__ and EfficientAD-30 fixed-budget~\cite{Batzner2024EfficientAD} obtains __EFFICIENTAD30__.
The fair LOCO fusion reaches __LOCO__, improving over PatchCore by __D_LOCO_PATCH__ and over EfficientAD-30 by __D_LOCO_EAD__.
The same-set result __SAME_SET__ is reported only as an upper-bound diagnostic.

\input{tables/table1_system_baselines}

\section{Ablation Study}

Under the QCR primary protocol, naive detector-crop fusion obtains mean AUROC __QCR_NAIVE__.
Quality-Calibrated QCR obtains __QCR_QUALITY__, a gain of __D_QUALITY_NAIVE__.
The adaptive refinement obtains __QCR_ADAPTIVE__, improving over naive fusion by __D_ADAPTIVE_NAIVE__ but over the quality core by only __D_ADAPTIVE_QUALITY__.
Therefore, the main effective component is candidate quality calibration, not adaptive consistency.

\input{tables/table2_qcr_ablation}

\section{Failure and Boundary Analysis}

Boundary analysis shows that quality calibration is useful but not universal.
It can help by boosting true localized anomalies or suppressing normal false positives, but it can also fail when candidate quality is misleading or detector-VLM evidence disagrees.
Fixed Q+C can peak in some cases but is not robust enough to be final.

\input{tables/table3_boundary_summary}

\section{EfficientAD Budget Sensitivity}

A 100-epoch EfficientAD sensitivity check on fruit_jelly gives image-AUROC delta __E17_IMAGE_DELTA__ relative to EfficientAD-30.
This does not indicate severe image-level underestimation of EfficientAD-30, but EfficientAD remains a fixed-budget baseline and should not be described as full EfficientAD defeat.

\input{tables/table4_efficientad_sensitivity}

\section{Limitations}

The method is image-level anomaly recognition and candidate-level VLM evidence calibration.
It does not claim pixel-level segmentation SOTA, manufacturing-cause reasoning, or full anomaly understanding.
EfficientAD is fixed-budget, AnomalyCLIP~\cite{Zhou2024AnomalyCLIP} is not experimentally included, and adaptive consistency has only a small gain over quality-only.

\section{Conclusion}

Quality-Calibrated QCR provides a reliability-calibrated bridge between industrial anomaly localization and VLM anomaly evidence.
Candidate quality is the main effective component, while adaptive consistency is a conservative refinement.
The evidence supports a cautious but coherent claim: localization-guided VLM anomaly recognition becomes more reliable when crop-level evidence is calibrated by candidate quality.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""

    replacements = {
        "__FULL_VLM__": fmt(full_vlm),
        "__CONTEXT_VLM__": fmt(context_vlm),
        "__PATCHCORE__": fmt(patchcore),
        "__EFFICIENTAD30__": fmt(efficientad30),
        "__LOCO__": fmt(loco),
        "__D_LOCO_PATCH__": signed(d_loco_patch),
        "__D_LOCO_EAD__": signed(d_loco_ead),
        "__SAME_SET__": fmt(same_set),
        "__QCR_NAIVE__": fmt(qcr_mean.get("V3")),
        "__QCR_QUALITY__": fmt(qcr_mean.get("V4")),
        "__QCR_ADAPTIVE__": fmt(qcr_mean.get("V6")),
        "__D_QUALITY_NAIVE__": signed(d_quality_naive),
        "__D_ADAPTIVE_NAIVE__": signed(d_adaptive_naive),
        "__D_ADAPTIVE_QUALITY__": signed(d_adaptive_quality),
        "__E17_IMAGE_DELTA__": signed(e17_image["delta_100_minus_30"]),
    }

    for k, v in replacements.items():
        template = template.replace(k, v)

    OUT_MAIN_TEX.write_text(template, encoding="utf-8", newline="\n")


def write_report(artifacts: pd.DataFrame, checklist: pd.DataFrame) -> None:
    lines = []
    lines += [
        "# Paper Stage P10: LaTeX Manuscript Scaffold",
        "",
        "## 1. Outputs",
        "",
        f"- Main manuscript: `{OUT_MAIN_TEX.relative_to(ROOT)}`",
        f"- BibTeX: `{OUT_BIB.relative_to(ROOT)}`",
        f"- Tables directory: `{TABLE_DIR.relative_to(ROOT)}`",
        f"- Figure directory: `{FIG_DIR.relative_to(ROOT)}`",
        "",
        "## 2. Artifact Inventory",
        "",
        "| ID | Artifact | Path | Purpose | Status |",
        "|---|---|---|---|---|",
    ]

    for _, r in artifacts.iterrows():
        lines.append(
            f"| {r['artifact_id']} | {r['artifact']} | `{r['path']}` | {r['purpose']} | {r['status']} |"
        )

    lines += [
        "",
        "## 3. Compile and Submission Checklist",
        "",
        "| ID | Check | Status | Next Action |",
        "|---|---|---|---|",
    ]

    for _, r in checklist.iterrows():
        lines.append(
            f"| {r['check_id']} | {r['check']} | {r['status']} | {r['next_action']} |"
        )

    lines += [
        "",
        "## 4. Manual Compile Command",
        "",
        "Run this only if a LaTeX distribution is installed:",
        "",
        "```bash",
        "cd paper/quality_calibrated_qcr",
        "pdflatex main.tex",
        "bibtex main",
        "pdflatex main.tex",
        "pdflatex main.tex",
        "```",
        "",
        "## 5. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Paper Stage P11: Figure 2 manual image montage preparation",
        "```",
        "",
        "Do not polish submission text before Figure 2 cases are visually inspected.",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    system = read_csv_strict(IN_SYSTEM)
    qcr = read_csv_strict(IN_QCR)
    deltas = read_csv_strict(IN_DELTAS)
    boundary = read_csv_strict(IN_BOUNDARY)
    e17 = read_csv_strict(IN_E17)

    table1 = build_table1(system)
    table2 = build_table2(qcr)
    table3 = build_table3(boundary)
    table4 = build_table4(e17)

    write_booktabs_table(
        table1,
        OUT_TABLE1,
        "System-level strong baseline comparison.",
        "tab:system_baselines",
        "LOCO is the fair system-level result. Same-set is an upper-bound diagnostic only.",
        align="llll",
    )
    write_booktabs_table(
        table2,
        OUT_TABLE2,
        "QCR primary-protocol ablation.",
        "tab:qcr_ablation",
        "Quality-Calibrated QCR is the main method core. Fixed Q+C is diagnostic only.",
        align="lllllll"[: len(table2.columns)],
    )
    write_booktabs_table(
        table3,
        OUT_TABLE3,
        "Boundary and robustness summary.",
        "tab:boundary_summary",
        "Per-category deltas show quality calibration is useful but not universal; adaptive consistency is only a refinement.",
        align="llllll",
    )
    write_booktabs_table(
        table4,
        OUT_TABLE4,
        "EfficientAD-100 fruit_jelly sensitivity.",
        "tab:efficientad_sensitivity",
        "This is a defensive fixed-budget sensitivity check, not a full EfficientAD sweep.",
        align="llll",
    )

    OUT_BIB.write_text(normalize_bib(), encoding="utf-8", newline="\n")

    if IN_FIG1_SVG.exists():
        shutil.copy2(IN_FIG1_SVG, OUT_FIG1_COPY)

    write_main_tex(system, qcr, deltas, e17)

    artifacts = make_artifact_inventory()
    checklist = make_checklist()

    artifacts.to_csv(OUT_ARTIFACTS, index=False, lineterminator="\n")
    checklist.to_csv(OUT_CHECKLIST, index=False, lineterminator="\n")

    write_report(artifacts, checklist)

    print("[DONE]", OUT_MAIN_TEX)
    print("[DONE]", OUT_BIB)
    print("[DONE]", OUT_TABLE1)
    print("[DONE]", OUT_TABLE2)
    print("[DONE]", OUT_TABLE3)
    print("[DONE]", OUT_TABLE4)
    print("[DONE]", OUT_ARTIFACTS)
    print("[DONE]", OUT_CHECKLIST)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== artifacts =====")
    print(artifacts.to_string(index=False))
    print()
    print("===== checklist =====")
    print(checklist.to_string(index=False))


if __name__ == "__main__":
    main()
