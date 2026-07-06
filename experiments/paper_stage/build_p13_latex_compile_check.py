from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import pandas as pd


ROOT = Path(".").resolve()

PAPER_DIR = ROOT / "paper/quality_calibrated_qcr"
MAIN_TEX = PAPER_DIR / "main.tex"
BIB = PAPER_DIR / "references.bib"

TABLES = [
    PAPER_DIR / "tables/table1_system_baselines.tex",
    PAPER_DIR / "tables/table2_qcr_ablation.tex",
    PAPER_DIR / "tables/table3_boundary_summary.tex",
    PAPER_DIR / "tables/table4_efficientad_sensitivity.tex",
]

FIGURES = [
    PAPER_DIR / "figures/figure1_framework_schematic.svg",
    PAPER_DIR / "figures/figure2_boundary_cases_montage.png",
]

OUT_DIR = ROOT / "results/paper_p13"
DOC_DIR = ROOT / "docs/paper_p13"

OUT_SUMMARY = OUT_DIR / "paper_p13_compile_check_summary.csv"
OUT_FINDINGS = OUT_DIR / "paper_p13_latex_log_findings.csv"
OUT_REPORT = DOC_DIR / "paper_p13_compile_check_report.md"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    return proc.returncode, proc.stdout


def check_required_files() -> list[dict]:
    rows = []

    required = [MAIN_TEX, BIB] + TABLES + FIGURES

    for p in required:
        rows.append(
            {
                "check_group": "required_file",
                "item": str(p.relative_to(ROOT)),
                "status": "ok" if p.exists() else "missing",
                "detail": "" if p.exists() else "Required file missing.",
            }
        )

    return rows


def check_tex_content(tex: str) -> list[dict]:
    checks = [
        (
            "method_name",
            "Quality-Calibrated QCR",
            "ok" if "Quality-Calibrated QCR" in tex else "missing",
            "Main method name should be present.",
        ),
        (
            "figure1_label",
            "fig:framework",
            "ok" if r"\label{fig:framework}" in tex else "missing",
            "Figure 1 framework label should be present.",
        ),
        (
            "figure2_label",
            "fig:boundary_cases",
            "ok" if r"\label{fig:boundary_cases}" in tex else "missing",
            "Figure 2 boundary-case label should be present.",
        ),
        (
            "table1_input",
            "table1_system_baselines",
            "ok" if "table1_system_baselines" in tex else "missing",
            "System baseline table should be included.",
        ),
        (
            "table2_input",
            "table2_qcr_ablation",
            "ok" if "table2_qcr_ablation" in tex else "missing",
            "QCR ablation table should be included.",
        ),
        (
            "table3_input",
            "table3_boundary_summary",
            "ok" if "table3_boundary_summary" in tex else "missing",
            "Boundary summary table should be included.",
        ),
        (
            "table4_input",
            "table4_efficientad_sensitivity",
            "ok" if "table4_efficientad_sensitivity" in tex else "missing",
            "EfficientAD sensitivity table should be included.",
        ),
        (
            "no_unresolved_placeholders",
            "__PLACEHOLDER__",
            "ok" if "__" not in tex else "needs_patch",
            "No unresolved __PLACEHOLDER__ tokens should remain.",
        ),
        (
            "anomalyclip_limitation",
            "AnomalyCLIP limitation",
            "ok" if "AnomalyCLIP" in tex and "not experimentally included" in tex else "needs_patch",
            "AnomalyCLIP should remain limitation/related work, not result claim.",
        ),
        (
            "efficientad_fixed_budget",
            "EfficientAD-30 fixed-budget",
            "ok" if "EfficientAD-30 fixed-budget" in tex else "needs_patch",
            "EfficientAD should be marked fixed-budget.",
        ),
        (
            "fixed_qc_diagnostic",
            "fixed Q+C diagnostic",
            "ok" if "diagnostic" in tex and "fixed" in tex.lower() else "needs_patch",
            "Fixed Q+C should remain diagnostic.",
        ),
    ]

    rows = []
    for item, target, status, detail in checks:
        rows.append(
            {
                "check_group": "tex_content",
                "item": item,
                "status": status,
                "detail": detail,
            }
        )

    forbidden = [
        "state-of-the-art segmentation",
        "SOTA segmentation",
        "manufacturing cause",
        "full anomaly understanding",
        "universally beneficial",
        "defeat EfficientAD",
    ]

    lower = tex.lower()
    for phrase in forbidden:
        rows.append(
            {
                "check_group": "forbidden_claim_scan",
                "item": phrase,
                "status": "flag" if phrase.lower() in lower else "ok",
                "detail": "Flagged phrase should be removed or softened." if phrase.lower() in lower else "",
            }
        )

    return rows


def parse_latex_log(log_text: str) -> list[dict]:
    rows = []

    patterns = [
        ("latex_error", r"^! .+", "error"),
        ("undefined_reference", r"Reference `[^']+' on page .+ undefined", "warning"),
        ("undefined_citation", r"Citation `[^']+' on page .+ undefined", "warning"),
        ("rerun", r"Rerun to get cross-references right", "warning"),
        ("overfull_hbox", r"Overfull \\hbox", "warning"),
        ("underfull_hbox", r"Underfull \\hbox", "info"),
        ("missing_file", r"LaTeX Error: File `[^']+' not found", "error"),
    ]

    for line in log_text.splitlines():
        for finding_type, pat, severity in patterns:
            if re.search(pat, line):
                rows.append(
                    {
                        "finding_type": finding_type,
                        "severity": severity,
                        "line": line.strip(),
                    }
                )

    return rows


def compile_latex() -> tuple[str, list[dict]]:
    if not MAIN_TEX.exists():
        return "main_missing", [
            {
                "finding_type": "main_missing",
                "severity": "error",
                "line": "paper/quality_calibrated_qcr/main.tex is missing.",
            }
        ]

    if shutil.which("pdflatex") is None:
        return "tex_not_installed", [
            {
                "finding_type": "tex_not_installed",
                "severity": "warning",
                "line": "pdflatex not found. Compile check skipped.",
            }
        ]

    logs = []

    rc1, out1 = run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], PAPER_DIR)
    logs.append(out1)

    if shutil.which("bibtex") is not None:
        rc_bib, out_bib = run_cmd(["bibtex", "main"], PAPER_DIR)
        logs.append(out_bib)
    else:
        logs.append("bibtex not found. Bibliography pass skipped.")

    rc2, out2 = run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], PAPER_DIR)
    logs.append(out2)

    rc3, out3 = run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], PAPER_DIR)
    logs.append(out3)

    full_log = "\n".join(logs)
    findings = parse_latex_log(full_log)

    pdf = PAPER_DIR / "main.pdf"
    if pdf.exists() and rc3 == 0:
        compile_status = "compiled"
    elif pdf.exists():
        compile_status = "pdf_created_with_warnings_or_errors"
    else:
        compile_status = "failed"

    return compile_status, findings


def write_report(summary: pd.DataFrame, findings: pd.DataFrame, compile_status: str) -> None:
    error_count = int((findings["severity"] == "error").sum()) if not findings.empty else 0
    warning_count = int((findings["severity"] == "warning").sum()) if not findings.empty else 0

    bad_checks = summary[summary["status"].isin(["missing", "needs_patch", "flag"])]

    lines = [
        "# Paper Stage P13: LaTeX Compile Check and Manuscript Patch Report",
        "",
        "## 1. Compile Status",
        "",
        f"- compile_status: `{compile_status}`",
        f"- latex_errors: `{error_count}`",
        f"- warnings: `{warning_count}`",
        "",
        "## 2. Required File and Content Checks",
        "",
        "| Group | Item | Status | Detail |",
        "|---|---|---|---|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['check_group']} | {r['item']} | {r['status']} | {r['detail']} |"
        )

    lines += [
        "",
        "## 3. Findings",
        "",
        "| Type | Severity | Line |",
        "|---|---|---|",
    ]

    if findings.empty:
        lines.append("| none | ok | No LaTeX log findings recorded. |")
    else:
        for _, r in findings.head(80).iterrows():
            safe_line = str(r["line"]).replace("|", "\\|")
            lines.append(f"| {r['finding_type']} | {r['severity']} | `{safe_line}` |")

    lines += [
        "",
        "## 4. Decision",
        "",
    ]

    if compile_status == "compiled" and error_count == 0 and bad_checks.empty:
        lines.append("P13 passed. The manuscript scaffold compiles and no blocking content checks failed.")
        next_stage = "Paper Stage P14: language polish and venue-style compression"
    elif compile_status == "tex_not_installed":
        lines.append("P13 could not run LaTeX compilation because `pdflatex` is not installed. File/content checks still ran.")
        next_stage = "Install TeX or run compile check elsewhere; then proceed to P14 only after compile is clean."
    else:
        lines.append("P13 found issues that should be patched before polishing.")
        next_stage = "Patch LaTeX/table/figure issues and rerun P13."

    lines += [
        "",
        "## 5. Next Step",
        "",
        "```text",
        f"{next_stage}",
        "```",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    tex = read_text(MAIN_TEX)

    summary_rows = []
    summary_rows += check_required_files()
    summary_rows += check_tex_content(tex)

    compile_status, findings_rows = compile_latex()

    summary = pd.DataFrame(summary_rows)
    findings = pd.DataFrame(findings_rows)

    if findings.empty:
        findings = pd.DataFrame(columns=["finding_type", "severity", "line"])

    summary.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")
    findings.to_csv(OUT_FINDINGS, index=False, lineterminator="\n")

    write_report(summary, findings, compile_status)

    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_FINDINGS)
    print("[DONE]", OUT_REPORT)
    print()
    print("compile_status:", compile_status)
    print()
    print("===== failed checks =====")
    bad = summary[summary["status"].isin(["missing", "needs_patch", "flag"])]
    if bad.empty:
        print("none")
    else:
        print(bad.to_string(index=False))
    print()
    print("===== findings =====")
    if findings.empty:
        print("none")
    else:
        print(findings.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
