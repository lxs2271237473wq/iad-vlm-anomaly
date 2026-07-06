# Paper Stage P10: LaTeX Manuscript Scaffold

## 1. Outputs

- Main manuscript: `paper/quality_calibrated_qcr/main.tex`
- BibTeX: `paper/quality_calibrated_qcr/references.bib`
- Tables directory: `paper/quality_calibrated_qcr/tables`
- Figure directory: `paper/quality_calibrated_qcr/figures`

## 2. Artifact Inventory

| ID | Artifact | Path | Purpose | Status |
|---|---|---|---|---|
| P10-A1 | main.tex | `paper/quality_calibrated_qcr/main.tex` | Main LaTeX manuscript scaffold. | generated |
| P10-A2 | references.bib | `paper/quality_calibrated_qcr/references.bib` | Normalized BibTeX file copied from P9 references. | generated |
| P10-A3 | Table 1 | `paper/quality_calibrated_qcr/tables/table1_system_baselines.tex` | System-level baseline comparison. | generated |
| P10-A4 | Table 2 | `paper/quality_calibrated_qcr/tables/table2_qcr_ablation.tex` | QCR ablation table. | generated |
| P10-A5 | Table 3 | `paper/quality_calibrated_qcr/tables/table3_boundary_summary.tex` | Boundary and robustness summary. | generated |
| P10-A6 | Table 4 | `paper/quality_calibrated_qcr/tables/table4_efficientad_sensitivity.tex` | EfficientAD-100 sensitivity table. | generated |
| P10-A7 | Figure 1 SVG copy | `paper/quality_calibrated_qcr/figures/figure1_framework_schematic.svg` | Framework figure source copied from P8; main.tex uses placeholder for compile safety. | generated |

## 3. Compile and Submission Checklist

| ID | Check | Status | Next Action |
|---|---|---|---|
| P10-C1 | LaTeX scaffold generated | done | Open paper/quality_calibrated_qcr/main.tex and inspect section order. |
| P10-C2 | Booktabs tables generated | done | Manually compress table text if venue page budget is tight. |
| P10-C3 | References normalized | done | Later replace with official BibTeX if venue requires exact pages/proceedings fields. |
| P10-C4 | Figure 1 placeholder in main.tex | done | Convert SVG to PDF/PNG and replace placeholder before final submission. |
| P10-C5 | Figure 2 still missing | open | Manually inspect P8 selected cases before creating boundary montage. |
| P10-C6 | AnomalyCLIP not experimentally included | open | Keep as limitation unless later run under matched protocol. |
| P10-C7 | Compile test | not_run | Run pdflatex/bibtex locally if TeX is installed. |

## 4. Manual Compile Command

Run this only if a LaTeX distribution is installed:

```bash
cd paper/quality_calibrated_qcr
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 5. Next Step

Next stage:

```text
Paper Stage P11: Figure 2 manual image montage preparation
```

Do not polish submission text before Figure 2 cases are visually inspected.
