# Paper Stage P9: References and Citation Placement

## 1. Outputs

- BibTeX file: `docs/paper_p9/references.bib`
- Citation-marked compact draft: `docs/paper_p9/paper_p9_citation_marked_compact_draft.md`
- Reference inventory: `results/paper_p9/paper_p9_reference_inventory_verified.csv`
- Citation placement map: `results/paper_p9/paper_p9_citation_placement_map.csv`
- Reference risk checklist: `results/paper_p9/paper_p9_reference_risk_checklist.csv`

## 2. Reference Inventory

| Cite Key | Work | Category | Paper Sections | Risk Note |
|---|---|---|---|---|
| `Bergmann2019MVTecAD` | MVTec AD | industrial_anomaly_dataset | Introduction; Related Work; Experimental Setup | Use for industrial anomaly benchmark background; do not imply current paper evaluates on all MVTec categories unless true. |
| `Zou2022VisA` | VisA / SPot-the-Difference | industrial_anomaly_dataset | Introduction; Related Work; Experimental Setup | Use when discussing VisA protocol/categories. |
| `Roth2022PatchCore` | PatchCore | industrial_anomaly_detector | Related Work; Baselines; Main Results | Treat PatchCore as strong baseline; do not frame as weak or replaced. |
| `Yu2021FastFlow` | FastFlow | industrial_anomaly_detector | Related Work; Method context | Use as detector-family context; do not claim VLM replaces flow detectors. |
| `Batzner2024EfficientAD` | EfficientAD | industrial_anomaly_detector | Related Work; Baselines; EfficientAD budget sensitivity | Our result is EfficientAD-30 fixed-budget, not full EfficientAD defeat. |
| `Radford2021CLIP` | CLIP | vision_language_model | Introduction; Related Work | Use for general VLM/CLIP background only. |
| `Jeong2023WinCLIP` | WinCLIP | clip_anomaly_detection | Related Work; Baselines; Main Results | Our comparison is fixed protocol only; do not claim comprehensive WinCLIP defeat. |
| `Zhou2024AnomalyCLIP` | AnomalyCLIP | clip_anomaly_detection | Related Work; Limitations | Not experimentally included; cite as related work and limitation, not defeated baseline. |

## 3. Citation Placement Map

| ID | Section | Target Text | Citation | Reason |
|---|---|---|---|---|
| C1 | Introduction | industrial anomaly recognition / industrial inspection benchmark motivation | `\cite{Bergmann2019MVTecAD,Zou2022VisA}` | Ground the industrial anomaly benchmark context. |
| C2 | Introduction | vision-language models / CLIP background | `\cite{Radford2021CLIP}` | Ground VLM/CLIP motivation. |
| C3 | Related Work: Industrial anomaly detection | PatchCore, FastFlow, EfficientAD detector line | `\cite{Roth2022PatchCore,Yu2021FastFlow,Batzner2024EfficientAD}` | Ground detector baselines and localization evidence line. |
| C4 | Related Work: VLM anomaly detection | CLIP, WinCLIP, AnomalyCLIP | `\cite{Radford2021CLIP,Jeong2023WinCLIP,Zhou2024AnomalyCLIP}` | Ground CLIP/VLM anomaly detection line. |
| C5 | Experimental Setup | VisA-based experimental protocol | `\cite{Zou2022VisA}` | Dataset citation. |
| C6 | Baselines | PatchCore, EfficientAD, WinCLIP baselines | `\cite{Roth2022PatchCore,Batzner2024EfficientAD,Jeong2023WinCLIP}` | Baseline citation. |
| C7 | Limitations | AnomalyCLIP missing external VLM anomaly baseline | `\cite{Zhou2024AnomalyCLIP}` | Cite the explicitly missing baseline. |

## 4. Reference Risk Checklist

| ID | Risk | Severity | Handling |
|---|---|---|---|
| P9-R1 | BibTeX page numbers are omitted. | low | Acceptable for internal draft. Add pages automatically later via venue template or official BibTeX if required. |
| P9-R2 | AnomalyCLIP is cited but not experimentally compared. | medium_high | Keep it in related work and limitations only. Do not include it in result tables unless later run. |
| P9-R3 | EfficientAD can be misread as fully optimized. | medium_high | Always write EfficientAD-30 fixed-budget in result text. |
| P9-R4 | WinCLIP comparison can be overgeneralized. | medium | Always write WinCLIP fixed protocol, not broad WinCLIP defeat. |
| P9-R5 | Dataset citations can imply broader dataset coverage than experiments actually use. | medium | Experimental Setup must state exact categories/protocols used. |
| P9-R6 | Related Work may sound like method replaces detectors. | medium | Use complement language: detector localization evidence is used as input evidence. |

## 5. Citation Rules for the Paper

- Cite MVTec AD / VisA for industrial anomaly dataset context.
- Cite PatchCore / FastFlow / EfficientAD for detector and localization baselines.
- Cite CLIP / WinCLIP / AnomalyCLIP for VLM anomaly context.
- Cite AnomalyCLIP only as related work or limitation unless it is later run.
- Do not cite EfficientAD in a way that implies full-budget comparison.
- Do not cite WinCLIP in a way that implies broad CLIP-family SOTA.

## 6. Next Step

Next stage:

```text
Paper Stage P10: convert compact Markdown tables to LaTeX booktabs and prepare manuscript .tex scaffold
```
