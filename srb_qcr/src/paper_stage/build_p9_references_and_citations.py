from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_P7_DRAFT = ROOT / "docs/paper_p7/paper_p7_latex_style_compact_draft.md"
IN_P4_REF_INV = ROOT / "results/paper_p4/paper_p4_reference_inventory.csv"

OUT_DIR = ROOT / "results/paper_p9"
DOC_DIR = ROOT / "docs/paper_p9"

OUT_BIB = DOC_DIR / "references.bib"
OUT_REPORT = DOC_DIR / "paper_p9_reference_and_citation_plan.md"
OUT_MARKED_DRAFT = DOC_DIR / "paper_p9_citation_marked_compact_draft.md"

OUT_REF_INV = OUT_DIR / "paper_p9_reference_inventory_verified.csv"
OUT_CITE_MAP = OUT_DIR / "paper_p9_citation_placement_map.csv"
OUT_RISK = OUT_DIR / "paper_p9_reference_risk_checklist.csv"


def read_text_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def build_bib_entries() -> dict[str, str]:
    return {
        "Bergmann2019MVTecAD": r"""@inproceedings{Bergmann2019MVTecAD,
  title     = {{MVTec AD} -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author    = {Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2019}
}""",
        "Zou2022VisA": r"""@inproceedings{Zou2022VisA,
  title     = {{SPot-the-Difference} Self-Supervised Pre-training for Anomaly Detection and Segmentation},
  author    = {Zou, Yang and Jeong, Jongheon and Pemula, Latha and Zhang, Dongqing and Dabeer, Onkar},
  booktitle = {European Conference on Computer Vision},
  year      = {2022}
}""",
        "Roth2022PatchCore": r"""@inproceedings{Roth2022PatchCore,
  title     = {Towards Total Recall in Industrial Anomaly Detection},
  author    = {Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch{\"o}lkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2022}
}""",
        "Yu2021FastFlow": r"""@article{Yu2021FastFlow,
  title   = {{FastFlow}: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows},
  author  = {Yu, Jiawei and Zheng, Ye and Wang, Xiang and Li, Wei and Wu, Yushuang and Zhao, Rui and Wu, Liwei},
  journal = {arXiv preprint arXiv:2111.07677},
  year    = {2021}
}""",
        "Batzner2024EfficientAD": r"""@inproceedings{Batzner2024EfficientAD,
  title     = {{EfficientAD}: Accurate Visual Anomaly Detection at Millisecond-Level Latencies},
  author    = {Batzner, Kilian and Heckler, Lars and K{\"o}nig, Rebecca},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2024}
}""",
        "Radford2021CLIP": r"""@inproceedings{Radford2021CLIP,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle = {International Conference on Machine Learning},
  year      = {2021}
}""",
        "Jeong2023WinCLIP": r"""@inproceedings{Jeong2023WinCLIP,
  title     = {{WinCLIP}: Zero-/Few-Shot Anomaly Classification and Segmentation},
  author    = {Jeong, Jongheon and Zou, Yang and Kim, Taewan and Zhang, Dongqing and Ravichandran, Avinash and Dabeer, Onkar},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2023}
}""",
        "Zhou2024AnomalyCLIP": r"""@inproceedings{Zhou2024AnomalyCLIP,
  title     = {{AnomalyCLIP}: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection},
  author    = {Zhou, Qihang and Pang, Guansong and Tian, Yu and He, Shibo and Chen, Jiming},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}""",
    }


def build_reference_inventory() -> pd.DataFrame:
    rows = [
        {
            "cite_key": "Bergmann2019MVTecAD",
            "work": "MVTec AD",
            "category": "industrial_anomaly_dataset",
            "citation_role": "dataset/background",
            "paper_sections": "Introduction; Related Work; Experimental Setup",
            "source_type": "CVF / CVPR",
            "must_cite": True,
            "risk_note": "Use for industrial anomaly benchmark background; do not imply current paper evaluates on all MVTec categories unless true.",
        },
        {
            "cite_key": "Zou2022VisA",
            "work": "VisA / SPot-the-Difference",
            "category": "industrial_anomaly_dataset",
            "citation_role": "dataset/background/current benchmark context",
            "paper_sections": "Introduction; Related Work; Experimental Setup",
            "source_type": "ECCV / arXiv / AWS dataset page",
            "must_cite": True,
            "risk_note": "Use when discussing VisA protocol/categories.",
        },
        {
            "cite_key": "Roth2022PatchCore",
            "work": "PatchCore",
            "category": "industrial_anomaly_detector",
            "citation_role": "strong detector baseline",
            "paper_sections": "Related Work; Baselines; Main Results",
            "source_type": "CVF / CVPR",
            "must_cite": True,
            "risk_note": "Treat PatchCore as strong baseline; do not frame as weak or replaced.",
        },
        {
            "cite_key": "Yu2021FastFlow",
            "work": "FastFlow",
            "category": "industrial_anomaly_detector",
            "citation_role": "flow-based detector/localization backbone context",
            "paper_sections": "Related Work; Method context",
            "source_type": "arXiv",
            "must_cite": True,
            "risk_note": "Use as detector-family context; do not claim VLM replaces flow detectors.",
        },
        {
            "cite_key": "Batzner2024EfficientAD",
            "work": "EfficientAD",
            "category": "industrial_anomaly_detector",
            "citation_role": "modern efficient detector baseline",
            "paper_sections": "Related Work; Baselines; EfficientAD budget sensitivity",
            "source_type": "CVF / WACV",
            "must_cite": True,
            "risk_note": "Our result is EfficientAD-30 fixed-budget, not full EfficientAD defeat.",
        },
        {
            "cite_key": "Radford2021CLIP",
            "work": "CLIP",
            "category": "vision_language_model",
            "citation_role": "foundation VLM background",
            "paper_sections": "Introduction; Related Work",
            "source_type": "PMLR / ICML",
            "must_cite": True,
            "risk_note": "Use for general VLM/CLIP background only.",
        },
        {
            "cite_key": "Jeong2023WinCLIP",
            "work": "WinCLIP",
            "category": "clip_anomaly_detection",
            "citation_role": "external CLIP anomaly baseline",
            "paper_sections": "Related Work; Baselines; Main Results",
            "source_type": "CVF / CVPR",
            "must_cite": True,
            "risk_note": "Our comparison is fixed protocol only; do not claim comprehensive WinCLIP defeat.",
        },
        {
            "cite_key": "Zhou2024AnomalyCLIP",
            "work": "AnomalyCLIP",
            "category": "clip_anomaly_detection",
            "citation_role": "important missing external VLM anomaly baseline / limitation",
            "paper_sections": "Related Work; Limitations",
            "source_type": "OpenReview / ICLR",
            "must_cite": True,
            "risk_note": "Not experimentally included; cite as related work and limitation, not defeated baseline.",
        },
    ]
    return pd.DataFrame(rows)


def build_citation_map() -> pd.DataFrame:
    rows = [
        {
            "placement_id": "C1",
            "paper_section": "Introduction",
            "target_text": "industrial anomaly recognition / industrial inspection benchmark motivation",
            "citation": r"\cite{Bergmann2019MVTecAD,Zou2022VisA}",
            "reason": "Ground the industrial anomaly benchmark context.",
        },
        {
            "placement_id": "C2",
            "paper_section": "Introduction",
            "target_text": "vision-language models / CLIP background",
            "citation": r"\cite{Radford2021CLIP}",
            "reason": "Ground VLM/CLIP motivation.",
        },
        {
            "placement_id": "C3",
            "paper_section": "Related Work: Industrial anomaly detection",
            "target_text": "PatchCore, FastFlow, EfficientAD detector line",
            "citation": r"\cite{Roth2022PatchCore,Yu2021FastFlow,Batzner2024EfficientAD}",
            "reason": "Ground detector baselines and localization evidence line.",
        },
        {
            "placement_id": "C4",
            "paper_section": "Related Work: VLM anomaly detection",
            "target_text": "CLIP, WinCLIP, AnomalyCLIP",
            "citation": r"\cite{Radford2021CLIP,Jeong2023WinCLIP,Zhou2024AnomalyCLIP}",
            "reason": "Ground CLIP/VLM anomaly detection line.",
        },
        {
            "placement_id": "C5",
            "paper_section": "Experimental Setup",
            "target_text": "VisA-based experimental protocol",
            "citation": r"\cite{Zou2022VisA}",
            "reason": "Dataset citation.",
        },
        {
            "placement_id": "C6",
            "paper_section": "Baselines",
            "target_text": "PatchCore, EfficientAD, WinCLIP baselines",
            "citation": r"\cite{Roth2022PatchCore,Batzner2024EfficientAD,Jeong2023WinCLIP}",
            "reason": "Baseline citation.",
        },
        {
            "placement_id": "C7",
            "paper_section": "Limitations",
            "target_text": "AnomalyCLIP missing external VLM anomaly baseline",
            "citation": r"\cite{Zhou2024AnomalyCLIP}",
            "reason": "Cite the explicitly missing baseline.",
        },
    ]
    return pd.DataFrame(rows)


def build_risk_checklist() -> pd.DataFrame:
    rows = [
        {
            "risk_id": "P9-R1",
            "risk": "BibTeX page numbers are omitted.",
            "severity": "low",
            "handling": "Acceptable for internal draft. Add pages automatically later via venue template or official BibTeX if required.",
        },
        {
            "risk_id": "P9-R2",
            "risk": "AnomalyCLIP is cited but not experimentally compared.",
            "severity": "medium_high",
            "handling": "Keep it in related work and limitations only. Do not include it in result tables unless later run.",
        },
        {
            "risk_id": "P9-R3",
            "risk": "EfficientAD can be misread as fully optimized.",
            "severity": "medium_high",
            "handling": "Always write EfficientAD-30 fixed-budget in result text.",
        },
        {
            "risk_id": "P9-R4",
            "risk": "WinCLIP comparison can be overgeneralized.",
            "severity": "medium",
            "handling": "Always write WinCLIP fixed protocol, not broad WinCLIP defeat.",
        },
        {
            "risk_id": "P9-R5",
            "risk": "Dataset citations can imply broader dataset coverage than experiments actually use.",
            "severity": "medium",
            "handling": "Experimental Setup must state exact categories/protocols used.",
        },
        {
            "risk_id": "P9-R6",
            "risk": "Related Work may sound like method replaces detectors.",
            "severity": "medium",
            "handling": "Use complement language: detector localization evidence is used as input evidence.",
        },
    ]
    return pd.DataFrame(rows)


def write_bib(entries: dict[str, str]) -> None:
    text = "\n\n".join(entries[k] for k in entries.keys()) + "\n"
    OUT_BIB.write_text(text, encoding="utf-8", newline="\n")


def citation_mark_draft(draft: str) -> str:
    if not draft.strip():
        return "[MISSING] P7 compact draft not found."

    replacements = [
        (
            "Industrial anomalies often occupy small localized regions and may not dominate global image semantics.",
            r"Industrial anomalies often occupy small localized regions and may not dominate global image semantics~\cite{Bergmann2019MVTecAD,Zou2022VisA}."
        ),
        (
            "**Industrial anomaly detection.** PatchCore, FastFlow-style detectors, and EfficientAD provide strong anomaly detection and localization evidence.",
            r"**Industrial anomaly detection.** PatchCore~\cite{Roth2022PatchCore}, FastFlow-style detectors~\cite{Yu2021FastFlow}, and EfficientAD~\cite{Batzner2024EfficientAD} provide strong anomaly detection and localization evidence."
        ),
        (
            "**Vision-language anomaly detection.** CLIP-based anomaly methods such as WinCLIP and AnomalyCLIP show the value of language-supervised representations.",
            r"**Vision-language anomaly detection.** CLIP~\cite{Radford2021CLIP}-based anomaly methods such as WinCLIP~\cite{Jeong2023WinCLIP} and AnomalyCLIP~\cite{Zhou2024AnomalyCLIP} show the value of language-supervised representations."
        ),
        (
            "The system-level view compares VLM baselines, detector baselines, and localization-guided fusion.",
            r"The system-level view compares VLM baselines, detector baselines, and localization-guided fusion, using VisA-style industrial anomaly evaluation context~\cite{Zou2022VisA}."
        ),
        (
            "PatchCore obtains",
            r"PatchCore~\cite{Roth2022PatchCore} obtains"
        ),
        (
            "EfficientAD-30 fixed-budget obtains",
            r"EfficientAD-30 fixed-budget~\cite{Batzner2024EfficientAD} obtains"
        ),
        (
            "AnomalyCLIP is not included",
            r"AnomalyCLIP~\cite{Zhou2024AnomalyCLIP} is not included"
        ),
    ]

    out = draft
    for old, new in replacements:
        if old in out and new not in out:
            out = out.replace(old, new, 1)

    header = [
        "# Paper Stage P9: Citation-marked Compact Draft",
        "",
        "This file is generated from the P7 compact draft with first-pass citation commands inserted.",
        "It is still a Markdown/LaTeX hybrid draft; final venue formatting is not done.",
        "",
        "---",
        "",
    ]
    return "\n".join(header) + out


def write_report(ref_inv: pd.DataFrame, cite_map: pd.DataFrame, risk: pd.DataFrame) -> None:
    lines = []
    lines += [
        "# Paper Stage P9: References and Citation Placement",
        "",
        "## 1. Outputs",
        "",
        f"- BibTeX file: `{OUT_BIB.relative_to(ROOT)}`",
        f"- Citation-marked compact draft: `{OUT_MARKED_DRAFT.relative_to(ROOT)}`",
        f"- Reference inventory: `{OUT_REF_INV.relative_to(ROOT)}`",
        f"- Citation placement map: `{OUT_CITE_MAP.relative_to(ROOT)}`",
        f"- Reference risk checklist: `{OUT_RISK.relative_to(ROOT)}`",
        "",
        "## 2. Reference Inventory",
        "",
        "| Cite Key | Work | Category | Paper Sections | Risk Note |",
        "|---|---|---|---|---|",
    ]

    for _, r in ref_inv.iterrows():
        lines.append(
            f"| `{r['cite_key']}` | {r['work']} | {r['category']} | {r['paper_sections']} | {r['risk_note']} |"
        )

    lines += [
        "",
        "## 3. Citation Placement Map",
        "",
        "| ID | Section | Target Text | Citation | Reason |",
        "|---|---|---|---|---|",
    ]

    for _, r in cite_map.iterrows():
        lines.append(
            f"| {r['placement_id']} | {r['paper_section']} | {r['target_text']} | `{r['citation']}` | {r['reason']} |"
        )

    lines += [
        "",
        "## 4. Reference Risk Checklist",
        "",
        "| ID | Risk | Severity | Handling |",
        "|---|---|---|---|",
    ]

    for _, r in risk.iterrows():
        lines.append(
            f"| {r['risk_id']} | {r['risk']} | {r['severity']} | {r['handling']} |"
        )

    lines += [
        "",
        "## 5. Citation Rules for the Paper",
        "",
        "- Cite MVTec AD / VisA for industrial anomaly dataset context.",
        "- Cite PatchCore / FastFlow / EfficientAD for detector and localization baselines.",
        "- Cite CLIP / WinCLIP / AnomalyCLIP for VLM anomaly context.",
        "- Cite AnomalyCLIP only as related work or limitation unless it is later run.",
        "- Do not cite EfficientAD in a way that implies full-budget comparison.",
        "- Do not cite WinCLIP in a way that implies broad CLIP-family SOTA.",
        "",
        "## 6. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Paper Stage P10: convert compact Markdown tables to LaTeX booktabs and prepare manuscript .tex scaffold",
        "```",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    _ = read_csv_optional(IN_P4_REF_INV)

    entries = build_bib_entries()
    ref_inv = build_reference_inventory()
    cite_map = build_citation_map()
    risk = build_risk_checklist()

    write_bib(entries)

    ref_inv.to_csv(OUT_REF_INV, index=False, lineterminator="\n")
    cite_map.to_csv(OUT_CITE_MAP, index=False, lineterminator="\n")
    risk.to_csv(OUT_RISK, index=False, lineterminator="\n")

    draft = read_text_optional(IN_P7_DRAFT)
    marked = citation_mark_draft(draft)
    OUT_MARKED_DRAFT.write_text(marked, encoding="utf-8", newline="\n")

    write_report(ref_inv, cite_map, risk)

    print("[DONE]", OUT_BIB)
    print("[DONE]", OUT_MARKED_DRAFT)
    print("[DONE]", OUT_REF_INV)
    print("[DONE]", OUT_CITE_MAP)
    print("[DONE]", OUT_RISK)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== reference inventory =====")
    print(ref_inv.to_string(index=False))
    print()
    print("===== citation map =====")
    print(cite_map.to_string(index=False))


if __name__ == "__main__":
    main()
