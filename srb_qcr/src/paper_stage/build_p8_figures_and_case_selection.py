from __future__ import annotations

from pathlib import Path
from io import StringIO
import re
import html
import pandas as pd


ROOT = Path(".").resolve()

IN_CASES = ROOT / "results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv"
IN_BOUNDARY = ROOT / "results/stage16_qcru_ablation/stage16_e_category_boundary_summary.csv"
IN_TABLE_INV = ROOT / "results/paper_p7/paper_p7_compact_table_inventory.csv"

OUT_DIR = ROOT / "results/paper_p8"
DOC_DIR = ROOT / "docs/paper_p8"
FIG_DIR = DOC_DIR / "figures"

OUT_FIG1 = FIG_DIR / "figure1_framework_schematic.svg"
OUT_FIG2_PLAN = DOC_DIR / "figure2_boundary_case_selection_plan.md"

OUT_FIG1_NODES = OUT_DIR / "paper_p8_figure1_framework_nodes.csv"
OUT_FIG2_CASES = OUT_DIR / "paper_p8_figure2_selected_boundary_cases.csv"
OUT_CHECKLIST = OUT_DIR / "paper_p8_figure_readiness_checklist.csv"
OUT_REPORT = DOC_DIR / "paper_p8_figure_plan_report.md"


CASE_TYPES = [
    "quality_helps_anomaly_boost",
    "quality_helps_normal_suppression",
    "quality_boundary_anomaly_suppression",
    "quality_boundary_normal_boost",
    "fixed_consistency_boundary_anomaly_suppression",
    "fixed_consistency_boundary_normal_boost",
    "adaptive_refinement_high_gate",
    "detector_vlm_disagreement_boundary",
]


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    raw = path.read_text(encoding="utf-8").strip()

    # Local files should normally be fine. This fallback handles accidental one-line CSV artifacts.
    try:
        df = pd.read_csv(StringIO(raw))
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    if path == IN_CASES and raw.startswith("case_type,"):
        # Split before known case_type tokens if the file was flattened into one line.
        header_end = raw.find("quality_")
        if header_end == -1:
            header_end = raw.find("adaptive_refinement")
        if header_end == -1:
            header_end = raw.find("detector_vlm")
        if header_end > 0:
            header = raw[:header_end].strip().rstrip(",")
            body = raw[header_end:].strip()
            pattern = r"\s+(?=(?:" + "|".join(map(re.escape, CASE_TYPES)) + r"),)"
            rows = re.split(pattern, body)
            rows = [r.strip() for r in rows if r.strip()]
            repaired = header + "\n" + "\n".join(rows) + "\n"
            df = pd.read_csv(StringIO(repaired))
            if len(df.columns) > 1:
                return df

    raise RuntimeError(f"{path} could not be read as a normal CSV. Repair local CSV formatting first.")


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


def svg_text(x: int, y: int, text: str, size: int = 14, weight: str = "normal") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="middle">'
        f'{html.escape(text)}</text>'
    )


def svg_box(x: int, y: int, w: int, h: int, title: str, subtitle: str, fill: str = "#ffffff") -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" ry="12" '
            f'fill="{fill}" stroke="#222222" stroke-width="1.4"/>',
            svg_text(x + w // 2, y + 28, title, 15, "bold"),
            svg_text(x + w // 2, y + 52, subtitle, 12, "normal"),
        ]
    )


def svg_arrow(x1: int, y1: int, x2: int, y2: int, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#222222" stroke-width="1.6"{dash} marker-end="url(#arrow)"/>'
    )


def write_figure1_svg() -> pd.DataFrame:
    nodes = pd.DataFrame(
        [
            {
                "node_id": "N1",
                "label": "Input image",
                "description": "Industrial test image x",
                "paper_message": "The method starts from image-level industrial inspection input.",
            },
            {
                "node_id": "N2",
                "label": "Detector localization",
                "description": "Localization evidence A and detector score D",
                "paper_message": "Detector evidence is used, not replaced.",
            },
            {
                "node_id": "N3",
                "label": "Candidate crops",
                "description": "Candidate regions C = {c_i}",
                "paper_message": "Localization evidence is converted into crop-level evidence.",
            },
            {
                "node_id": "N4",
                "label": "Crop VLM scoring",
                "description": "Aggregated VLM anomaly score M",
                "paper_message": "VLM is applied to localized evidence rather than full image only.",
            },
            {
                "node_id": "N5",
                "label": "Candidate quality",
                "description": "Reliability score Q",
                "paper_message": "Quality is the main method core.",
            },
            {
                "node_id": "N6",
                "label": "Quality-Calibrated QCR",
                "description": "S_quality = 0.5D + 0.5M(0.5+0.5Q)",
                "paper_message": "Main final method core.",
            },
            {
                "node_id": "N7",
                "label": "Adaptive refinement",
                "description": "S_adaptive = S_quality + 0.05g",
                "paper_message": "Small conservative refinement only.",
            },
            {
                "node_id": "N8",
                "label": "Diagnostic fixed Q+C",
                "description": "Not final method",
                "paper_message": "Shown as diagnostic branch only.",
            },
        ]
    )

    svg = []
    svg += [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="560" viewBox="0 0 1320 560">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#222222"/>',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="1320" height="560" fill="#ffffff"/>',
        svg_text(660, 36, "Quality-Calibrated Localization-Guided VLM Reasoning", 22, "bold"),
        svg_text(660, 62, "Detector localization -> candidate crop -> VLM evidence -> candidate quality calibration -> adaptive refinement", 14),
    ]

    # Main pipeline boxes.
    boxes = [
        (40, 150, 150, 78, "Input image", "x", "#f7f7f7"),
        (230, 150, 180, 78, "Detector", "A, D", "#f7f7f7"),
        (455, 150, 180, 78, "Candidate crops", "C = {c_i}", "#f7f7f7"),
        (680, 150, 190, 78, "Crop VLM scoring", "M", "#f7f7f7"),
        (915, 150, 190, 78, "Quality calibration", "Q modulates M", "#f0f0f0"),
        (1135, 150, 160, 78, "Image score", "S_quality", "#f0f0f0"),
    ]

    for b in boxes:
        svg.append(svg_box(*b))

    arrows = [
        (190, 189, 230, 189, False),
        (410, 189, 455, 189, False),
        (635, 189, 680, 189, False),
        (870, 189, 915, 189, False),
        (1105, 189, 1135, 189, False),
    ]
    for a in arrows:
        svg.append(svg_arrow(*a))

    # Adaptive refinement branch.
    svg.append(svg_box(915, 315, 190, 86, "Adaptive consistency", "g = QK(1-|D-M|)min(D,M)", "#ffffff"))
    svg.append(svg_arrow(1010, 228, 1010, 315, False))
    svg.append(svg_arrow(1105, 358, 1215, 228, False))
    svg.append(svg_text(1050, 430, "Conservative refinement; not main performance source", 13))

    # Diagnostic fixed Q+C branch.
    svg.append(svg_box(680, 315, 190, 86, "Fixed Q+C", "diagnostic only", "#ffffff"))
    svg.append(svg_arrow(775, 228, 775, 315, True))
    svg.append(svg_text(775, 430, "Not final method; robustness insufficient", 13))

    # Evidence labels.
    svg.append(svg_text(330, 270, "Detector evidence is used as localization guidance", 13))
    svg.append(svg_text(760, 270, "VLM evidence is localized but must be calibrated", 13))
    svg.append(svg_text(1010, 270, "Candidate quality is the main reliable component", 13, "bold"))

    svg += ["</svg>"]

    OUT_FIG1.write_text("\n".join(svg), encoding="utf-8", newline="\n")
    return nodes


def select_boundary_cases(cases: pd.DataFrame) -> pd.DataFrame:
    df = cases.copy()

    for col in [
        "D", "M", "Q", "K",
        "adaptive_gate",
        "detector_vlm_disagreement",
        "delta_quality_minus_naive",
        "delta_fixed_minus_quality",
        "delta_adaptive_minus_quality",
        "is_anomaly_final",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    selection_rules = [
        {
            "figure_panel": "A",
            "case_type": "quality_helps_anomaly_boost",
            "selection_col": "delta_quality_minus_naive",
            "ascending": False,
            "paper_purpose": "Positive example: quality calibration boosts true anomaly evidence.",
        },
        {
            "figure_panel": "B",
            "case_type": "quality_helps_normal_suppression",
            "selection_col": "delta_quality_minus_naive",
            "ascending": True,
            "paper_purpose": "Positive example: quality calibration suppresses normal false-positive evidence.",
        },
        {
            "figure_panel": "C",
            "case_type": "quality_boundary_anomaly_suppression",
            "selection_col": "delta_quality_minus_naive",
            "ascending": True,
            "paper_purpose": "Boundary case: quality calibration can suppress true anomaly evidence.",
        },
        {
            "figure_panel": "D",
            "case_type": "quality_boundary_normal_boost",
            "selection_col": "delta_quality_minus_naive",
            "ascending": False,
            "paper_purpose": "Boundary case: quality calibration can boost normal evidence.",
        },
        {
            "figure_panel": "E",
            "case_type": "fixed_consistency_boundary_normal_boost",
            "selection_col": "delta_fixed_minus_quality",
            "ascending": False,
            "paper_purpose": "Explains why fixed Q+C can be risky.",
        },
        {
            "figure_panel": "F",
            "case_type": "detector_vlm_disagreement_boundary",
            "selection_col": "detector_vlm_disagreement",
            "ascending": False,
            "paper_purpose": "Shows detector-VLM conflict as a method boundary.",
        },
    ]

    selected_rows = []

    for rule in selection_rules:
        sub = df[df["case_type"] == rule["case_type"]].copy()
        if sub.empty:
            selected_rows.append(
                {
                    "figure_panel": rule["figure_panel"],
                    "case_type": rule["case_type"],
                    "status": "missing_case_type",
                    "paper_purpose": rule["paper_purpose"],
                }
            )
            continue

        col = rule["selection_col"]
        if col in sub.columns:
            sub = sub.sort_values(col, ascending=rule["ascending"])
        row = sub.iloc[0].to_dict()

        keep = {
            "figure_panel": rule["figure_panel"],
            "case_type": rule["case_type"],
            "status": "selected_for_manual_inspection",
            "paper_purpose": rule["paper_purpose"],
            "selection_col": col,
            "selection_order": "ascending" if rule["ascending"] else "descending",
        }

        for c in [
            "backbone", "dataset", "strategy", "eval_mode", "category",
            "image_key", "image_path", "is_anomaly_final", "gt_label", "defect_type",
            "D", "M", "Q", "K", "adaptive_gate",
            "score_naive", "score_quality", "score_fixed_qc", "score_adaptive",
            "delta_quality_minus_naive", "delta_fixed_minus_quality",
            "delta_adaptive_minus_quality", "detector_vlm_disagreement",
        ]:
            if c in row:
                keep[c] = row[c]

        selected_rows.append(keep)

    out = pd.DataFrame(selected_rows)
    return out


def write_figure2_plan(selected: pd.DataFrame) -> None:
    lines = []
    lines += [
        "# Figure 2 Boundary Case Selection Plan",
        "",
        "## Purpose",
        "",
        "Figure 2 should visually support the paper's boundary-aware claim:",
        "",
        "```text",
        "Quality calibration is the main reliability mechanism, but it is not universally correct. Fixed consistency can be risky, and detector-VLM disagreement remains a boundary case.",
        "```",
        "",
        "This file is a selection plan only. Do not use these panels in the paper until the original images/crops are manually inspected.",
        "",
        "## Selected Panels",
        "",
        "| Panel | Case Type | Category | Image Key | GT | Purpose | Manual Status |",
        "|---|---|---|---|---:|---|---|",
    ]

    for _, r in selected.iterrows():
        lines.append(
            f"| {r.get('figure_panel', '')} | {r.get('case_type', '')} | "
            f"{r.get('category', '')} | {r.get('image_key', '')} | "
            f"{r.get('is_anomaly_final', '')} | {r.get('paper_purpose', '')} | "
            f"{r.get('status', '')} |"
        )

    lines += [
        "",
        "## Required Manual Inspection",
        "",
        "For each selected panel, inspect:",
        "",
        "1. Original image.",
        "2. Detector heatmap or localization evidence, if available.",
        "3. Candidate crop used for VLM scoring.",
        "4. Whether the case visually matches the intended paper purpose.",
        "5. Whether the case could accidentally overclaim segmentation or cause reasoning.",
        "",
        "## Panel Interpretation",
        "",
        "- Panel A/B should show positive behavior of quality calibration.",
        "- Panel C/D should show quality calibration boundaries.",
        "- Panel E should explain why fixed Q+C is diagnostic only.",
        "- Panel F should explain detector-VLM disagreement.",
        "",
    ]

    OUT_FIG2_PLAN.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def make_checklist(selected: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "item_id": "P8-C1",
            "item": "Figure 1 framework SVG generated",
            "status": "done" if OUT_FIG1.exists() else "missing",
            "next_action": "Review visual layout and convert to PDF/LaTeX figure if needed.",
        },
        {
            "item_id": "P8-C2",
            "item": "Figure 2 candidate case selection generated",
            "status": "done" if not selected.empty else "missing",
            "next_action": "Manually inspect original images and crops before using in paper.",
        },
        {
            "item_id": "P8-C3",
            "item": "Figure 2 actual image montage",
            "status": "not_done",
            "next_action": "Create after manual inspection confirms selected cases.",
        },
        {
            "item_id": "P8-C4",
            "item": "Avoid overclaiming in figure captions",
            "status": "required",
            "next_action": "Caption must say boundary examples, not proof of universal behavior.",
        },
        {
            "item_id": "P8-C5",
            "item": "Check image paths",
            "status": "required",
            "next_action": "If image_path is absent, resolve image_key to original dataset file before montage.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(nodes: pd.DataFrame, selected: pd.DataFrame, checklist: pd.DataFrame) -> None:
    lines = []
    lines += [
        "# Paper Stage P8: Figure Plan",
        "",
        "## 1. Outputs",
        "",
        f"- Figure 1 schematic: `{OUT_FIG1.relative_to(ROOT)}`",
        f"- Figure 1 node inventory: `{OUT_FIG1_NODES.relative_to(ROOT)}`",
        f"- Figure 2 selected cases: `{OUT_FIG2_CASES.relative_to(ROOT)}`",
        f"- Figure 2 selection plan: `{OUT_FIG2_PLAN.relative_to(ROOT)}`",
        "",
        "## 2. Figure 1 Message",
        "",
        "Figure 1 should communicate the full method pipeline:",
        "",
        "```text",
        "image -> detector localization -> candidate crops -> crop VLM score -> candidate quality calibration -> image-level anomaly score",
        "```",
        "",
        "It must also show:",
        "",
        "- candidate quality is the main method core;",
        "- adaptive consistency is a conservative refinement;",
        "- fixed Q+C is diagnostic only.",
        "",
        "## 3. Figure 1 Nodes",
        "",
        "| Node | Label | Paper Message |",
        "|---|---|---|",
    ]

    for _, r in nodes.iterrows():
        lines.append(f"| {r['node_id']} | {r['label']} | {r['paper_message']} |")

    lines += [
        "",
        "## 4. Figure 2 Selected Boundary Cases",
        "",
        "| Panel | Case Type | Backbone | Category | Image Key | Purpose |",
        "|---|---|---|---|---|---|",
    ]

    for _, r in selected.iterrows():
        lines.append(
            f"| {r.get('figure_panel', '')} | {r.get('case_type', '')} | "
            f"{r.get('backbone', '')} | {r.get('category', '')} | "
            f"{r.get('image_key', '')} | {r.get('paper_purpose', '')} |"
        )

    lines += [
        "",
        "## 5. Checklist",
        "",
        "| ID | Item | Status | Next Action |",
        "|---|---|---|---|",
    ]

    for _, r in checklist.iterrows():
        lines.append(
            f"| {r['item_id']} | {r['item']} | {r['status']} | {r['next_action']} |"
        )

    lines += [
        "",
        "## 6. Caption Drafts",
        "",
        "### Figure 1 caption draft",
        "",
        "```text",
        "Overview of Quality-Calibrated QCR. Detector localization evidence is converted into candidate crops, which are scored by a VLM to obtain localized anomaly evidence. Candidate quality calibrates the crop-level VLM score and forms the main method core. Adaptive consistency is used only as a conservative refinement, while fixed Q+C fusion is retained as a diagnostic ablation.",
        "```",
        "",
        "### Figure 2 caption draft",
        "",
        "```text",
        "Representative boundary cases for quality-calibrated localization-guided VLM reasoning. Quality calibration can boost true anomaly evidence and suppress normal false positives, but it can also fail when candidate quality is misleading or detector and VLM evidence disagree. These examples illustrate method boundaries rather than universal behavior.",
        "```",
        "",
        "## 7. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Paper Stage P9: BibTeX/reference preparation and citation placement",
        "```",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cases = read_csv_robust(IN_CASES)
    _ = read_csv_robust(IN_BOUNDARY)
    _ = read_csv_robust(IN_TABLE_INV) if IN_TABLE_INV.exists() else pd.DataFrame()

    nodes = write_figure1_svg()
    selected = select_boundary_cases(cases)
    checklist = make_checklist(selected)

    nodes.to_csv(OUT_FIG1_NODES, index=False, lineterminator="\n")
    selected.to_csv(OUT_FIG2_CASES, index=False, lineterminator="\n")
    checklist.to_csv(OUT_CHECKLIST, index=False, lineterminator="\n")

    write_figure2_plan(selected)
    write_report(nodes, selected, checklist)

    print("[DONE]", OUT_FIG1)
    print("[DONE]", OUT_FIG1_NODES)
    print("[DONE]", OUT_FIG2_CASES)
    print("[DONE]", OUT_FIG2_PLAN)
    print("[DONE]", OUT_CHECKLIST)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== selected boundary cases =====")
    print(selected.to_string(index=False))
    print()
    print("===== checklist =====")
    print(checklist.to_string(index=False))


if __name__ == "__main__":
    main()
