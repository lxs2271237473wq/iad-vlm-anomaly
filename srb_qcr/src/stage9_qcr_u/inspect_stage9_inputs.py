from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(".").resolve()
OUT_DIR = ROOT / "results" / "stage9_qcr_u"
OUT_CSV = OUT_DIR / "stage9_a0_input_structure.csv"
OUT_MD = OUT_DIR / "stage9_a0_input_structure_report.md"


IMPORTANT_ROOTS = [
    ROOT / "results",
    ROOT / "runs",
]

RELEVANT_KEYWORDS = [
    "candidate",
    "region",
    "bbox",
    "box",
    "prediction",
    "prompt",
    "reasoning",
    "binary",
    "clip",
    "patchcore",
    "fastflow",
    "visa",
    "stage7",
    "stage8",
]

FIELD_GROUPS: Dict[str, List[str]] = {
    "image_path": ["image_path", "img_path", "path", "image", "filename", "file_path"],
    "bbox": ["bbox", "box", "x1", "y1", "x2", "y2", "xmin", "ymin", "xmax", "ymax"],
    "label": ["label", "gt", "target", "is_anomaly", "anomaly_label", "class"],
    "category": ["category", "object", "object_type", "class_name"],
    "candidate_quality": ["candidate", "region_score", "candidate_score", "anomaly_score", "area", "mean_score", "max_score"],
    "vlm_score": ["clip", "vlm", "prompt", "normal_score", "anomaly_score", "crop_score", "full_score", "margin"],
    "normal_score": ["normal_score", "normal_prompt_score", "score_normal"],
    "anomaly_score": ["anomaly_score", "abnormal_score", "anomaly_prompt_score", "score_anomaly"],
    "crop_score": ["crop_score", "crop_auroc", "crop_margin", "crop_anomaly_score"],
    "full_score": ["full_score", "full_image_score", "full_margin", "full_anomaly_score"],
}


def is_relevant_csv(path: Path) -> bool:
    text = str(path.relative_to(ROOT)).lower()
    if path.suffix.lower() != ".csv":
        return False
    return any(k in text for k in RELEVANT_KEYWORDS)


def infer_backbone(path: Path) -> str:
    text = str(path).lower()
    if "patchcore" in text:
        return "PatchCore"
    if "fastflow" in text:
        return "FastFlow"
    return "unknown"


def infer_role(path: Path, columns: List[str]) -> str:
    text = str(path).lower()
    col_text = " ".join(columns).lower()

    if "candidate" in text or "region" in text or "bbox" in col_text or "box" in col_text:
        return "candidate_or_region"
    if "prompt" in text or "reasoning" in text or "clip" in text or "normal_score" in col_text or "anomaly_score" in col_text:
        return "vlm_reasoning"
    if "prediction" in text or "metric" in text or "summary" in text or "auroc" in col_text:
        return "detector_or_summary"
    return "unknown"


def has_group(columns: List[str], group_terms: List[str]) -> bool:
    lower_cols = [c.lower() for c in columns]
    for c in lower_cols:
        for term in group_terms:
            if term in c:
                return True
    return False


def inspect_csv(path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "backbone": infer_backbone(path),
        "role": "unread",
        "rows": None,
        "cols": None,
        "columns": "",
        "preview_json": "",
        "error": "",
    }

    try:
        df_head = pd.read_csv(path, nrows=5)
        columns = list(df_head.columns)
        row["cols"] = len(columns)
        row["columns"] = json.dumps(columns, ensure_ascii=False)
        row["preview_json"] = df_head.head(3).to_json(orient="records", force_ascii=False)

        try:
            row["rows"] = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
        except Exception:
            row["rows"] = None

        row["role"] = infer_role(path, columns)

        for group_name, group_terms in FIELD_GROUPS.items():
            row[f"has_{group_name}"] = has_group(columns, group_terms)

    except Exception as e:
        row["error"] = repr(e)
        for group_name in FIELD_GROUPS:
            row[f"has_{group_name}"] = False

    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files: List[Path] = []
    for root in IMPORTANT_ROOTS:
        if root.exists():
            csv_files.extend([p for p in root.rglob("*.csv") if is_relevant_csv(p)])

    csv_files = sorted(set(csv_files), key=lambda p: str(p))

    rows = [inspect_csv(p) for p in csv_files]
    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame([{
            "path": "",
            "size_bytes": "",
            "backbone": "",
            "role": "",
            "rows": "",
            "cols": "",
            "columns": "",
            "preview_json": "",
            "error": "No relevant CSV files found under results/ or runs/.",
        }])

    df.to_csv(OUT_CSV, index=False)

    lines: List[str] = []
    lines.append("# Stage 9-A0 QCR-U 输入文件结构检查报告")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report inspects existing CSV files required by the next QCR-U fusion step.")
    lines.append("It does not train models, rerun CLIP, regenerate anomaly maps, or modify previous experiment results.")
    lines.append("")
    lines.append("## 2. Scanned Files")
    lines.append("")
    lines.append(f"- Total relevant CSV files: {len(rows)}")
    lines.append(f"- Output CSV: `{OUT_CSV.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 3. QCR-U Required Information")
    lines.append("")
    lines.append("| Required item | Meaning |")
    lines.append("|---|---|")
    lines.append("| candidate / region / bbox | Candidate location and region information for Q_i |")
    lines.append("| detector anomaly score / region score | Detector-side evidence for candidate quality |")
    lines.append("| normal/anomaly prompt scores | VLM-side margin M_i |")
    lines.append("| image path / category / label | Alignment keys for merging detector and VLM records |")
    lines.append("| crop score / full score | Baseline comparison and fusion input |")
    lines.append("")
    lines.append("## 4. File-level Summary")
    lines.append("")
    lines.append("| Path | Backbone | Role | Rows | Cols | Key Fields | Error |")
    lines.append("|---|---|---:|---:|---:|---|---|")

    for row in rows:
        key_fields = []
        for group_name in FIELD_GROUPS:
            if row.get(f"has_{group_name}", False):
                key_fields.append(group_name)
        key_text = ", ".join(key_fields) if key_fields else "-"
        lines.append(
            f"| `{row.get('path', '')}` | {row.get('backbone', '')} | {row.get('role', '')} | "
            f"{row.get('rows', '')} | {row.get('cols', '')} | {key_text} | {row.get('error', '')} |"
        )

    lines.append("")
    lines.append("## 5. Column Details")
    lines.append("")

    for row in rows:
        lines.append(f"### `{row.get('path', '')}`")
        lines.append("")
        lines.append(f"- Backbone: `{row.get('backbone', '')}`")
        lines.append(f"- Role: `{row.get('role', '')}`")
        lines.append(f"- Rows: `{row.get('rows', '')}`")
        lines.append(f"- Cols: `{row.get('cols', '')}`")
        lines.append(f"- Columns: `{row.get('columns', '')}`")
        if row.get("error"):
            lines.append(f"- Error: `{row.get('error')}`")
        lines.append("")
        lines.append("Preview:")
        lines.append("")
        lines.append("```json")
        lines.append(str(row.get("preview_json", "")))
        lines.append("```")
        lines.append("")

    lines.append("## 6. Next Step Decision")
    lines.append("")
    lines.append("After this report is committed, use the real column names to implement Stage 9-A1:")
    lines.append("")
    lines.append("```text")
    lines.append("Q_i: candidate quality")
    lines.append("M_i: VLM abnormal margin")
    lines.append("K_i: detector-VLM consistency")
    lines.append("F_i = alpha * M_i + beta * Q_i + gamma * K_i")
    lines.append("F_image = max_i F_i")
    lines.append("```")
    lines.append("")
    lines.append("Do not implement Stage 9-A1 before confirming the merge keys and score columns from this report.")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_MD)
    print("relevant_csv_files:", len(rows))


if __name__ == "__main__":
    main()
