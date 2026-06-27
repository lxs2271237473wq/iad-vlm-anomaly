from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(".").resolve()

STAGE10_REGIONS = ROOT / "results/stage10_dataset_expansion/stage10_d_patchcore_candidate_regions.csv"
STAGE11_REGIONS = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv"
STAGE11A_MAPPING = ROOT / "results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_mapping.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_KEYS = OUT_DIR / "stage11_f3_vial_image_set_keys.csv"
OUT_OVERLAP = OUT_DIR / "stage11_f3_vial_image_set_overlap.csv"
OUT_SUMMARY = OUT_DIR / "stage11_f3_vial_image_set_overlap_summary.csv"
OUT_REPORT = DOC_DIR / "stage11_f3_vial_image_set_overlap_report.md"


def rel_norm(x) -> str:
    text = "" if pd.isna(x) else str(x).replace("\\", "/")
    root = str(ROOT).replace("\\", "/")

    if text.startswith(root + "/"):
        text = text[len(root) + 1:]

    return text.strip()


def strip_adapter_prefix(stem: str) -> str:
    return re.sub(r"^\d{6}_", "", stem)


def label_from_path_or_gt(path_text, gt=None) -> str:
    text = rel_norm(path_text).lower()

    if "/test/good/" in text or "/train/good/" in text:
        return "good"

    if "/test/bad/" in text or "/ground_truth/bad/" in text:
        return "bad"

    if gt is not None and not pd.isna(gt):
        s = str(gt).strip().lower()
        if s in {"1", "true", "bad", "anomaly", "abnormal", "tensor(1)"}:
            return "bad"
        if s in {"0", "false", "good", "normal", "tensor(0)"}:
            return "good"
        try:
            return "bad" if float(s) > 0 else "good"
        except Exception:
            pass

    return "unknown"


def stem_key(path_text) -> str:
    return strip_adapter_prefix(Path(rel_norm(path_text)).stem)


def labeled_stem_key(path_text, gt=None) -> str:
    return f"{label_from_path_or_gt(path_text, gt)}:{stem_key(path_text)}"


def compact_key(path_text, gt=None) -> str:
    stem = stem_key(path_text)
    # Keep only trailing original AD2-like id if possible.
    # Examples:
    # 000372_000_shift_4 -> 000_shift_4 after strip_adapter_prefix
    # 000_shift_4 -> 000_shift_4
    # 001_underexposed -> 001_underexposed
    return f"{label_from_path_or_gt(path_text, gt)}:{stem}"


def build_stage11_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(STAGE11A_MAPPING)
    mapping = mapping[mapping["category"].astype(str) == "vial"].copy()

    rows = []
    for _, r in mapping.iterrows():
        target = rel_norm(r["target_image_path"])
        source = rel_norm(r["source_image_path"])

        rows.append({
            "target_image_path": target,
            "target_basename": Path(target).name,
            "source_image_path": source,
            "source_basename": Path(source).name,
            "source_stem": stem_key(source),
            "source_labeled_key": labeled_stem_key(source, r.get("is_anomaly", None)),
            "target_stem": stem_key(target),
            "target_labeled_key": labeled_stem_key(target, r.get("is_anomaly", None)),
            "is_anomaly": r.get("is_anomaly", ""),
            "source_split": r.get("source_split", ""),
            "target_subset": r.get("target_subset", ""),
            "target_label": r.get("target_label", ""),
        })

    return pd.DataFrame(rows)


def collect_stage10_keys() -> pd.DataFrame:
    df = pd.read_csv(STAGE10_REGIONS)
    df = df[df["category"].astype(str) == "vial"].copy()

    rows = []
    for image_path, part in df.groupby("image_path", sort=False):
        first = part.iloc[0]
        gt = first.get("gt_label", None)

        rows.append({
            "stage": "stage10",
            "image_path": rel_norm(image_path),
            "basename": Path(rel_norm(image_path)).name,
            "stem": stem_key(image_path),
            "label": label_from_path_or_gt(image_path, gt),
            "labeled_key": labeled_stem_key(image_path, gt),
            "candidate_rows": int(len(part)),
            "gt_label": gt,
            "pred_score_first": first.get("pred_score", ""),
        })

    return pd.DataFrame(rows)


def collect_stage11_keys(mapping: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(STAGE11_REGIONS)
    df = df[df["category"].astype(str) == "vial"].copy()

    target_to_source: Dict[str, pd.Series] = {}
    for _, r in mapping.iterrows():
        target_to_source[rel_norm(r["target_image_path"])] = r
        target_to_source[Path(rel_norm(r["target_image_path"])).name] = r

    rows = []
    for image_path, part in df.groupby("image_path", sort=False):
        first = part.iloc[0]
        rel = rel_norm(image_path)
        m = target_to_source.get(rel, target_to_source.get(Path(rel).name, None))

        if m is not None:
            source_path = m["source_image_path"]
            source_labeled_key = m["source_labeled_key"]
            source_stem = m["source_stem"]
            source_split = m["source_split"]
        else:
            source_path = ""
            source_labeled_key = labeled_stem_key(image_path, first.get("gt_label", None))
            source_stem = stem_key(image_path)
            source_split = ""

        rows.append({
            "stage": "stage11",
            "image_path": rel,
            "basename": Path(rel).name,
            "stem": stem_key(image_path),
            "label": label_from_path_or_gt(image_path, first.get("gt_label", None)),
            "labeled_key": labeled_stem_key(image_path, first.get("gt_label", None)),
            "source_image_path": source_path,
            "source_stem": source_stem,
            "source_labeled_key": source_labeled_key,
            "source_split": source_split,
            "candidate_rows": int(len(part)),
            "gt_label": first.get("gt_label", ""),
            "pred_score_first": first.get("pred_score", ""),
        })

    return pd.DataFrame(rows)


def overlap_table(stage10: pd.DataFrame, stage11: pd.DataFrame) -> pd.DataFrame:
    rows = []

    comparisons = [
        ("stage10_labeled_key_vs_stage11_source_labeled_key", "labeled_key", "source_labeled_key"),
        ("stage10_stem_vs_stage11_source_stem", "stem", "source_stem"),
        ("stage10_labeled_key_vs_stage11_labeled_key", "labeled_key", "labeled_key"),
        ("stage10_stem_vs_stage11_stem", "stem", "stem"),
    ]

    for name, c10, c11 in comparisons:
        s10 = set(stage10[c10].dropna().astype(str).tolist())
        s11 = set(stage11[c11].dropna().astype(str).tolist())

        rows.append({
            "comparison": name,
            "stage10_key_col": c10,
            "stage11_key_col": c11,
            "stage10_unique": len(s10),
            "stage11_unique": len(s11),
            "intersection": len(s10 & s11),
            "stage10_only": len(s10 - s11),
            "stage11_only": len(s11 - s10),
            "jaccard": len(s10 & s11) / max(1, len(s10 | s11)),
        })

    return pd.DataFrame(rows)


def unmatched_examples(stage10: pd.DataFrame, stage11: pd.DataFrame, key10: str, key11: str) -> pd.DataFrame:
    s10 = set(stage10[key10].dropna().astype(str).tolist())
    s11 = set(stage11[key11].dropna().astype(str).tolist())

    rows = []

    for k in sorted(s10 - s11):
        r = stage10[stage10[key10].astype(str) == k].iloc[0].to_dict()
        rows.append({
            "missing_from": "stage11",
            "key": k,
            "stage10_image_path": r.get("image_path", ""),
            "stage10_basename": r.get("basename", ""),
            "stage10_label": r.get("label", ""),
            "stage11_image_path": "",
            "stage11_source_image_path": "",
        })

    for k in sorted(s11 - s10):
        r = stage11[stage11[key11].astype(str) == k].iloc[0].to_dict()
        rows.append({
            "missing_from": "stage10",
            "key": k,
            "stage10_image_path": "",
            "stage10_basename": "",
            "stage10_label": "",
            "stage11_image_path": r.get("image_path", ""),
            "stage11_source_image_path": r.get("source_image_path", ""),
        })

    return pd.DataFrame(rows)


def write_report(keys: pd.DataFrame, overlap: pd.DataFrame, unmatched: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    best = overlap.sort_values(["intersection", "jaccard"], ascending=[False, False]).iloc[0]

    lines: List[str] = []

    lines.append("# Stage 11-F3 Vial Image-set Overlap Audit")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report audits whether Stage 10-D/G and Stage 11-C/D use the same vial image set.")
    lines.append("It does not run PatchCore, VLM inference, or crop generation.")
    lines.append("")
    lines.append("## 2. Why this matters")
    lines.append("")
    lines.append("Stage 10-G and Stage 11-D reported different vial context-crop effects. Before attributing this to candidate policy drift, the exact image sets must be aligned.")
    lines.append("")
    lines.append("## 3. Key-overlap diagnostics")
    lines.append("")
    lines.append("| Comparison | Stage10 unique | Stage11 unique | Intersection | Stage10 only | Stage11 only | Jaccard |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, r in overlap.iterrows():
        lines.append(
            f"| {r['comparison']} | {int(r['stage10_unique'])} | {int(r['stage11_unique'])} | "
            f"{int(r['intersection'])} | {int(r['stage10_only'])} | {int(r['stage11_only'])} | "
            f"{float(r['jaccard']):.4f} |"
        )

    lines.append("")
    lines.append("## 4. Best available alignment")
    lines.append("")
    lines.append(
        f"The best available alignment is `{best['comparison']}`, with intersection "
        f"`{int(best['intersection'])}` out of Stage 10 `{int(best['stage10_unique'])}` and Stage 11 `{int(best['stage11_unique'])}` unique keys."
    )
    lines.append("")
    lines.append("## 5. Decision")
    lines.append("")

    if int(best["intersection"]) >= 68:
        lines.append("The image sets are effectively aligned. The next diagnostic can compare candidate boxes and VLM scores directly.")
    else:
        lines.append("The image sets are not sufficiently aligned. Stage 10-G and Stage 11-D vial AUROC should not be treated as directly comparable until the source of the image-set mismatch is resolved.")
        lines.append("")
        lines.append("The next step should inspect unmatched examples and decide whether to rerun vial under a single unified Stage 11 pipeline or retire Stage 10-G vial as a historical single-category result.")

    lines.append("")
    lines.append("## 6. Output")
    lines.append("")
    lines.append(f"- Keys CSV: `{OUT_KEYS.relative_to(ROOT)}`")
    lines.append(f"- Overlap CSV: `{OUT_OVERLAP.relative_to(ROOT)}`")
    lines.append(f"- Summary CSV: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    mapping = build_stage11_mapping()
    stage10 = collect_stage10_keys()
    stage11 = collect_stage11_keys(mapping)

    keys = pd.concat([stage10, stage11], ignore_index=True, sort=False)
    overlap = overlap_table(stage10, stage11)

    best = overlap.sort_values(["intersection", "jaccard"], ascending=[False, False]).iloc[0]
    unmatched = unmatched_examples(
        stage10,
        stage11,
        str(best["stage10_key_col"]),
        str(best["stage11_key_col"]),
    )

    summary = pd.DataFrame([{
        "best_comparison": best["comparison"],
        "best_intersection": int(best["intersection"]),
        "stage10_unique": int(best["stage10_unique"]),
        "stage11_unique": int(best["stage11_unique"]),
        "stage10_only": int(best["stage10_only"]),
        "stage11_only": int(best["stage11_only"]),
        "jaccard": float(best["jaccard"]),
        "directly_comparable": bool(int(best["intersection"]) >= 68),
    }])

    keys.to_csv(OUT_KEYS, index=False)
    overlap.to_csv(OUT_OVERLAP, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(keys, overlap, unmatched)

    print("[DONE]", OUT_KEYS)
    print("[DONE]", OUT_OVERLAP)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)

    print("\n===== overlap =====")
    print(overlap.to_string(index=False))

    print("\n===== summary =====")
    print(summary.to_string(index=False))

    print("\n===== unmatched examples =====")
    print(unmatched.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
