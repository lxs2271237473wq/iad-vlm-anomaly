from __future__ import annotations

from pathlib import Path
from io import StringIO
import itertools
import re
import pandas as pd


ROOT = Path(".").resolve()

AD2_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

SOURCE_FILES = [
    ROOT / "results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_predictions.csv",
    ROOT / "results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv",
    ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv",
    ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_candidate_scores.csv",
    ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_image_predictions.csv",
    ROOT / "results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv",
    ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
    ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_score_complementarity.csv",
]

OUT_DIR = ROOT / "results/stage18_ad2_qcr_ablation"
DOC_DIR = ROOT / "docs/stage18_ad2_qcr_ablation"

OUT_PROFILE = OUT_DIR / "stage18_b1_ad2_qcr_source_schema_profile.csv"
OUT_COLUMNS = OUT_DIR / "stage18_b1_ad2_qcr_source_columns_long.csv"
OUT_JOIN = OUT_DIR / "stage18_b1_ad2_qcr_join_feasibility.csv"
OUT_REPORT = DOC_DIR / "stage18_b1_ad2_qcr_schema_profile_report.md"


KEY_PATTERNS = [
    "image_key", "image_path", "img_path", "path", "filename", "file_name",
    "category", "class", "object", "sample_id", "candidate_id", "crop_id",
    "candidate_rank", "rank", "bbox", "x1", "y1", "x2", "y2",
]

LABEL_PATTERNS = [
    "label", "gt", "target", "is_anomaly", "anomaly", "defect",
]

DETECTOR_PATTERNS = [
    "detector", "patchcore", "fastflow", "efficientad",
    "anomaly_score", "image_score", "heat", "map_score",
]

VLM_PATTERNS = [
    "vlm", "clip", "winclip", "llava", "gpt", "gemini",
    "abnormal", "normal_prompt", "anomaly_prompt", "language",
    "context", "caption",
]

QUALITY_PATTERNS = [
    "quality", "candidate_quality", "area", "bbox", "crop",
    "coverage", "compactness", "max", "mean", "region",
    "saliency", "mask", "localization",
]

CONSISTENCY_PATTERNS = [
    "consistency", "agreement", "high_high", "hh", "k",
]

NUMERIC_EXCLUDE = {
    "width", "height", "x", "y", "x1", "y1", "x2", "y2",
}


def read_csv_robust(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    raw = path.read_text(encoding="utf-8", errors="ignore").strip()

    # Fallback for accidentally flattened CSV files.
    if "\n" not in raw and "," in raw:
        # Try common ISO timestamp row splitting first.
        repaired = re.sub(r"\s+(?=\d{4}-\d{2}-\d{2}T)", "\n", raw)
        try:
            df = pd.read_csv(StringIO(repaired))
            if len(df.columns) > 1:
                return df
        except Exception:
            pass

    return None


def has_any(name: str, patterns: list[str]) -> bool:
    n = name.lower()
    return any(p.lower() in n for p in patterns)


def classify_col(col: str) -> str:
    c = col.lower()

    if has_any(c, KEY_PATTERNS):
        return "key_or_geometry"
    if has_any(c, LABEL_PATTERNS):
        return "label"
    if has_any(c, VLM_PATTERNS):
        return "vlm_or_text_score"
    if has_any(c, QUALITY_PATTERNS):
        return "quality_or_region_feature"
    if has_any(c, CONSISTENCY_PATTERNS):
        return "consistency"
    if has_any(c, DETECTOR_PATTERNS):
        return "detector_score"

    return "other"


def ad2_categories_in_df(df: pd.DataFrame, path: Path) -> list[str]:
    found = set()

    path_s = str(path).replace("\\", "/")
    for c in AD2_CATEGORIES:
        if c in path_s:
            found.add(c)

    for col in df.columns:
        if col.lower() in {"category", "class", "object", "object_name", "category_name"}:
            vals = set(df[col].dropna().astype(str).unique())
            for c in AD2_CATEGORIES:
                if c in vals:
                    found.add(c)

    for col in df.columns:
        if col.lower() in {"image_path", "image_key", "path", "filename", "file_name"}:
            sample = "\n".join(df[col].dropna().astype(str).head(5000).tolist())
            for c in AD2_CATEGORIES:
                if c in sample:
                    found.add(c)

    return sorted(found)


def rows_by_category(df: pd.DataFrame) -> dict[str, int]:
    for col in ["category", "class", "object", "category_name"]:
        if col in df.columns:
            return {
                cat: int((df[col].astype(str) == cat).sum())
                for cat in AD2_CATEGORIES
            }

    # Fallback by image path strings.
    out = {}
    text_cols = [c for c in ["image_path", "image_key", "path", "filename"] if c in df.columns]
    for cat in AD2_CATEGORIES:
        count = 0
        for c in text_cols:
            count += int(df[c].astype(str).str.contains(cat, regex=False).sum())
        out[cat] = count
    return out


def numeric_summary(df: pd.DataFrame, col: str) -> tuple[str, str, str]:
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return "", "", ""
    return f"{s.min():.6g}", f"{s.max():.6g}", f"{s.mean():.6g}"


def profile_file(path: Path) -> tuple[dict, list[dict]]:
    df = read_csv_robust(path)

    if df is None:
        return {
            "file": str(path.relative_to(ROOT)),
            "exists_readable": False,
            "num_rows": None,
            "num_cols": None,
            "ad2_categories_found": "",
            "ad2_coverage_count": 0,
            "rows_fruit_jelly": 0,
            "rows_sheet_metal": 0,
            "rows_vial": 0,
            "rows_walnuts": 0,
            "key_cols": "",
            "label_cols": "",
            "detector_like_cols": "",
            "vlm_like_cols": "",
            "quality_like_cols": "",
            "consistency_like_cols": "",
            "numeric_cols": "",
            "qcr_readiness": "unreadable",
            "notes": "missing or unreadable",
        }, []

    cats = ad2_categories_in_df(df, path)
    row_counts = rows_by_category(df)

    classified = {c: classify_col(c) for c in df.columns}

    key_cols = [c for c, t in classified.items() if t == "key_or_geometry"]
    label_cols = [c for c, t in classified.items() if t == "label"]
    detector_cols = [c for c, t in classified.items() if t == "detector_score"]
    vlm_cols = [c for c, t in classified.items() if t == "vlm_or_text_score"]
    quality_cols = [c for c, t in classified.items() if t == "quality_or_region_feature"]
    consistency_cols = [c for c, t in classified.items() if t == "consistency"]

    numeric_cols = []
    col_rows = []

    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        is_numeric = s.notna().sum() > 0 and s.notna().sum() >= max(3, int(0.2 * len(df)))

        if is_numeric:
            numeric_cols.append(c)

        mn, mx, mean = numeric_summary(df, c)

        col_rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "column": c,
                "classified_as": classified[c],
                "dtype": str(df[c].dtype),
                "non_null": int(df[c].notna().sum()),
                "unique_sample_count": int(df[c].dropna().astype(str).nunique()),
                "numeric_min": mn,
                "numeric_max": mx,
                "numeric_mean": mean,
                "sample_values": "; ".join(df[c].dropna().astype(str).unique()[:5]),
            }
        )

    has_label = len(label_cols) > 0
    has_detector = len(detector_cols) > 0
    has_vlm = len(vlm_cols) > 0
    has_quality = len(quality_cols) > 0
    has_key = any(c in df.columns for c in ["image_path", "image_key", "path", "filename", "category"])

    if len(cats) == 4 and has_key and has_label and has_detector and has_vlm and has_quality:
        readiness = "qcr_ready"
    elif len(cats) == 4 and has_key and has_label and (has_detector or has_vlm or has_quality):
        readiness = "partial_join_source"
    elif len(cats) > 0:
        readiness = "ad2_summary_or_auxiliary"
    else:
        readiness = "non_ad2_or_irrelevant"

    profile = {
        "file": str(path.relative_to(ROOT)),
        "exists_readable": True,
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "ad2_categories_found": ";".join(cats),
        "ad2_coverage_count": len(cats),
        "rows_fruit_jelly": row_counts.get("fruit_jelly", 0),
        "rows_sheet_metal": row_counts.get("sheet_metal", 0),
        "rows_vial": row_counts.get("vial", 0),
        "rows_walnuts": row_counts.get("walnuts", 0),
        "key_cols": ";".join(key_cols),
        "label_cols": ";".join(label_cols),
        "detector_like_cols": ";".join(detector_cols),
        "vlm_like_cols": ";".join(vlm_cols),
        "quality_like_cols": ";".join(quality_cols),
        "consistency_like_cols": ";".join(consistency_cols),
        "numeric_cols": ";".join(numeric_cols),
        "qcr_readiness": readiness,
        "notes": "",
    }

    return profile, col_rows


def normalized_join_key_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}

    for col in ["image_path", "path", "img_path", "filename", "file_name", "image_key"]:
        if col in df.columns:
            s = df[col].astype(str)
            out[col] = s
            out[col + "_basename"] = s.map(lambda x: Path(x).name)
            out[col + "_stem"] = s.map(lambda x: Path(x).stem)

    if "category" in df.columns:
        out["category"] = df["category"].astype(str)

    return out


def join_feasibility(source_paths: list[Path]) -> pd.DataFrame:
    loaded = {}
    for p in source_paths:
        df = read_csv_robust(p)
        if df is not None and len(df.columns) > 1:
            loaded[str(p.relative_to(ROOT))] = df

    rows = []

    for (file_a, df_a), (file_b, df_b) in itertools.combinations(loaded.items(), 2):
        keys_a = normalized_join_key_series(df_a)
        keys_b = normalized_join_key_series(df_b)

        best = {
            "file_a": file_a,
            "file_b": file_b,
            "best_key_a": "",
            "best_key_b": "",
            "overlap_count": 0,
            "overlap_ratio_a": 0.0,
            "overlap_ratio_b": 0.0,
            "notes": "",
        }

        for ka, sa in keys_a.items():
            set_a = set(sa.dropna().astype(str))
            if not set_a:
                continue
            for kb, sb in keys_b.items():
                set_b = set(sb.dropna().astype(str))
                if not set_b:
                    continue

                inter = len(set_a & set_b)
                if inter > best["overlap_count"]:
                    best.update(
                        {
                            "best_key_a": ka,
                            "best_key_b": kb,
                            "overlap_count": inter,
                            "overlap_ratio_a": inter / max(1, len(set_a)),
                            "overlap_ratio_b": inter / max(1, len(set_b)),
                        }
                    )

        if best["overlap_count"] == 0:
            best["notes"] = "no obvious key overlap"
        elif best["overlap_ratio_a"] > 0.8 or best["overlap_ratio_b"] > 0.8:
            best["notes"] = "strong possible join"
        else:
            best["notes"] = "partial possible join"

        rows.append(best)

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    profile_rows = []
    column_rows = []

    for path in SOURCE_FILES:
        profile, cols = profile_file(path)
        profile_rows.append(profile)
        column_rows.extend(cols)

    profile = pd.DataFrame(profile_rows)
    columns = pd.DataFrame(column_rows)
    joins = join_feasibility(SOURCE_FILES)

    profile.to_csv(OUT_PROFILE, index=False, lineterminator="\n")
    columns.to_csv(OUT_COLUMNS, index=False, lineterminator="\n")
    joins.to_csv(OUT_JOIN, index=False, lineterminator="\n")

    qcr_ready = profile[profile["qcr_readiness"] == "qcr_ready"]
    partial = profile[profile["qcr_readiness"] == "partial_join_source"]

    lines = [
        "# Stage 18-B1 AD2 QCR Source Schema Profile",
        "",
        "## Purpose",
        "",
        "Inspect Stage11/Stage13 AD2 source files to decide whether AD2 four-category QCR predictions can be assembled from existing files.",
        "",
        "## Summary",
        "",
        f"- qcr_ready files: `{len(qcr_ready)}`",
        f"- partial_join_source files: `{len(partial)}`",
        "",
        "## File profile",
        "",
        "| File | Rows | AD2 coverage | Readiness | Key cols | Label cols | Detector-like | VLM-like | Quality-like |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]

    for _, r in profile.iterrows():
        lines.append(
            f"| `{r['file']}` | {r['num_rows']} | {r['ad2_coverage_count']}/4 | "
            f"{r['qcr_readiness']} | {r['key_cols']} | {r['label_cols']} | "
            f"{r['detector_like_cols']} | {r['vlm_like_cols']} | {r['quality_like_cols']} |"
        )

    lines += [
        "",
        "## Strong join candidates",
        "",
        "| File A | File B | Key A | Key B | Overlap | Ratio A | Ratio B | Notes |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]

    if joins.empty:
        lines.append("| none | none |  |  | 0 | 0 | 0 | no files loaded |")
    else:
        top = joins.sort_values(["overlap_count", "overlap_ratio_a"], ascending=[False, False]).head(20)
        for _, r in top.iterrows():
            lines.append(
                f"| `{r['file_a']}` | `{r['file_b']}` | {r['best_key_a']} | {r['best_key_b']} | "
                f"{r['overlap_count']} | {float(r['overlap_ratio_a']):.3f} | "
                f"{float(r['overlap_ratio_b']):.3f} | {r['notes']} |"
            )

    lines += [
        "",
        "## Decision rule",
        "",
        "- If a qcr_ready file exists, proceed to Stage 18-B2 directly.",
        "- If partial files have strong joins and contain D/M/Q/label across files, assemble in Stage 18-B2.",
        "- If VLM or quality is missing, proceed to Stage 18-C to generate missing AD2 QCR predictions.",
        "",
    ]

    if len(qcr_ready) > 0:
        decision = "proceed_to_stage18_b2_direct_qcr_assembly"
    elif len(partial) > 0:
        decision = "inspect_columns_and_attempt_stage18_b2_partial_join"
    else:
        decision = "proceed_to_stage18_c_generate_missing_predictions"

    lines += [
        "## Recommended next action",
        "",
        f"`{decision}`",
        "",
        "## Outputs",
        "",
        f"- `{OUT_PROFILE.relative_to(ROOT)}`",
        f"- `{OUT_COLUMNS.relative_to(ROOT)}`",
        f"- `{OUT_JOIN.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_PROFILE)
    print("[DONE]", OUT_COLUMNS)
    print("[DONE]", OUT_JOIN)
    print("[DONE]", OUT_REPORT)
    print()
    print(profile.to_string(index=False))
    print()
    print("===== top joins =====")
    if joins.empty:
        print("none")
    else:
        print(joins.sort_values(["overlap_count", "overlap_ratio_a"], ascending=[False, False]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
