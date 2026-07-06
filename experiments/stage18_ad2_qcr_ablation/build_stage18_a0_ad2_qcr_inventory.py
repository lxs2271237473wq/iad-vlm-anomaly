from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

CANDIDATE_FILES = [
    ROOT / "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv",
    ROOT / "results/stage16_qcru_ablation/stage16_b_adaptive_qcru_all_variants_per_config.csv",
    ROOT / "results/stage16_qcru_ablation/stage16_b_adaptive_qcru_all_variants_per_category.csv",
    ROOT / "results/stage16_qcru_ablation/stage16_d_paper_facing_qcr_ablation_table.csv",
]

AD2_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

OUT_CSV = ROOT / "results/stage18_ad2_qcr_ablation/stage18_a0_ad2_qcr_inventory.csv"
OUT_REPORT = ROOT / "docs/stage18_ad2_qcr_ablation/stage18_a0_ad2_qcr_inventory_report.md"


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        if len(df.columns) <= 1:
            return None
        return df
    except Exception:
        return None


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for path in CANDIDATE_FILES:
        df = read_csv_safe(path)

        if df is None:
            rows.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "exists_and_readable": False,
                    "num_rows": None,
                    "num_cols": None,
                    "has_dataset_col": False,
                    "datasets": "",
                    "has_category_col": False,
                    "categories_found": "",
                    "ad2_categories_found": "",
                    "ad2_coverage_count": 0,
                    "can_directly_run_ad2_qcr_ablation": False,
                    "notes": "missing or unreadable",
                }
            )
            continue

        cols = set(df.columns)
        has_dataset = "dataset" in cols
        has_category = "category" in cols

        datasets = ""
        if has_dataset:
            datasets = ";".join(sorted(map(str, df["dataset"].dropna().unique())))

        categories_found = ""
        ad2_found = []

        if has_category:
            cats = sorted(map(str, df["category"].dropna().unique()))
            categories_found = ";".join(cats)
            ad2_found = [c for c in AD2_CATEGORIES if c in cats]

        required_score_cols = [
            "vlm_score_norm",
            "detector_score_norm",
            "candidate_quality_norm",
        ]

        has_required_score_cols = all(c in cols for c in required_score_cols)

        can_direct = (
            has_category
            and len(ad2_found) == len(AD2_CATEGORIES)
            and has_required_score_cols
        )

        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "exists_and_readable": True,
                "num_rows": len(df),
                "num_cols": len(df.columns),
                "has_dataset_col": has_dataset,
                "datasets": datasets,
                "has_category_col": has_category,
                "categories_found": categories_found,
                "ad2_categories_found": ";".join(ad2_found),
                "ad2_coverage_count": len(ad2_found),
                "can_directly_run_ad2_qcr_ablation": can_direct,
                "notes": (
                    "contains AD2 QCR-ready predictions"
                    if can_direct
                    else "does not contain full AD2 QCR-ready prediction columns/categories"
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, lineterminator="\n")

    can_any = bool(out["can_directly_run_ad2_qcr_ablation"].any())

    lines = [
        "# Stage 18-A0 AD2 QCR Inventory",
        "",
        "## Purpose",
        "",
        "Check whether existing QCR prediction/result files already contain AD2 four-category data for:",
        "",
        "```text",
        "fruit_jelly",
        "sheet_metal",
        "vial",
        "walnuts",
        "```",
        "",
        "## Decision",
        "",
        f"- can_directly_run_ad2_qcr_ablation: `{can_any}`",
        "",
        "## Inventory",
        "",
        "| File | Readable | Rows | AD2 coverage | Directly usable | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for _, r in out.iterrows():
        lines.append(
            f"| `{r['file']}` | {int(bool(r['exists_and_readable']))} | "
            f"{r['num_rows']} | {r['ad2_coverage_count']}/4 | "
            f"{int(bool(r['can_directly_run_ad2_qcr_ablation']))} | {r['notes']} |"
        )

    lines += [
        "",
        "## Next Action",
        "",
    ]

    if can_any:
        lines += [
            "Proceed to Stage 18-A1: run AD2 four-category QCR ablation directly from existing prediction file.",
        ]
    else:
        lines += [
            "Proceed to Stage 18-B: generate missing AD2 QCR prediction file first.",
            "",
            "This means the current QCR ablation evidence is not yet aligned with the AD2 four-category system-level baseline.",
        ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_REPORT)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
