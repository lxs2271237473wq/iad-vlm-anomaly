from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_METRICS = ROOT / "results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_metrics.csv"
IN_STATUS = ROOT / "results/stage11_mvtecad2_multicategory/stage11_b_patchcore_multicategory_status.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_ANALYSIS = OUT_DIR / "stage11_b1_patchcore_detector_quality_analysis.csv"
OUT_REPORT = DOC_DIR / "stage11_b1_patchcore_detector_quality_analysis.md"


def assign_priority(row: pd.Series) -> tuple[str, str]:
    image_auroc = float(row["image_AUROC"])
    pixel_auroc = float(row["pixel_AUROC"])
    pixel_f1 = float(row["pixel_F1Score"])

    if image_auroc >= 0.75 and pixel_auroc >= 0.85 and pixel_f1 >= 0.30:
        return (
            "primary",
            "Detector quality is sufficient for localization-guided crop/VLM reasoning.",
        )

    if image_auroc >= 0.70 and pixel_auroc >= 0.75:
        return (
            "secondary",
            "Image-level detection is acceptable, but localization quality should be treated cautiously.",
        )

    if image_auroc < 0.60:
        return (
            "detector_risk",
            "PatchCore image-level detection is weak; VLM crop results may mainly reflect detector failure.",
        )

    return (
        "diagnostic",
        "Mixed detector quality; keep for diagnosis rather than main claim.",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(IN_METRICS)
    status = pd.read_csv(IN_STATUS)

    df = metrics.merge(
        status[["category", "success", "num_prediction_rows", "elapsed_sec", "error"]],
        on="category",
        how="left",
    )

    priority_rows = []
    for _, row in df.iterrows():
        group, interpretation = assign_priority(row)
        priority_rows.append((group, interpretation))

    df["stage11_c_priority_group"] = [x[0] for x in priority_rows]
    df["stage11_c_interpretation"] = [x[1] for x in priority_rows]

    df = df.sort_values(
        ["stage11_c_priority_group", "image_AUROC", "pixel_F1Score"],
        ascending=[True, False, False],
    )

    df.to_csv(OUT_ANALYSIS, index=False)

    primary = df[df["stage11_c_priority_group"] == "primary"]["category"].tolist()
    secondary = df[df["stage11_c_priority_group"] == "secondary"]["category"].tolist()
    risk = df[df["stage11_c_priority_group"] == "detector_risk"]["category"].tolist()
    diagnostic = df[df["stage11_c_priority_group"] == "diagnostic"]["category"].tolist()

    lines = []
    lines.append("# Stage 11-B1 PatchCore Detector Quality Analysis")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This analysis summarizes the Stage 11-B multi-category PatchCore baseline and decides which MVTec AD 2 categories are suitable for Stage 11-C candidate-region generation and Stage 11-D VLM reasoning.")
    lines.append("")
    lines.append("This step does not train PatchCore, run VLM inference, or generate candidate crops.")
    lines.append("")
    lines.append("## 2. Decision Rule")
    lines.append("")
    lines.append("| Group | Rule | Usage |")
    lines.append("|---|---|---|")
    lines.append("| primary | image AUROC >= 0.75, pixel AUROC >= 0.85, pixel F1 >= 0.30 | Main evidence for context-aware crop reasoning |")
    lines.append("| secondary | image AUROC >= 0.70 and pixel AUROC >= 0.75 | Usable, but localization needs caution |")
    lines.append("| detector_risk | image AUROC < 0.60 | Do not use as main VLM claim; detector may dominate failure |")
    lines.append("| diagnostic | mixed metrics | Use only for qualitative or failure analysis |")
    lines.append("")
    lines.append("## 3. Category Ranking")
    lines.append("")
    lines.append("| Category | image AUROC | image F1 | pixel AUROC | pixel F1 | Group | Interpretation |")
    lines.append("|---|---:|---:|---:|---:|---|---|")

    for _, r in df.sort_values("image_AUROC", ascending=False).iterrows():
        lines.append(
            f"| {r['category']} | {float(r['image_AUROC']):.4f} | "
            f"{float(r['image_F1Score']):.4f} | {float(r['pixel_AUROC']):.4f} | "
            f"{float(r['pixel_F1Score']):.4f} | {r['stage11_c_priority_group']} | "
            f"{r['stage11_c_interpretation']} |"
        )

    lines.append("")
    lines.append("## 4. Stage 11-C Category Plan")
    lines.append("")
    lines.append(f"- Primary categories: `{', '.join(primary) if primary else 'none'}`")
    lines.append(f"- Secondary categories: `{', '.join(secondary) if secondary else 'none'}`")
    lines.append(f"- Diagnostic categories: `{', '.join(diagnostic) if diagnostic else 'none'}`")
    lines.append(f"- Detector-risk categories: `{', '.join(risk) if risk else 'none'}`")
    lines.append("")
    lines.append("Recommended Stage 11-C execution plan:")
    lines.append("")
    lines.append("1. First run candidate-region generation on the primary categories.")
    lines.append("2. Then include secondary categories if runtime is acceptable.")
    lines.append("3. Keep detector-risk categories for failure analysis, not for the main paper claim.")
    lines.append("")
    lines.append("## 5. Paper-level Interpretation")
    lines.append("")
    lines.append("The multi-category results show that the proposed VLM reasoning branch should be evaluated conditionally on detector quality.")
    lines.append("If PatchCore localization is weak, crop-based VLM reasoning may fail because the crop is not a reliable visual bridge.")
    lines.append("Therefore, the next module should report both detector quality and VLM reasoning quality instead of treating all categories as equally valid evidence.")
    lines.append("")
    lines.append("## 6. Output")
    lines.append("")
    lines.append(f"- Analysis CSV: `{OUT_ANALYSIS.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("[DONE]", OUT_ANALYSIS)
    print("[DONE]", OUT_REPORT)
    print("")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
