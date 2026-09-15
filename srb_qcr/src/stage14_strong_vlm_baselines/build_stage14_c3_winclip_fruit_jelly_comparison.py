from pathlib import Path
import json
import pandas as pd


ROOT = Path(".").resolve()

DOC_DIR = ROOT / "docs/stage14_strong_vlm_baselines"
RES_DIR = ROOT / "results/stage14_strong_vlm_baselines"

WINCLIP_CSV = RES_DIR / "stage14_c2_winclip_fruit_jelly_metrics.csv"
STAGE11_MAIN = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv"
STAGE13_PER_CAT = ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv"
STAGE13_LOCO = ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_loco_category.csv"

OUT_CSV = RES_DIR / "stage14_c3_fruit_jelly_external_baseline_comparison.csv"
OUT_REPORT = DOC_DIR / "stage14_c3_fruit_jelly_external_baseline_comparison.md"


def f4(x):
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def main():
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    if not WINCLIP_CSV.exists():
        raise FileNotFoundError(WINCLIP_CSV)

    win = pd.read_csv(WINCLIP_CSV).iloc[0]
    main_table = pd.read_csv(STAGE11_MAIN)
    per_cat = pd.read_csv(STAGE13_PER_CAT)
    loco = pd.read_csv(STAGE13_LOCO)

    fruit = main_table[main_table["category_or_scope"] == "fruit_jelly"].iloc[0]

    same_context = per_cat[
        (per_cat["category"] == "fruit_jelly")
        & (per_cat["fusion_pair"] == "patchcore_plus_context")
    ].iloc[0]

    loco_context = loco[
        (loco["category"] == "fruit_jelly")
        & (loco["fusion_pair"] == "patchcore_plus_context")
    ].iloc[0]

    rows = [
        {
            "category": "fruit_jelly",
            "method_group": "external_vlm_baseline",
            "method": "WinCLIP zero-shot",
            "auroc": float(win["image_AUROC"]),
            "ap": "",
            "pixel_auroc": float(win["pixel_AUROC"]),
            "pixel_f1": float(win["pixel_F1Score"]),
            "source": "stage14_c2_winclip_fruit_jelly_metrics.csv",
            "interpretation": "External VLM anomaly detection baseline under default zero-shot setting.",
        },
        {
            "category": "fruit_jelly",
            "method_group": "classical_detector",
            "method": "PatchCore",
            "auroc": float(fruit["patchcore_reference_auroc"]),
            "ap": "",
            "pixel_auroc": "",
            "pixel_f1": "",
            "source": "stage11_i_paper_ready_main_table.csv",
            "interpretation": "Classical detector reference.",
        },
        {
            "category": "fruit_jelly",
            "method_group": "vlm_branch",
            "method": "full-image VLM",
            "auroc": float(fruit["full_image_auroc"]),
            "ap": "",
            "pixel_auroc": "",
            "pixel_f1": "",
            "source": "stage11_i_paper_ready_main_table.csv",
            "interpretation": "Full-image VLM baseline.",
        },
        {
            "category": "fruit_jelly",
            "method_group": "vlm_branch",
            "method": "context-aware VLM",
            "auroc": float(fruit["reported_method_auroc"]),
            "ap": "",
            "pixel_auroc": "",
            "pixel_f1": "",
            "source": "stage11_i_paper_ready_main_table.csv",
            "interpretation": "Our context-aware VLM branch before detector fusion.",
        },
        {
            "category": "fruit_jelly",
            "method_group": "fusion_same_set",
            "method": "PatchCore + context VLM, same-set",
            "auroc": float(same_context["auroc"]),
            "ap": float(same_context["ap"]),
            "pixel_auroc": "",
            "pixel_f1": "",
            "source": "stage13_a_patchcore_vlm_fusion_per_category.csv",
            "interpretation": "Upper-bound same-set fusion diagnostic.",
        },
        {
            "category": "fruit_jelly",
            "method_group": "fusion_loco",
            "method": "PatchCore + context VLM, leave-one-category-out",
            "auroc": float(loco_context["auroc"]),
            "ap": float(loco_context["ap"]),
            "pixel_auroc": "",
            "pixel_f1": "",
            "source": "stage13_a_patchcore_vlm_fusion_loco_category.csv",
            "interpretation": "More conservative fusion result using weights selected on other categories.",
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# Stage 14-C3 fruit_jelly External Baseline Comparison")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report compares the WinCLIP fruit_jelly pilot with PatchCore, the context-aware VLM branch, and PatchCore+context fusion.")
    lines.append("")
    lines.append("## 2. Comparison Table")
    lines.append("")
    lines.append("| Method | Group | AUROC | AP | Pixel AUROC | Pixel F1 | Interpretation |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for _, r in out.iterrows():
        lines.append(
            f"| {r['method']} | {r['method_group']} | {f4(r['auroc'])} | "
            f"{f4(r['ap'])} | {f4(r['pixel_auroc'])} | {f4(r['pixel_f1'])} | {r['interpretation']} |"
        )

    lines.append("")
    lines.append("## 3. Main Observation")
    lines.append("")
    lines.append("The default zero-shot WinCLIP pilot is weak on AD2 fruit_jelly, with image-level AUROC below 0.5.")
    lines.append("")
    lines.append("This result should not yet be used to claim that our method generally outperforms WinCLIP, because only one category and one zero-shot configuration have been tested.")
    lines.append("")
    lines.append("However, it does show that simply introducing a known VLM anomaly baseline is not automatically stronger on the AD2 setting.")
    lines.append("")
    lines.append("## 4. Next Decision")
    lines.append("")
    lines.append("Before expanding to all primary categories, the next step should test whether WinCLIP few-shot settings or class-name variants improve the fruit_jelly result.")
    lines.append("")
    lines.append("Recommended next step: Stage 14-D should run WinCLIP k-shot/class-name sensitivity on fruit_jelly.")
    lines.append("")
    lines.append("## 5. Output")
    lines.append("")
    lines.append(f"- Comparison CSV: `{OUT_CSV.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_REPORT)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
