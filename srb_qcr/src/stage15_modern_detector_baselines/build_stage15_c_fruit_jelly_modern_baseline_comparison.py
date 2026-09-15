from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

DOC_DIR = ROOT / "docs/stage15_modern_detector_baselines"
RES_DIR = ROOT / "results/stage15_modern_detector_baselines"

STAGE15_B = RES_DIR / "stage15_b_efficientad_fruit_jelly_metrics.csv"

STAGE14_C3 = ROOT / "results/stage14_strong_vlm_baselines/stage14_c3_fruit_jelly_external_baseline_comparison.csv"
STAGE14_D = ROOT / "results/stage14_strong_vlm_baselines/stage14_d_winclip_fruit_jelly_sensitivity.csv"
STAGE13_LOCO = ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_loco_category.csv"
STAGE13_SAME = ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv"
STAGE11_MAIN = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv"

OUT_CSV = RES_DIR / "stage15_c_fruit_jelly_modern_baseline_comparison.csv"
OUT_REPORT = DOC_DIR / "stage15_c_fruit_jelly_modern_baseline_comparison.md"


def f4(x):
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def main():
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    eff = pd.read_csv(STAGE15_B).iloc[0]
    c3 = pd.read_csv(STAGE14_C3)
    d = pd.read_csv(STAGE14_D)
    main = pd.read_csv(STAGE11_MAIN)
    loco = pd.read_csv(STAGE13_LOCO)
    same = pd.read_csv(STAGE13_SAME)

    category = "fruit_jelly"

    main_row = main[main["category_or_scope"] == category].iloc[0]

    winclip_fixed = c3[c3["method"] == "WinCLIP zero-shot"].iloc[0]

    d_success = d[d["status"] == "success"].copy()
    d_success["image_AUROC_num"] = pd.to_numeric(d_success["image_AUROC"], errors="coerce")
    winclip_best = d_success.sort_values("image_AUROC_num", ascending=False).iloc[0]

    loco_context = loco[
        (loco["category"] == category)
        & (loco["fusion_pair"] == "patchcore_plus_context")
    ].iloc[0]

    same_context = same[
        (same["category"] == category)
        & (same["fusion_pair"] == "patchcore_plus_context")
    ].iloc[0]

    rows = [
        {
            "category": category,
            "method_group": "external_vlm_baseline",
            "method": "WinCLIP zero-shot",
            "image_auroc": float(winclip_fixed["auroc"]),
            "image_f1": "",
            "pixel_auroc": float(winclip_fixed["pixel_auroc"]),
            "pixel_f1": float(winclip_fixed["pixel_f1"]),
            "protocol": "Stage 14-C2 zero-shot pilot",
            "interpretation": "External VLM anomaly detection baseline under default zero-shot setting.",
        },
        {
            "category": category,
            "method_group": "external_vlm_baseline",
            "method": "WinCLIP sensitivity best",
            "image_auroc": float(winclip_best["image_AUROC"]),
            "image_f1": float(winclip_best["image_F1Score"]),
            "pixel_auroc": float(winclip_best["pixel_AUROC"]),
            "pixel_f1": float(winclip_best["pixel_F1Score"]),
            "protocol": f"class_name={winclip_best['class_name']}, k_shot={winclip_best['k_shot']}, scales={winclip_best['scales']}",
            "interpretation": "Best fruit_jelly WinCLIP setting from sensitivity search.",
        },
        {
            "category": category,
            "method_group": "modern_detector_baseline",
            "method": "EfficientAD pilot",
            "image_auroc": float(eff["image_AUROC"]),
            "image_f1": float(eff["image_F1Score"]),
            "pixel_auroc": float(eff["pixel_AUROC"]),
            "pixel_f1": float(eff["pixel_F1Score"]),
            "protocol": "model_size=small, max_epochs=20, train_batch_size=1",
            "interpretation": "Modern non-VLM detector pilot; fit and test succeeded.",
        },
        {
            "category": category,
            "method_group": "classical_detector",
            "method": "PatchCore",
            "image_auroc": float(main_row["patchcore_reference_auroc"]),
            "image_f1": "",
            "pixel_auroc": "",
            "pixel_f1": "",
            "protocol": "Stage 11 reference",
            "interpretation": "Classic anomaly detector reference.",
        },
        {
            "category": category,
            "method_group": "vlm_branch",
            "method": "full-image VLM",
            "image_auroc": float(main_row["full_image_auroc"]),
            "image_f1": "",
            "pixel_auroc": "",
            "pixel_f1": "",
            "protocol": "Stage 11 full-image baseline",
            "interpretation": "Direct full-image VLM scoring.",
        },
        {
            "category": category,
            "method_group": "vlm_branch",
            "method": "context-aware VLM",
            "image_auroc": float(main_row["reported_method_auroc"]),
            "image_f1": "",
            "pixel_auroc": "",
            "pixel_f1": "",
            "protocol": "Stage 11 context-aware VLM",
            "interpretation": "Our context-aware VLM branch.",
        },
        {
            "category": category,
            "method_group": "fusion_loco",
            "method": "PatchCore + context VLM, LOCO",
            "image_auroc": float(loco_context["auroc"]),
            "image_f1": float(loco_context["best_f1"]),
            "pixel_auroc": "",
            "pixel_f1": "",
            "protocol": "Stage 13 leave-one-category-out fusion",
            "interpretation": "Conservative fusion result with alpha selected on other categories.",
        },
        {
            "category": category,
            "method_group": "fusion_same_set",
            "method": "PatchCore + context VLM, same-set",
            "image_auroc": float(same_context["auroc"]),
            "image_f1": float(same_context["best_f1"]),
            "pixel_auroc": "",
            "pixel_f1": "",
            "protocol": "Stage 13 same-set fusion upper-bound",
            "interpretation": "Upper-bound diagnostic; should not be overclaimed as fair final protocol.",
        },
    ]

    out = pd.DataFrame(rows)
    out = out.sort_values("image_auroc", ascending=False)
    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# Stage 15-C fruit_jelly Modern Baseline Comparison")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report compares EfficientAD, WinCLIP, PatchCore, context-aware VLM, and PatchCore+context fusion on AD2 fruit_jelly.")
    lines.append("")
    lines.append("EfficientAD is included as a modern non-VLM detector baseline, while WinCLIP is included as an external VLM anomaly detection baseline.")
    lines.append("")
    lines.append("## 2. Result Table")
    lines.append("")
    lines.append("| Rank | Method | Group | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Protocol |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")

    for idx, (_, r) in enumerate(out.iterrows(), start=1):
        lines.append(
            f"| {idx} | {r['method']} | {r['method_group']} | {f4(r['image_auroc'])} | "
            f"{f4(r['image_f1'])} | {f4(r['pixel_auroc'])} | {f4(r['pixel_f1'])} | {r['protocol']} |"
        )

    lines.append("")
    lines.append("## 3. Main Observations")
    lines.append("")

    eff_auroc = float(eff["image_AUROC"])
    patch_auroc = float(main_row["patchcore_reference_auroc"])
    context_auroc = float(main_row["reported_method_auroc"])
    loco_auroc = float(loco_context["auroc"])
    win_best_auroc = float(winclip_best["image_AUROC"])

    lines.append(f"1. EfficientAD pilot reaches image AUROC `{f4(eff_auroc)}` on fruit_jelly.")
    lines.append(f"2. EfficientAD is higher than the best tested WinCLIP fruit_jelly setting `{f4(win_best_auroc)}`.")
    lines.append(f"3. EfficientAD is lower than PatchCore `{f4(patch_auroc)}`.")
    lines.append(f"4. EfficientAD is lower than context-aware VLM `{f4(context_auroc)}` and PatchCore+context LOCO fusion `{f4(loco_auroc)}`.")
    lines.append(f"5. EfficientAD has pixel F1 `{f4(float(eff['pixel_F1Score']))}`, which is useful for localization discussion but not directly comparable to image-only VLM scores.")
    lines.append("")
    lines.append("## 4. Safe Interpretation")
    lines.append("")
    lines.append("The EfficientAD fruit_jelly pilot verifies that the modern detector baseline can run in the current environment.")
    lines.append("")
    lines.append("However, because this is a 20-epoch one-category pilot, it should not yet be used as the final EfficientAD baseline.")
    lines.append("")
    lines.append("The next stage should run a fixed EfficientAD protocol on the four AD2 primary categories, or first increase the fruit_jelly training budget if the goal is to obtain a stronger formal detector baseline.")
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
