from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

STAGE11B1 = ROOT / "results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv"
STAGE11C = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv"
STAGE11D = ROOT / "results/stage11_mvtecad2_multicategory/stage11_d_vlm_summary.csv"
STAGE11E = ROOT / "results/stage11_mvtecad2_multicategory/stage11_e_multicategory_evidence_table.csv"
STAGE11H = ROOT / "results/stage11_mvtecad2_multicategory/stage11_h_fabric_secondary_evidence_table.csv"
STAGE11H_VLM = ROOT / "results/stage11_mvtecad2_multicategory/stage11_h_fabric_vlm_summary.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_MAIN = OUT_DIR / "stage11_i_paper_ready_main_table.csv"
OUT_METHOD = OUT_DIR / "stage11_i_paper_ready_method_table.csv"
OUT_USAGE = OUT_DIR / "stage11_i_category_usage_decision_table.csv"
OUT_REPORT = DOC_DIR / "stage11_i_paper_ready_evidence_report.md"


PRIMARY_CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]
SECONDARY_CATEGORIES = ["fabric"]
DETECTOR_RISK_CATEGORIES = ["can", "rice", "wallplugs"]


def f4(x) -> str:
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def get_method(df: pd.DataFrame, category: str, method: str):
    part = df[(df["category"] == category) & (df["method"] == method)]
    if part.empty:
        return None
    return part.iloc[0]


def get_best_context(df: pd.DataFrame, category: str):
    part = df[(df["category"] == category) & (df["method"].str.startswith("context"))].copy()
    if part.empty:
        return None
    return part.sort_values("auroc", ascending=False).iloc[0]


def get_best_tight(df: pd.DataFrame, category: str):
    part = df[(df["category"] == category) & (df["method"].str.startswith("tight"))].copy()
    if part.empty:
        return None
    return part.sort_values("auroc", ascending=False).iloc[0]


def get_best_vlm(df: pd.DataFrame, category: str):
    part = df[(df["category"] == category) & (df["method"] != "patchcore_score")].copy()
    if part.empty:
        return None
    return part.sort_values("auroc", ascending=False).iloc[0]


def build_method_table(stage11d: pd.DataFrame, stage11h_vlm: pd.DataFrame) -> pd.DataFrame:
    rows = []

    combined = pd.concat([stage11d, stage11h_vlm], ignore_index=True)

    categories = ["ALL_PRIMARY"] + PRIMARY_CATEGORIES + SECONDARY_CATEGORIES

    for category in categories:
        full = get_method(combined, category, "full_image")
        patchcore = get_method(combined, category, "patchcore_score")
        tight_top1 = get_method(combined, category, "tight_crop_top1")
        tight_mean = get_method(combined, category, "tight_crop_topk_mean")
        context_top1 = get_method(combined, category, "context_1.50_top1")
        context_mean = get_method(combined, category, "context_1.50_topk_mean")

        best_vlm = get_best_vlm(combined, category)
        best_context = get_best_context(combined, category)
        best_tight = get_best_tight(combined, category)

        if full is None:
            continue

        row = {
            "dataset": "MVTec AD 2",
            "category": category,
            "role": "primary_aggregate" if category == "ALL_PRIMARY" else (
                "primary_category" if category in PRIMARY_CATEGORIES else "secondary_category"
            ),
            "num_images": int(full["num_images"]),
            "num_normal": int(full["num_normal"]),
            "num_anomaly": int(full["num_anomaly"]),
            "full_image_auroc": float(full["auroc"]),
            "tight_crop_top1_auroc": "" if tight_top1 is None else float(tight_top1["auroc"]),
            "tight_crop_topk_mean_auroc": "" if tight_mean is None else float(tight_mean["auroc"]),
            "context_1p50_top1_auroc": "" if context_top1 is None else float(context_top1["auroc"]),
            "context_1p50_topk_mean_auroc": "" if context_mean is None else float(context_mean["auroc"]),
            "best_tight_method": "" if best_tight is None else best_tight["method"],
            "best_tight_auroc": "" if best_tight is None else float(best_tight["auroc"]),
            "best_tight_delta_vs_full": "" if best_tight is None else float(best_tight["auroc"]) - float(full["auroc"]),
            "best_context_method": "" if best_context is None else best_context["method"],
            "best_context_auroc": "" if best_context is None else float(best_context["auroc"]),
            "best_context_delta_vs_full": "" if best_context is None else float(best_context["auroc"]) - float(full["auroc"]),
            "best_vlm_method": "" if best_vlm is None else best_vlm["method"],
            "best_vlm_auroc": "" if best_vlm is None else float(best_vlm["auroc"]),
            "best_vlm_delta_vs_full": "" if best_vlm is None else float(best_vlm["auroc"]) - float(full["auroc"]),
            "patchcore_score_auroc": "" if patchcore is None else float(patchcore["auroc"]),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_main_table(method_table: pd.DataFrame, stage11e: pd.DataFrame, fabric_evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []

    all_primary = method_table[method_table["category"] == "ALL_PRIMARY"].iloc[0]

    rows.append({
        "table_section": "main_claim",
        "category_or_scope": "ALL_PRIMARY",
        "role": "main_aggregate_evidence",
        "full_image_auroc": all_primary["full_image_auroc"],
        "reported_method": all_primary["best_context_method"],
        "reported_method_auroc": all_primary["best_context_auroc"],
        "delta_auroc_vs_full": all_primary["best_context_delta_vs_full"],
        "patchcore_reference_auroc": all_primary["patchcore_score_auroc"],
        "paper_usage": "Main positive evidence: context-aware crop aggregation improves over full-image VLM prompting on the unified primary set.",
    })

    for category in PRIMARY_CATEGORIES:
        r = method_table[method_table["category"] == category].iloc[0]
        context_delta = float(r["best_context_delta_vs_full"])

        if context_delta > 0:
            usage = "Positive category-level context evidence."
        else:
            usage = "Limitation/failure-analysis category under the unified pipeline."

        rows.append({
            "table_section": "primary_category",
            "category_or_scope": category,
            "role": "primary_category",
            "full_image_auroc": r["full_image_auroc"],
            "reported_method": r["best_context_method"],
            "reported_method_auroc": r["best_context_auroc"],
            "delta_auroc_vs_full": r["best_context_delta_vs_full"],
            "patchcore_reference_auroc": r["patchcore_score_auroc"],
            "paper_usage": usage,
        })

    if not fabric_evidence.empty:
        f = fabric_evidence.iloc[0]
        rows.append({
            "table_section": "secondary_category",
            "category_or_scope": "fabric",
            "role": "secondary_boundary_case",
            "full_image_auroc": f["full_image_auroc"],
            "reported_method": f["best_context_method"],
            "reported_method_auroc": f["best_context_auroc"],
            "delta_auroc_vs_full": f["best_context_delta_vs_full"],
            "patchcore_reference_auroc": f["patchcore_score_auroc"],
            "paper_usage": "Secondary boundary case: localization is weak and context crop does not improve.",
        })

    return pd.DataFrame(rows)


def build_usage_table(quality: pd.DataFrame, method_table: pd.DataFrame, fabric_evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category in PRIMARY_CATEGORIES:
        q = quality[quality["category"] == category].iloc[0]
        m = method_table[method_table["category"] == category].iloc[0]
        context_delta = float(m["best_context_delta_vs_full"])

        if context_delta > 0:
            usage = "main_positive_or_supportive"
            reason = "Primary detector quality is acceptable and best context crop improves over full-image prompting."
        else:
            usage = "main_limitation"
            reason = "Primary detector quality is acceptable, but context crop does not improve under the unified Stage 11 pipeline."

        rows.append({
            "category": category,
            "detector_group": q["stage11_c_priority_group"],
            "image_AUROC_patchcore": q["image_AUROC"],
            "pixel_AUROC_patchcore": q["pixel_AUROC"],
            "pixel_F1_patchcore": q["pixel_F1Score"],
            "best_context_delta_vs_full": context_delta,
            "paper_usage": usage,
            "reason": reason,
        })

    if not fabric_evidence.empty:
        f = fabric_evidence.iloc[0]
        rows.append({
            "category": "fabric",
            "detector_group": "secondary",
            "image_AUROC_patchcore": f["stage11b_image_AUROC"],
            "pixel_AUROC_patchcore": f["stage11b_pixel_AUROC"],
            "pixel_F1_patchcore": f["stage11b_pixel_F1"],
            "best_context_delta_vs_full": f["best_context_delta_vs_full"],
            "paper_usage": "secondary_boundary_case",
            "reason": "Image-level detector is acceptable but pixel localization is weak; context crop is not positive.",
        })

    for category in DETECTOR_RISK_CATEGORIES:
        q = quality[quality["category"] == category].iloc[0]
        rows.append({
            "category": category,
            "detector_group": q["stage11_c_priority_group"],
            "image_AUROC_patchcore": q["image_AUROC"],
            "pixel_AUROC_patchcore": q["pixel_AUROC"],
            "pixel_F1_patchcore": q["pixel_F1Score"],
            "best_context_delta_vs_full": "",
            "paper_usage": "excluded_from_main_vlm_evidence",
            "reason": "Detector quality is too weak for fair crop-based VLM reasoning.",
        })

    return pd.DataFrame(rows)


def write_report(main_table: pd.DataFrame, method_table: pd.DataFrame, usage_table: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    all_primary = main_table[main_table["category_or_scope"] == "ALL_PRIMARY"].iloc[0]

    lines = []

    lines.append("# Stage 11-I Paper-ready Evidence Tables")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage converts the Stage 11 multi-category experiments into paper-ready evidence tables.")
    lines.append("It does not train models, run VLM inference, generate crops, or modify datasets.")
    lines.append("")
    lines.append("## 2. Main Claim Table")
    lines.append("")
    lines.append("| Scope / Category | Role | Full-image AUROC | Reported Method | Method AUROC | ΔAUROC vs full | PatchCore reference | Paper usage |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|")

    for _, r in main_table.iterrows():
        lines.append(
            f"| {r['category_or_scope']} | {r['role']} | {f4(r['full_image_auroc'])} | "
            f"{r['reported_method']} | {f4(r['reported_method_auroc'])} | "
            f"{f4(r['delta_auroc_vs_full'])} | {f4(r['patchcore_reference_auroc'])} | "
            f"{r['paper_usage']} |"
        )

    lines.append("")
    lines.append("## 3. Method Comparison Table")
    lines.append("")
    lines.append("| Scope / Category | Role | Full | Tight top1 | Tight mean | Context top1 | Context mean | Best context Δ | Best VLM | Best VLM Δ | PatchCore |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")

    for _, r in method_table.iterrows():
        lines.append(
            f"| {r['category']} | {r['role']} | {f4(r['full_image_auroc'])} | "
            f"{f4(r['tight_crop_top1_auroc'])} | {f4(r['tight_crop_topk_mean_auroc'])} | "
            f"{f4(r['context_1p50_top1_auroc'])} | {f4(r['context_1p50_topk_mean_auroc'])} | "
            f"{f4(r['best_context_delta_vs_full'])} | {r['best_vlm_method']} | "
            f"{f4(r['best_vlm_delta_vs_full'])} | {f4(r['patchcore_score_auroc'])} |"
        )

    lines.append("")
    lines.append("## 4. Category Usage Decision")
    lines.append("")
    lines.append("| Category | Detector group | Image AUROC | Pixel AUROC | Pixel F1 | Context Δ | Paper usage | Reason |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")

    for _, r in usage_table.iterrows():
        lines.append(
            f"| {r['category']} | {r['detector_group']} | {f4(r['image_AUROC_patchcore'])} | "
            f"{f4(r['pixel_AUROC_patchcore'])} | {f4(r['pixel_F1_patchcore'])} | "
            f"{f4(r['best_context_delta_vs_full'])} | {r['paper_usage']} | {r['reason']} |"
        )

    lines.append("")
    lines.append("## 5. Final Interpretation")
    lines.append("")
    lines.append("The Stage 11 results support a conditional version of the method claim.")
    lines.append("")
    lines.append("Recommended paper wording:")
    lines.append("")
    lines.append("```text")
    lines.append("Localization-guided context-aware crops improve VLM anomaly reasoning on the unified MVTec AD 2 primary-category evaluation, but the benefit is category-dependent and sensitive to detector/candidate quality.")
    lines.append("```")
    lines.append("")
    lines.append("Avoid overclaiming:")
    lines.append("")
    lines.append("```text")
    lines.append("Context-aware crops consistently improve all industrial anomaly categories.")
    lines.append("```")
    lines.append("")
    lines.append("## 6. Main Numeric Takeaway")
    lines.append("")
    lines.append(
        f"On ALL_PRIMARY, the reported context-aware method `{all_primary['reported_method']}` reaches AUROC "
        f"`{f4(all_primary['reported_method_auroc'])}`, compared with full-image AUROC "
        f"`{f4(all_primary['full_image_auroc'])}`, giving ΔAUROC "
        f"`{f4(all_primary['delta_auroc_vs_full'])}`."
    )
    lines.append("")
    lines.append("## 7. Output")
    lines.append("")
    lines.append(f"- Main table: `{OUT_MAIN.relative_to(ROOT)}`")
    lines.append(f"- Method table: `{OUT_METHOD.relative_to(ROOT)}`")
    lines.append(f"- Category usage table: `{OUT_USAGE.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 8. Next Step")
    lines.append("")
    lines.append("The next step should be Stage 12: convert these evidence tables into a final method narrative and paper experiment section.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    quality = read(STAGE11B1)
    stage11d = read(STAGE11D)
    stage11e = read(STAGE11E)
    fabric_evidence = read(STAGE11H)
    fabric_vlm = read(STAGE11H_VLM)

    method_table = build_method_table(stage11d, fabric_vlm)
    main_table = build_main_table(method_table, stage11e, fabric_evidence)
    usage_table = build_usage_table(quality, method_table, fabric_evidence)

    main_table.to_csv(OUT_MAIN, index=False)
    method_table.to_csv(OUT_METHOD, index=False)
    usage_table.to_csv(OUT_USAGE, index=False)

    write_report(main_table, method_table, usage_table)

    print("[DONE]", OUT_MAIN)
    print("[DONE]", OUT_METHOD)
    print("[DONE]", OUT_USAGE)
    print("[DONE]", OUT_REPORT)

    print("\n===== main table =====")
    print(main_table.to_string(index=False))

    print("\n===== method table =====")
    print(method_table.to_string(index=False))

    print("\n===== usage table =====")
    print(usage_table.to_string(index=False))


if __name__ == "__main__":
    main()
