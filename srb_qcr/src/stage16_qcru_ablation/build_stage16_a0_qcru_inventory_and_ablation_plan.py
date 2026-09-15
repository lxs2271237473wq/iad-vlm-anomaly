from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_INVENTORY = OUT_DIR / "stage16_a0_qcru_source_inventory.csv"
OUT_PLAN = OUT_DIR / "stage16_a0_qcru_ablation_plan.csv"
OUT_DOC = DOC_DIR / "stage16_a0_qcru_inventory_and_ablation_plan.md"

SOURCE_PATHS = [
    "results/stage9_qcr_u/stage9_a0_input_structure.csv",
    "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_predictions.csv",
    "results/stage9_qcr_u/stage9_a1_qcr_u_fusion_summary.csv",
    "results/stage9_qcr_u/stage9_a2_qcr_u_macro_summary.csv",
    "results/stage9_qcr_u/stage9_a2_qcr_u_per_category.csv",
    "results/stage9_qcr_u/stage9_a2_qcr_u_signal_diagnostics.csv",
    "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_predictions.csv",
    "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_summary.csv",
    "results/stage9_qcr_u/stage9_a3_qcr_u_debiased_per_category.csv",
    "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_grid.csv",
    "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_summary.csv",
    "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv",
    "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv",
    "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv",
    "results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv",
]


def inspect_csv(path: Path) -> dict:
    item = {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "rows": "",
        "cols": "",
        "columns": "",
        "read_status": "missing",
        "note": "",
    }

    if not path.exists():
        return item

    try:
        df = pd.read_csv(path)
        item["rows"] = len(df)
        item["cols"] = len(df.columns)
        item["columns"] = ";".join(map(str, df.columns))
        item["read_status"] = "ok"

        if len(df) == 0:
            item["note"] = "empty_dataframe"
        elif len(df.columns) <= 1:
            item["note"] = "possibly_malformed_or_one_column"
        else:
            item["note"] = "usable"

    except Exception as exc:
        item["read_status"] = "failed"
        item["note"] = repr(exc)

    return item


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    inventory_rows = [inspect_csv(ROOT / p) for p in SOURCE_PATHS]
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(OUT_INVENTORY, index=False, lineterminator="\n")

    plan_rows = [
        {
            "variant_id": "V0",
            "variant": "detector_only",
            "uses_detector_score": True,
            "uses_full_image_vlm": False,
            "uses_crop_vlm": False,
            "uses_quality": False,
            "uses_consistency": False,
            "uses_unknown": False,
            "purpose": "Anchor baseline; proves whether QCR-U beats the detector alone.",
        },
        {
            "variant_id": "V1",
            "variant": "full_image_vlm",
            "uses_detector_score": False,
            "uses_full_image_vlm": True,
            "uses_crop_vlm": False,
            "uses_quality": False,
            "uses_consistency": False,
            "uses_unknown": False,
            "purpose": "Weak VLM sanity baseline; should not be the main comparison target.",
        },
        {
            "variant_id": "V2",
            "variant": "crop_topk_vlm",
            "uses_detector_score": False,
            "uses_full_image_vlm": False,
            "uses_crop_vlm": True,
            "uses_quality": False,
            "uses_consistency": False,
            "uses_unknown": False,
            "purpose": "Tests whether localization-guided crops improve VLM scoring.",
        },
        {
            "variant_id": "V3",
            "variant": "naive_detector_crop_fusion",
            "uses_detector_score": True,
            "uses_full_image_vlm": False,
            "uses_crop_vlm": True,
            "uses_quality": False,
            "uses_consistency": False,
            "uses_unknown": False,
            "purpose": "Naive fusion baseline; QCR-U must beat this or the method is not justified.",
        },
        {
            "variant_id": "V4",
            "variant": "quality_weighted_crop",
            "uses_detector_score": True,
            "uses_full_image_vlm": False,
            "uses_crop_vlm": True,
            "uses_quality": True,
            "uses_consistency": False,
            "uses_unknown": False,
            "purpose": "Tests whether candidate quality contributes beyond crop scoring.",
        },
        {
            "variant_id": "V5",
            "variant": "quality_consistency_fusion",
            "uses_detector_score": True,
            "uses_full_image_vlm": False,
            "uses_crop_vlm": True,
            "uses_quality": True,
            "uses_consistency": True,
            "uses_unknown": False,
            "purpose": "Core QCR-U binary anomaly recognition variant.",
        },
        {
            "variant_id": "V6",
            "variant": "qcr_u_full_optional_unknown",
            "uses_detector_score": True,
            "uses_full_image_vlm": False,
            "uses_crop_vlm": True,
            "uses_quality": True,
            "uses_consistency": True,
            "uses_unknown": True,
            "purpose": "Only valid if a strict known/unknown protocol is available.",
        },
    ]

    plan = pd.DataFrame(plan_rows)
    plan.to_csv(OUT_PLAN, index=False, lineterminator="\n")

    usable = inventory[inventory["note"] == "usable"]
    missing = inventory[inventory["exists"] == False]
    malformed = inventory[
        (inventory["exists"] == True)
        & (inventory["note"].isin(["possibly_malformed_or_one_column", "empty_dataframe"]))
    ]

    lines = []
    lines += [
        "# Stage 16-A0 QCR-U 输入审计与消融计划",
        "",
        "## 1. 本阶段目的",
        "",
        "Stage 15 已经完成强基线结论锁定。下一阶段进入 QCR-U，但不能直接写新方法或乱调融合权重。",
        "",
        "本阶段只做三件事：",
        "",
        "1. 审计 Stage 9 / Stage 13 / Stage 15 已有结果文件。",
        "2. 判断哪些文件可以作为 QCR-U ablation 的输入。",
        "3. 锁定 QCR-U 消融变量，避免把 heuristic fusion 包装成方法。",
        "",
        "## 2. 输入文件审计结果",
        "",
        f"- total_sources_checked: `{len(inventory)}`",
        f"- usable_sources: `{len(usable)}`",
        f"- missing_sources: `{len(missing)}`",
        f"- malformed_or_empty_sources: `{len(malformed)}`",
        "",
        "完整审计表见：",
        "",
        f"`{OUT_INVENTORY.relative_to(ROOT)}`",
        "",
        "## 3. 必须解决的硬问题",
        "",
        "QCR-U 不能只是：",
        "",
        "```text",
        "score = alpha * detector + beta * vlm",
        "```",
        "",
        "它必须至少证明：",
        "",
        "1. candidate quality 是否有效。",
        "2. detector-VLM consistency 是否有效。",
        "3. QCR-U 是否稳定优于 naive fusion。",
        "4. 参数是否来自固定协议，而不是 test-set 调参。",
        "",
        "## 4. QCR-U 消融计划",
        "",
        "| Variant | Detector | Crop VLM | Quality | Consistency | Unknown | Purpose |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for _, r in plan.iterrows():
        lines.append(
            f"| {r['variant']} | "
            f"{int(bool(r['uses_detector_score']))} | "
            f"{int(bool(r['uses_crop_vlm']))} | "
            f"{int(bool(r['uses_quality']))} | "
            f"{int(bool(r['uses_consistency']))} | "
            f"{int(bool(r['uses_unknown']))} | "
            f"{r['purpose']} |"
        )

    lines += [
        "",
        "## 5. 下一步决策",
        "",
        "如果 Stage 9 的旧 QCR-U 文件已经包含足够字段，下一步进入：",
        "",
        "```text",
        "Stage 16-A1: QCR-U fixed-protocol ablation implementation",
        "```",
        "",
        "如果字段不够，先补：",
        "",
        "```text",
        "Stage 16-A1-input: 构建统一 prediction table",
        "```",
        "",
        "统一 prediction table 至少要包含：",
        "",
        "- category",
        "- image_id 或 image_path",
        "- gt_binary",
        "- detector_score",
        "- full_image_vlm_score",
        "- crop_topk_vlm_score",
        "- candidate_quality",
        "- consistency_score",
        "- candidate_count",
        "",
        "## 6. 本阶段输出",
        "",
        f"- `{OUT_INVENTORY.relative_to(ROOT)}`",
        f"- `{OUT_PLAN.relative_to(ROOT)}`",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_INVENTORY)
    print("[DONE]", OUT_PLAN)
    print("[DONE]", OUT_DOC)
    print()
    print("===== inventory =====")
    print(inventory.to_string(index=False))
    print()
    print("===== ablation plan =====")
    print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
