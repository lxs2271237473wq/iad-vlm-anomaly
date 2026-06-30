from __future__ import annotations

from pathlib import Path
import json
import traceback
from datetime import datetime

import pandas as pd


ROOT = Path(".").resolve()

OUT_DIR = ROOT / "results/stage14_strong_vlm_baselines"
DOC_DIR = ROOT / "docs/stage14_strong_vlm_baselines"

OUT_CSV = OUT_DIR / "stage14_e_winclip_primary_fixed_protocol.csv"
OUT_JSON = OUT_DIR / "stage14_e_winclip_primary_fixed_protocol_raw.json"
OUT_REPORT = DOC_DIR / "stage14_e_winclip_primary_fixed_protocol_report.md"
OUT_ERROR = OUT_DIR / "stage14_e_winclip_primary_fixed_protocol_errors.txt"

STAGE11_MAIN = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv"
STAGE13_LOCO = ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_loco_category.csv"
STAGE13_PER_CAT = ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv"

ENGINE_ROOT = ROOT / "runs/stage14_strong_vlm_baselines/winclip_primary_fixed_protocol"

CATEGORIES = [
    {
        "category": "fruit_jelly",
        "class_name": "jelly",
        "data_root": ROOT / "datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder",
    },
    {
        "category": "sheet_metal",
        "class_name": "sheet metal",
        "data_root": ROOT / "datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder",
    },
    {
        "category": "vial",
        "class_name": "vial",
        "data_root": ROOT / "datasets/MVTec_AD_2_anomalib_all/vial_folder",
    },
    {
        "category": "walnuts",
        "class_name": "walnut",
        "data_root": ROOT / "datasets/MVTec_AD_2_anomalib_all/walnuts_folder",
    },
]

K_SHOT = 1
SCALES = (1, 2, 3)


def f4(x):
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def flatten_metrics(metrics):
    if isinstance(metrics, list):
        if len(metrics) == 0:
            return {}
        if isinstance(metrics[0], dict):
            return dict(metrics[0])
        return {"metrics": str(metrics)}

    if isinstance(metrics, dict):
        return dict(metrics)

    return {"metrics": str(metrics)}


def run_one_category(category: str, class_name: str, data_root: Path):
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import WinClip

    required_paths = [
        data_root / "train/good",
        data_root / "test/good",
        data_root / "test/bad",
        data_root / "ground_truth/bad",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset paths:\n" + "\n".join(missing))

    run_id = f"{category}_k{K_SHOT}_scales_" + "_".join(str(x) for x in SCALES)

    datamodule = Folder(
        name=f"ad2_{category}_winclip_fixed",
        root=str(data_root),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/bad",
        mask_dir="ground_truth/bad",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = WinClip(
        class_name=class_name,
        k_shot=K_SHOT,
        scales=SCALES,
    )

    engine = Engine(
        default_root_dir=ENGINE_ROOT / run_id,
        logger=False,
    )

    metrics = engine.test(model=model, datamodule=datamodule)
    return metrics


def build_comparison_table(winclip_df: pd.DataFrame) -> pd.DataFrame:
    main = pd.read_csv(STAGE11_MAIN)
    loco = pd.read_csv(STAGE13_LOCO)
    same = pd.read_csv(STAGE13_PER_CAT)

    rows = []

    for _, w in winclip_df.iterrows():
        category = w["category"]

        main_row = main[main["category_or_scope"] == category].iloc[0]

        loco_context = loco[
            (loco["category"] == category)
            & (loco["fusion_pair"] == "patchcore_plus_context")
        ].iloc[0]

        same_context = same[
            (same["category"] == category)
            & (same["fusion_pair"] == "patchcore_plus_context")
        ].iloc[0]

        rows += [
            {
                "category": category,
                "method_group": "external_vlm_baseline",
                "method": "WinCLIP fixed protocol",
                "auroc": w["image_AUROC"],
                "ap": "",
                "pixel_auroc": w["pixel_AUROC"],
                "pixel_f1": w["pixel_F1Score"],
                "note": f"class_name={w['class_name']}, k_shot={K_SHOT}, scales={SCALES}",
            },
            {
                "category": category,
                "method_group": "classical_detector",
                "method": "PatchCore",
                "auroc": main_row["patchcore_reference_auroc"],
                "ap": "",
                "pixel_auroc": "",
                "pixel_f1": "",
                "note": "Stage 11 reference",
            },
            {
                "category": category,
                "method_group": "vlm_branch",
                "method": "full-image VLM",
                "auroc": main_row["full_image_auroc"],
                "ap": "",
                "pixel_auroc": "",
                "pixel_f1": "",
                "note": "Stage 11 reference",
            },
            {
                "category": category,
                "method_group": "vlm_branch",
                "method": "context-aware VLM",
                "auroc": main_row["reported_method_auroc"],
                "ap": "",
                "pixel_auroc": "",
                "pixel_f1": "",
                "note": "Stage 11 reported context method",
            },
            {
                "category": category,
                "method_group": "fusion_loco",
                "method": "PatchCore + context VLM, LOCO",
                "auroc": loco_context["auroc"],
                "ap": loco_context["ap"],
                "pixel_auroc": "",
                "pixel_f1": "",
                "note": "Stage 13 leave-one-category-out fusion",
            },
            {
                "category": category,
                "method_group": "fusion_same_set",
                "method": "PatchCore + context VLM, same-set",
                "auroc": same_context["auroc"],
                "ap": same_context["ap"],
                "pixel_auroc": "",
                "pixel_f1": "",
                "note": "Stage 13 same-set upper-bound fusion",
            },
        ]

    return pd.DataFrame(rows)


def write_report(winclip_df: pd.DataFrame, comparison: pd.DataFrame, errors: list[str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Stage 14-E WinCLIP Primary Fixed Protocol")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage evaluates WinCLIP on all AD2 primary categories using one fixed protocol.")
    lines.append("")
    lines.append("This avoids claiming a per-category over-tuned WinCLIP result.")
    lines.append("")
    lines.append("## 2. Fixed Protocol")
    lines.append("")
    lines.append(f"- k-shot: `{K_SHOT}`")
    lines.append(f"- scales: `{SCALES}`")
    lines.append("")
    lines.append("| Category | class_name |")
    lines.append("|---|---|")
    for item in CATEGORIES:
        lines.append(f"| {item['category']} | `{item['class_name']}` |")

    lines.append("")
    lines.append("## 3. WinCLIP Results")
    lines.append("")
    lines.append("| Category | Status | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for _, r in winclip_df.iterrows():
        err = str(r.get("error", ""))
        if len(err) > 120:
            err = err[:120] + "..."
        lines.append(
            f"| {r['category']} | {r['status']} | {f4(r.get('image_AUROC', ''))} | "
            f"{f4(r.get('image_F1Score', ''))} | {f4(r.get('pixel_AUROC', ''))} | "
            f"{f4(r.get('pixel_F1Score', ''))} | `{err}` |"
        )

    lines.append("")
    lines.append("## 4. Unified Category-level Comparison")
    lines.append("")
    lines.append("| Category | Method | Group | AUROC | AP | Pixel AUROC | Pixel F1 |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    for _, r in comparison.iterrows():
        lines.append(
            f"| {r['category']} | {r['method']} | {r['method_group']} | "
            f"{f4(r['auroc'])} | {f4(r['ap'])} | {f4(r['pixel_auroc'])} | {f4(r['pixel_f1'])} |"
        )

    lines.append("")
    lines.append("## 5. Aggregate Observation")
    lines.append("")

    success = winclip_df[winclip_df["status"] == "success"].copy()
    if not success.empty:
        success["image_AUROC_num"] = pd.to_numeric(success["image_AUROC"], errors="coerce")
        lines.append(f"- Mean WinCLIP image AUROC over successful categories: `{f4(success['image_AUROC_num'].mean())}`")
        lines.append(f"- Best WinCLIP category AUROC: `{f4(success['image_AUROC_num'].max())}`")
        lines.append(f"- Worst WinCLIP category AUROC: `{f4(success['image_AUROC_num'].min())}`")
    else:
        lines.append("No successful WinCLIP category result.")

    lines.append("")
    lines.append("## 6. Decision")
    lines.append("")
    lines.append("If WinCLIP remains below PatchCore or PatchCore+context fusion on most categories, it should be reported as an external VLM anomaly detection baseline that is not directly robust under this AD2 fixed protocol.")
    lines.append("")
    lines.append("If WinCLIP outperforms our method on some categories, those categories should become failure-analysis cases rather than being ignored.")
    lines.append("")
    lines.append("## 7. Errors")
    lines.append("")
    if errors:
        lines.append("Some categories failed. See the error log.")
    else:
        lines.append("No category failed.")

    lines.append("")
    lines.append("## 8. Output")
    lines.append("")
    lines.append(f"- WinCLIP CSV: `{OUT_CSV.relative_to(ROOT)}`")
    lines.append(f"- Raw JSON: `{OUT_JSON.relative_to(ROOT)}`")
    lines.append(f"- Error log: `{OUT_ERROR.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    raw_records = []
    errors = []

    for item in CATEGORIES:
        category = item["category"]
        class_name = item["class_name"]
        data_root = item["data_root"]

        print(f"[RUN] category={category} class_name={class_name} k_shot={K_SHOT} scales={SCALES}")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "category": category,
            "class_name": class_name,
            "k_shot": K_SHOT,
            "scales": str(SCALES),
            "status": "failed",
            "image_AUROC": "",
            "image_F1Score": "",
            "pixel_AUROC": "",
            "pixel_F1Score": "",
            "error": "",
        }

        try:
            metrics = run_one_category(category, class_name, data_root)
            flat = flatten_metrics(metrics)

            row["status"] = "success"
            row["image_AUROC"] = flat.get("image_AUROC", "")
            row["image_F1Score"] = flat.get("image_F1Score", "")
            row["pixel_AUROC"] = flat.get("pixel_AUROC", "")
            row["pixel_F1Score"] = flat.get("pixel_F1Score", "")
            row["raw_metrics"] = str(flat)

            raw_records.append({
                "category": category,
                "class_name": class_name,
                "k_shot": K_SHOT,
                "scales": str(SCALES),
                "metrics": flat,
            })

            print("[OK]", flat)

        except Exception:
            err = traceback.format_exc()
            row["error"] = err.splitlines()[-1] if err else "unknown error"
            errors.append(f"\n===== {category} =====\n{err}")
            print("[ERROR]", row["error"])

        rows.append(row)

    winclip_df = pd.DataFrame(rows)
    comparison = build_comparison_table(winclip_df[winclip_df["status"] == "success"].copy())

    winclip_df.to_csv(OUT_CSV, index=False)

    comparison_csv = OUT_DIR / "stage14_e_primary_external_baseline_comparison.csv"
    comparison.to_csv(comparison_csv, index=False)

    OUT_JSON.write_text(
        json.dumps(raw_records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    OUT_ERROR.write_text("\n".join(errors), encoding="utf-8")

    write_report(winclip_df, comparison, errors)

    print("[DONE]", OUT_CSV)
    print("[DONE]", comparison_csv)
    print("[DONE]", OUT_JSON)
    print("[DONE]", OUT_ERROR)
    print("[DONE]", OUT_REPORT)
    print(winclip_df.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
