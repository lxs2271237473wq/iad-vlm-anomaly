from __future__ import annotations

from pathlib import Path
import json
import traceback
from datetime import datetime

import pandas as pd


ROOT = Path(".").resolve()

OUT_DIR = ROOT / "results/stage15_modern_detector_baselines"
DOC_DIR = ROOT / "docs/stage15_modern_detector_baselines"

OUT_CSV = OUT_DIR / "stage15_d_efficientad_primary_fixed_protocol.csv"
OUT_JSON = OUT_DIR / "stage15_d_efficientad_primary_fixed_protocol_raw.json"
OUT_ERROR = OUT_DIR / "stage15_d_efficientad_primary_fixed_protocol_errors.txt"
OUT_COMPARISON = OUT_DIR / "stage15_d_primary_modern_detector_comparison.csv"
OUT_REPORT = DOC_DIR / "stage15_d_efficientad_primary_fixed_protocol_report.md"

STAGE11_MAIN = ROOT / "results/stage11_mvtecad2_multicategory/stage11_i_paper_ready_main_table.csv"
STAGE13_LOCO = ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_loco_category.csv"
STAGE13_SAME = ROOT / "results/stage13_strong_baseline/stage13_a_patchcore_vlm_fusion_per_category.csv"
STAGE14_E = ROOT / "results/stage14_strong_vlm_baselines/stage14_e_winclip_primary_fixed_protocol.csv"

IMAGENETTE_DIR = ROOT / "datasets/imagenette"
ENGINE_ROOT = ROOT / "runs/stage15_modern_detector_baselines/efficientad_primary_5000step_screening"

CATEGORIES = [
    {
        "category": "fruit_jelly",
        "data_root": ROOT / "datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder",
    },
    {
        "category": "sheet_metal",
        "data_root": ROOT / "datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder",
    },
    {
        "category": "vial",
        "data_root": ROOT / "datasets/MVTec_AD_2_anomalib_all/vial_folder",
    },
    {
        "category": "walnuts",
        "data_root": ROOT / "datasets/MVTec_AD_2_anomalib_all/walnuts_folder",
    },
]

MODEL_SIZE = "small"
MAX_EPOCHS = -1
MAX_STEPS = 5000
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 0.0001
WEIGHT_DECAY = 0.00001


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


def run_one_category(category: str, data_root: Path):
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    required_paths = [
        data_root / "train/good",
        data_root / "test/good",
        data_root / "test/bad",
        data_root / "ground_truth/bad",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset paths:\n" + "\n".join(missing))

    run_id = f"{category}_efficientad_{MODEL_SIZE}_{MAX_STEPS}steps"

    datamodule = Folder(
        name=f"ad2_{category}_efficientad_fixed",
        root=str(data_root),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/bad",
        mask_dir="ground_truth/bad",
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = EfficientAd(
        imagenet_dir=IMAGENETTE_DIR,
        model_size=MODEL_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    engine = Engine(
        default_root_dir=ENGINE_ROOT / run_id,
        logger=False,
        max_epochs=MAX_EPOCHS,
        max_steps=MAX_STEPS,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=5,
        log_every_n_steps=50,
    )

    print(f"[FIT] {category}")
    engine.fit(model=model, datamodule=datamodule)

    print(f"[TEST] {category}")
    metrics = engine.test(model=model, datamodule=datamodule)

    return metrics


def build_comparison(efficientad_df: pd.DataFrame) -> pd.DataFrame:
    main = pd.read_csv(STAGE11_MAIN)
    loco = pd.read_csv(STAGE13_LOCO)
    same = pd.read_csv(STAGE13_SAME)

    winclip = None
    if STAGE14_E.exists():
        winclip = pd.read_csv(STAGE14_E)

    rows = []

    for _, eff in efficientad_df[efficientad_df["status"] == "success"].iterrows():
        category = eff["category"]

        main_row = main[main["category_or_scope"] == category].iloc[0]

        loco_context = loco[
            (loco["category"] == category)
            & (loco["fusion_pair"] == "patchcore_plus_context")
        ].iloc[0]

        same_context = same[
            (same["category"] == category)
            & (same["fusion_pair"] == "patchcore_plus_context")
        ].iloc[0]

        rows.append({
            "category": category,
            "method_group": "modern_detector_baseline",
            "method": "EfficientAD fixed protocol",
            "image_auroc": eff["image_AUROC"],
            "image_f1": eff["image_F1Score"],
            "pixel_auroc": eff["pixel_AUROC"],
            "pixel_f1": eff["pixel_F1Score"],
            "protocol": f"model_size={MODEL_SIZE}, max_steps={MAX_STEPS}, train_batch_size={TRAIN_BATCH_SIZE}",
        })

        if winclip is not None:
            w = winclip[winclip["category"] == category]
            if not w.empty:
                w = w.iloc[0]
                rows.append({
                    "category": category,
                    "method_group": "external_vlm_baseline",
                    "method": "WinCLIP fixed protocol",
                    "image_auroc": w["image_AUROC"],
                    "image_f1": w["image_F1Score"],
                    "pixel_auroc": w["pixel_AUROC"],
                    "pixel_f1": w["pixel_F1Score"],
                    "protocol": f"class_name={w['class_name']}, k_shot={w['k_shot']}, scales={w['scales']}",
                })

        rows += [
            {
                "category": category,
                "method_group": "classical_detector",
                "method": "PatchCore",
                "image_auroc": main_row["patchcore_reference_auroc"],
                "image_f1": "",
                "pixel_auroc": "",
                "pixel_f1": "",
                "protocol": "Stage 11 reference",
            },
            {
                "category": category,
                "method_group": "vlm_branch",
                "method": "full-image VLM",
                "image_auroc": main_row["full_image_auroc"],
                "image_f1": "",
                "pixel_auroc": "",
                "pixel_f1": "",
                "protocol": "Stage 11 full-image baseline",
            },
            {
                "category": category,
                "method_group": "vlm_branch",
                "method": "context-aware VLM",
                "image_auroc": main_row["reported_method_auroc"],
                "image_f1": "",
                "pixel_auroc": "",
                "pixel_f1": "",
                "protocol": "Stage 11 context-aware VLM",
            },
            {
                "category": category,
                "method_group": "fusion_loco",
                "method": "PatchCore + context VLM, LOCO",
                "image_auroc": loco_context["auroc"],
                "image_f1": loco_context["best_f1"],
                "pixel_auroc": "",
                "pixel_f1": "",
                "protocol": "Stage 13 leave-one-category-out fusion",
            },
            {
                "category": category,
                "method_group": "fusion_same_set",
                "method": "PatchCore + context VLM, same-set",
                "image_auroc": same_context["auroc"],
                "image_f1": same_context["best_f1"],
                "pixel_auroc": "",
                "pixel_f1": "",
                "protocol": "Stage 13 same-set upper-bound fusion",
            },
        ]

    out = pd.DataFrame(rows)
    if not out.empty:
        out["image_auroc"] = pd.to_numeric(out["image_auroc"], errors="coerce")
        out = out.sort_values(["category", "image_auroc"], ascending=[True, False])

    return out


def write_report(efficientad_df: pd.DataFrame, comparison: pd.DataFrame, errors: list[str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Stage 15-D1 EfficientAD Primary 5000-Step Screening")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage runs EfficientAD on the four AD2 primary categories using a fixed 5000-step screening protocol.")
    lines.append("")
    lines.append("EfficientAD is included as a budgeted modern non-VLM detector screening baseline, so the paper does not rely only on the classic PatchCore baseline.")
    lines.append("")
    lines.append("## 2. Fixed Protocol")
    lines.append("")
    lines.append(f"- model_size: `{MODEL_SIZE}`")
    lines.append(f"- max_epochs: `{MAX_EPOCHS}`")
    lines.append(f"- max_steps: `{MAX_STEPS}`")
    lines.append(f"- train_batch_size: `{TRAIN_BATCH_SIZE}`")
    lines.append(f"- eval_batch_size: `{EVAL_BATCH_SIZE}`")
    lines.append(f"- num_workers: `{NUM_WORKERS}`")
    lines.append(f"- lr: `{LR}`")
    lines.append(f"- weight_decay: `{WEIGHT_DECAY}`")
    lines.append("")
    lines.append("## 3. EfficientAD Results")
    lines.append("")
    lines.append("| Category | Status | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for _, r in efficientad_df.iterrows():
        err = str(r.get("error", ""))
        if len(err) > 120:
            err = err[:120] + "..."
        lines.append(
            f"| {r['category']} | {r['status']} | {f4(r.get('image_AUROC', ''))} | "
            f"{f4(r.get('image_F1Score', ''))} | {f4(r.get('pixel_AUROC', ''))} | "
            f"{f4(r.get('pixel_F1Score', ''))} | `{err}` |"
        )

    lines.append("")
    lines.append("## 4. Unified Comparison")
    lines.append("")
    lines.append("| Category | Method | Group | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Protocol |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")

    for _, r in comparison.iterrows():
        lines.append(
            f"| {r['category']} | {r['method']} | {r['method_group']} | "
            f"{f4(r['image_auroc'])} | {f4(r['image_f1'])} | "
            f"{f4(r['pixel_auroc'])} | {f4(r['pixel_f1'])} | {r['protocol']} |"
        )

    lines.append("")
    lines.append("## 5. Aggregate Summary")
    lines.append("")

    success = efficientad_df[efficientad_df["status"] == "success"].copy()
    if success.empty:
        lines.append("No EfficientAD category succeeded.")
    else:
        success["image_AUROC_num"] = pd.to_numeric(success["image_AUROC"], errors="coerce")
        success["pixel_F1_num"] = pd.to_numeric(success["pixel_F1Score"], errors="coerce")
        lines.append(f"- Successful categories: `{len(success)}` / `{len(efficientad_df)}`")
        lines.append(f"- Mean EfficientAD image AUROC: `{f4(success['image_AUROC_num'].mean())}`")
        lines.append(f"- Best EfficientAD image AUROC: `{f4(success['image_AUROC_num'].max())}`")
        lines.append(f"- Worst EfficientAD image AUROC: `{f4(success['image_AUROC_num'].min())}`")
        lines.append(f"- Mean EfficientAD pixel F1: `{f4(success['pixel_F1_num'].mean())}`")

    lines.append("")
    lines.append("## 6. Decision")
    lines.append("")
    lines.append("This stage should be used to decide whether EfficientAD is a stronger formal detector baseline than PatchCore on AD2 primary categories.")
    lines.append("")
    lines.append("If EfficientAD remains weaker than PatchCore or PatchCore+context fusion on most categories, the paper can still use it as a modern detector baseline, but should not claim detector-level superiority over all modern detectors.")
    lines.append("")
    lines.append("If EfficientAD exceeds PatchCore on some categories, those categories should be discussed separately rather than ignored.")
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
    lines.append(f"- EfficientAD CSV: `{OUT_CSV.relative_to(ROOT)}`")
    lines.append(f"- Raw JSON: `{OUT_JSON.relative_to(ROOT)}`")
    lines.append(f"- Error log: `{OUT_ERROR.relative_to(ROOT)}`")
    lines.append(f"- Unified comparison CSV: `{OUT_COMPARISON.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    raw_records = []
    errors = []

    for item in CATEGORIES:
        category = item["category"]
        data_root = item["data_root"]

        print(f"[RUN] EfficientAD category={category}")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "category": category,
            "model_size": MODEL_SIZE,
            "max_epochs": MAX_EPOCHS,
            "max_steps": MAX_STEPS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "status": "failed",
            "image_AUROC": "",
            "image_F1Score": "",
            "pixel_AUROC": "",
            "pixel_F1Score": "",
            "error": "",
        }

        try:
            metrics = run_one_category(category, data_root)
            flat = flatten_metrics(metrics)

            row["status"] = "success"
            row["image_AUROC"] = flat.get("image_AUROC", "")
            row["image_F1Score"] = flat.get("image_F1Score", "")
            row["pixel_AUROC"] = flat.get("pixel_AUROC", "")
            row["pixel_F1Score"] = flat.get("pixel_F1Score", "")
            row["raw_metrics"] = str(flat)

            raw_records.append({
                "category": category,
                "model_size": MODEL_SIZE,
                "max_epochs": MAX_EPOCHS,
                "max_steps": MAX_STEPS,
                "metrics": flat,
            })

            print("[OK]", category, flat)

        except Exception:
            err = traceback.format_exc()
            row["error"] = err.splitlines()[-1] if err else "unknown error"
            errors.append(f"\n===== {category} =====\n{err}")
            print("[ERROR]", category, row["error"])

        rows.append(row)

    efficientad_df = pd.DataFrame(rows)
    efficientad_df.to_csv(OUT_CSV, index=False)

    comparison = build_comparison(efficientad_df)
    comparison.to_csv(OUT_COMPARISON, index=False)

    OUT_JSON.write_text(
        json.dumps(raw_records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    OUT_ERROR.write_text("\n".join(errors), encoding="utf-8")

    write_report(efficientad_df, comparison, errors)

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_COMPARISON)
    print("[DONE]", OUT_JSON)
    print("[DONE]", OUT_ERROR)
    print("[DONE]", OUT_REPORT)

    print("\n===== EfficientAD results =====")
    print(efficientad_df.to_string(index=False))

    print("\n===== Unified comparison =====")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
