import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

from experiments.baselines.patchcore_mvtec import MVTEC_DEFECTS
from experiments.analysis.patchcore_candidate_regions import (
    get_field,
    take_item,
    mask_to_2d,
    normalize_map,
    connected_components,
    component_to_record,
)


def load_thresholds(path):
    df = pd.read_csv(path)
    return {str(row["category"]): float(row["best_threshold"]) for _, row in df.iterrows()}


def canonical_path(path):
    s = str(path).replace("\\", "/")
    marker = "datasets/MVTecAD/"
    if marker in s:
        return s[s.index(marker):]
    return s


def build_full_test_datamodule(args, category):
    dataset_root = Path(args.data_root).resolve()
    category_root = dataset_root / category

    if category not in MVTEC_DEFECTS:
        raise ValueError(f"Unknown MVTec AD category: {category}")

    if not category_root.exists():
        raise FileNotFoundError(f"Category root does not exist: {category_root}")

    defects = MVTEC_DEFECTS[category]
    abnormal_dirs = [f"test/{defect}" for defect in defects]
    mask_dirs = [f"ground_truth/{defect}" for defect in defects]

    for rel_path in ["train/good", "test/good", *abnormal_dirs, *mask_dirs]:
        path = category_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Required MVTec path does not exist: {path}")

    # Important:
    # same_as_test prevents validation splitting from removing half of the test samples.
    datamodule = Folder(
        name=f"MVTecAD_{category}_full_test",
        root=str(category_root),
        normal_dir="train/good",
        abnormal_dir=abnormal_dirs,
        normal_test_dir="test/good",
        mask_dir=mask_dirs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split_mode="same_as_test",
    )

    return datamodule


def collect_predictions(args, category):
    datamodule = build_full_test_datamodule(args, category)

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(256, 256),
        center_crop_size=(224, 224),
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_processor=pre_processor,
    )

    work_dir = Path(args.work_root) / "MVTecAD" / category
    work_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        default_root_dir=str(work_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )

    print(f"[INFO] Fitting PatchCore full-test candidate model: {category}")
    engine.fit(model=model, datamodule=datamodule)

    print(f"[INFO] Predicting full-test anomaly maps: {category}")
    return engine.predict(model=model, datamodule=datamodule)


def run_category(args, category, threshold):
    predictions = collect_predictions(args, category)

    out_dir = Path(args.output_root) / "MVTecAD" / category / "candidate_regions"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    image_counter = 0
    no_candidate_count = 0
    normal_counter = 0

    for batch in predictions:
        image_paths = get_field(batch, "image_path")
        gt_masks = get_field(batch, "gt_mask")
        anomaly_maps = get_field(batch, "anomaly_map")

        batch_size = len(image_paths) if isinstance(image_paths, (list, tuple)) else 1

        for i in range(batch_size):
            image_path = str(take_item(image_paths, i))
            gt_mask = mask_to_2d(take_item(gt_masks, i))
            anomaly_map = normalize_map(take_item(anomaly_maps, i))

            if anomaly_map is None:
                continue

            if gt_mask is not None:
                gt_mask = gt_mask > 0

            # Only keep abnormal images for defect reasoning.
            if gt_mask is None or gt_mask.sum() == 0:
                normal_counter += 1
                continue

            binary_mask = anomaly_map >= threshold
            components = connected_components(binary_mask)

            records = []
            for comp in components:
                if len(comp) < args.min_area:
                    continue
                rec = component_to_record(comp, anomaly_map, gt_mask)
                records.append(rec)

            records = sorted(
                records,
                key=lambda r: (r["area"], r["mean_score"]),
                reverse=True,
            )
            records = records[: args.top_components]

            image_counter += 1

            if len(records) == 0:
                no_candidate_count += 1
                rows.append(
                    {
                        "category": category,
                        "image_path": image_path,
                        "canonical_image_path": canonical_path(image_path),
                        "component_rank": 0,
                        "threshold": threshold,
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                        "cx": "",
                        "cy": "",
                        "area": 0,
                        "mean_score": "",
                        "max_score": "",
                        "gt_iou": "",
                        "gt_f1": "",
                    }
                )
                continue

            for rank, rec in enumerate(records, start=1):
                rows.append(
                    {
                        "category": category,
                        "image_path": image_path,
                        "canonical_image_path": canonical_path(image_path),
                        "component_rank": rank,
                        "threshold": threshold,
                        "x1": rec["x1"],
                        "y1": rec["y1"],
                        "x2": rec["x2"],
                        "y2": rec["y2"],
                        "cx": f"{rec['cx']:.2f}",
                        "cy": f"{rec['cy']:.2f}",
                        "area": rec["area"],
                        "mean_score": f"{rec['mean_score']:.6f}",
                        "max_score": f"{rec['max_score']:.6f}",
                        "gt_iou": "" if rec["gt_iou"] is None else f"{rec['gt_iou']:.6f}",
                        "gt_f1": "" if rec["gt_f1"] is None else f"{rec['gt_f1']:.6f}",
                    }
                )

    csv_path = out_dir / "candidate_regions.csv"

    fieldnames = [
        "category",
        "image_path",
        "canonical_image_path",
        "component_rank",
        "threshold",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "area",
        "mean_score",
        "max_score",
        "gt_iou",
        "gt_f1",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[DONE] {category}: abnormal_images={image_counter}, "
        f"normal_skipped={normal_counter}, no_candidate={no_candidate_count}, "
        f"candidate_rows={len(rows)}"
    )
    print(f"[DONE] Saved full-test candidates to: {csv_path}")

    return {
        "category": category,
        "threshold": threshold,
        "num_abnormal_images": image_counter,
        "num_normal_skipped": normal_counter,
        "no_candidate_images": no_candidate_count,
        "candidate_rows": len(rows),
        "candidate_csv": str(csv_path),
    }


def evaluate_manifest_coverage(args):
    manifest_path = Path(args.manifest_csv)

    if not manifest_path.exists():
        print(f"[WARN] Manifest not found, skip coverage check: {manifest_path}")
        return None

    manifest = pd.read_csv(manifest_path)
    coverage_rows = []

    for category in args.categories:
        category_manifest = manifest[manifest["category"] == category].copy()
        manifest_paths = set(canonical_path(p) for p in category_manifest["image_path"].tolist())

        candidate_csv = (
            Path(args.output_root)
            / "MVTecAD"
            / category
            / "candidate_regions"
            / "candidate_regions.csv"
        )

        if not candidate_csv.exists():
            coverage_rows.append(
                {
                    "category": category,
                    "manifest_images": len(manifest_paths),
                    "candidate_images": 0,
                    "covered_images": 0,
                    "coverage_ratio": 0.0,
                    "missing_images": len(manifest_paths),
                }
            )
            continue

        candidate_df = pd.read_csv(candidate_csv)
        valid_df = candidate_df[pd.to_numeric(candidate_df["component_rank"], errors="coerce") > 0].copy()

        if "canonical_image_path" in valid_df.columns:
            candidate_paths = set(valid_df["canonical_image_path"].astype(str).tolist())
        else:
            candidate_paths = set(canonical_path(p) for p in valid_df["image_path"].tolist())

        covered = manifest_paths & candidate_paths
        missing = manifest_paths - candidate_paths

        coverage_rows.append(
            {
                "category": category,
                "manifest_images": len(manifest_paths),
                "candidate_images": len(candidate_paths),
                "covered_images": len(covered),
                "coverage_ratio": len(covered) / len(manifest_paths) if manifest_paths else 0.0,
                "missing_images": len(missing),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)

    mean_row = {
        "category": "MEAN",
        "manifest_images": coverage_df["manifest_images"].sum(),
        "candidate_images": coverage_df["candidate_images"].sum(),
        "covered_images": coverage_df["covered_images"].sum(),
        "coverage_ratio": (
            coverage_df["covered_images"].sum() / coverage_df["manifest_images"].sum()
            if coverage_df["manifest_images"].sum() > 0
            else 0.0
        ),
        "missing_images": coverage_df["missing_images"].sum(),
    }

    coverage_df = pd.concat([coverage_df, pd.DataFrame([mean_row])], ignore_index=True)

    out_path = Path(args.output_root) / "full_test_candidate_coverage_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(out_path, index=False)

    print("\n========== Full-test Candidate Coverage ==========")
    print(coverage_df.to_string(index=False))
    print(f"[DONE] Saved coverage summary to: {out_path}")

    return coverage_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument(
        "--threshold_csv",
        type=str,
        default="results/analysis/patchcore_threshold_diagnosis/threshold_diagnosis_summary.csv",
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default="results/analysis/defect_type_reasoning/mvtec_defect_type_reasoning_manifest.csv",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/analysis/full_test_patchcore_candidate_regions",
    )
    parser.add_argument(
        "--work_root",
        type=str,
        default="runs/analysis/full_test_patchcore_candidate_regions",
    )
    parser.add_argument("--top_components", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    thresholds = load_thresholds(args.threshold_csv)

    summary = []

    for category in args.categories:
        if category not in thresholds:
            raise KeyError(f"Category {category} not found in threshold CSV: {args.threshold_csv}")

        result = run_category(args, category, thresholds[category])
        summary.append(result)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "full_test_candidate_region_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print(f"[DONE] Saved full-test candidate summary to: {summary_path}")

    evaluate_manifest_coverage(args)


if __name__ == "__main__":
    main()
