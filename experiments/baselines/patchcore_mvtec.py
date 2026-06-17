import argparse
import csv
import json
from pathlib import Path

import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


MVTEC_DEFECTS = {
    "bottle": ["broken_large", "broken_small", "contamination"],
    "cable": [
        "bent_wire",
        "cable_swap",
        "combined",
        "cut_inner_insulation",
        "cut_outer_insulation",
        "missing_cable",
        "missing_wire",
        "poke_insulation",
    ],
    "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
    "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
    "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
    "hazelnut": ["crack", "cut", "hole", "print"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
    "metal_nut": ["bent", "color", "flip", "scratch"],
    "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
    "screw": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
    "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
    "toothbrush": ["defective"],
    "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
    "wood": ["color", "combined", "hole", "liquid", "scratch"],
    "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "split_teeth", "squeezed_teeth"],
}


def to_plain_dict(result):
    """Convert Anomalib / Lightning metric outputs into a JSON-serializable dict."""
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if not isinstance(result, dict):
        return {"raw_result": str(result)}

    out = {}
    for key, value in result.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()

        try:
            if isinstance(value, (int, float)):
                out[key] = float(value)
            else:
                out[key] = str(value)
        except Exception:
            out[key] = str(value)

    return out


def build_mvtec_folder_datamodule(args):
    dataset_root = Path(args.data_root).resolve()
    category_root = dataset_root / args.category

    if args.category not in MVTEC_DEFECTS:
        raise ValueError(f"Unknown MVTec AD category: {args.category}")

    if not category_root.exists():
        raise FileNotFoundError(f"Category root does not exist: {category_root}")

    defects = MVTEC_DEFECTS[args.category]
    abnormal_dirs = [f"test/{defect}" for defect in defects]
    mask_dirs = [f"ground_truth/{defect}" for defect in defects]

    for rel_path in ["train/good", "test/good", *abnormal_dirs, *mask_dirs]:
        path = category_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Required MVTec path does not exist: {path}")

    datamodule = Folder(
        name=f"MVTecAD_{args.category}",
        root=str(category_root),
        normal_dir="train/good",
        abnormal_dir=abnormal_dirs,
        normal_test_dir="test/good",
        mask_dir=mask_dirs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    return datamodule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--output_root", type=str, default="runs/baselines/patchcore")
    parser.add_argument("--summary_csv", type=str, default="results/baselines/patchcore_mvtec_summary.csv")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.output_root) / "MVTecAD" / args.category
    run_dir.mkdir(parents=True, exist_ok=True)

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(256, 256),
        center_crop_size=(224, 224),
    )

    datamodule = build_mvtec_folder_datamodule(args)

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_processor=pre_processor,
    )

    engine = Engine(
        default_root_dir=str(run_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )

    print(f"[INFO] Running PatchCore on MVTec AD category: {args.category}")
    print(f"[INFO] Dataset root: {Path(args.data_root).resolve()}")
    print(f"[INFO] Category root: {Path(args.data_root).resolve() / args.category}")
    print(f"[INFO] Output dir: {run_dir.resolve()}")

    engine.fit(model=model, datamodule=datamodule)
    test_result = engine.test(model=model, datamodule=datamodule)

    metrics = to_plain_dict(test_result)
    metrics["model"] = "patchcore"
    metrics["dataset"] = "MVTecAD"
    metrics["category"] = args.category
    metrics["datamodule"] = "Folder"
    metrics["backbone"] = "wide_resnet50_2"
    metrics["layers"] = "layer2,layer3"
    metrics["coreset_sampling_ratio"] = 0.1
    metrics["num_neighbors"] = 9
    metrics["image_size"] = "256"
    metrics["center_crop_size"] = "224"

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(metrics.keys())

    existing_rows = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys(reader.fieldnames + fieldnames))

    existing_rows = [
        row for row in existing_rows
        if not (
            row.get("model") == "patchcore"
            and row.get("dataset") == "MVTecAD"
            and row.get("category") == args.category
        )
    ]

    existing_rows.append(metrics)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

    print(f"[DONE] Metrics saved to: {metrics_path}")
    print(f"[DONE] Summary updated: {summary_csv}")


if __name__ == "__main__":
    main()
