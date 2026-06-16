import argparse
import csv
import json
from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


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

    run_dir = Path(args.output_root) / args.category
    run_dir.mkdir(parents=True, exist_ok=True)

    # PatchCore paper-style preprocessing:
    # resize to 256, center crop to 224.
    pre_processor = Patchcore.configure_pre_processor(
        image_size=(256, 256),
        center_crop_size=(224, 224),
    )

    datamodule = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

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
    print(f"[INFO] Output dir: {run_dir.resolve()}")

    engine.fit(model=model, datamodule=datamodule)
    test_result = engine.test(model=model, datamodule=datamodule)

    metrics = to_plain_dict(test_result)
    metrics["model"] = "patchcore"
    metrics["dataset"] = "MVTecAD"
    metrics["category"] = args.category
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

    write_header = not summary_csv.exists()
    fieldnames = list(metrics.keys())

    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            old_header = next(reader, None)
        if old_header:
            fieldnames = list(dict.fromkeys(old_header + fieldnames))

    existing_rows = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

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
