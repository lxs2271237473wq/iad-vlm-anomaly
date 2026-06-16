"""Run first-stage Anomalib baselines on selected MVTec AD categories.

This script intentionally uses standard Anomalib models first:
- PatchCore: strong traditional memory-bank baseline.
- PaDiM: classical feature-distribution baseline.
- WinCLIP zero-shot/few-shot: first VLM baseline.

The goal is not novelty. The goal is to establish reproducible reference numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def import_anomalib() -> tuple[Any, dict[str, Any], Any]:
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    try:
        from anomalib.models.image import EfficientAd, Padim, Patchcore, WinClip
    except Exception:  # noqa: BLE001
        from anomalib.models import EfficientAd, Padim, Patchcore, WinClip

    return MVTecAD, {
        "patchcore": Patchcore,
        "padim": Padim,
        "efficientad": EfficientAd,
        "winclip_zero": WinClip,
        "winclip_4shot": WinClip,
    }, Engine


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def make_mvtec_datamodule(MVTecAD: Any, data_cfg: dict[str, Any], category: str) -> Any:
    root = Path(data_cfg["root"])
    image_size = data_cfg.get("image_size", 256)
    kwargs = {
        "root": root,
        "category": category,
        "train_batch_size": data_cfg.get("train_batch_size", 16),
        "eval_batch_size": data_cfg.get("eval_batch_size", 16),
        "num_workers": data_cfg.get("num_workers", 8),
        "image_size": image_size,
    }
    try:
        return MVTecAD(**kwargs)
    except TypeError:
        # Fallback for older/newer Anomalib signatures.
        minimal = {"root": root, "category": category, "image_size": image_size}
        return MVTecAD(**minimal)


def make_model(name: str, ModelClass: Any, category: str) -> Any:
    if name == "winclip_zero":
        return ModelClass(class_name=category, k_shot=0)
    if name == "winclip_4shot":
        return ModelClass(class_name=category, k_shot=4)
    return ModelClass()


def flatten_metrics(result: Any) -> dict[str, Any]:
    if isinstance(result, list) and result:
        result = result[0]
    if not isinstance(result, dict):
        return {"raw_result": str(result)}
    out: dict[str, Any] = {}
    for key, value in result.items():
        try:
            if hasattr(value, "item"):
                value = value.item()
            out[str(key)] = float(value) if isinstance(value, (int, float)) else value
        except Exception:  # noqa: BLE001
            out[str(key)] = str(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/first_baselines.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of model names to run.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    MVTecAD, model_map, Engine = import_anomalib()
    out_root = Path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    baselines = cfg["baselines"] if args.only is None else args.only

    for category in cfg["data"]["categories"]:
        for model_name in baselines:
            if model_name not in model_map:
                print(f"Skipping unknown baseline: {model_name}")
                continue

            run_dir = out_root / model_name / category
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== Running {model_name} on {category} ===")

            datamodule = make_mvtec_datamodule(MVTecAD, cfg["data"], category)
            model = make_model(model_name, model_map[model_name], category)

            try:
                engine = Engine(default_root_dir=str(run_dir))
            except TypeError:
                engine = Engine()

            # WinCLIP does not train; one-class baselines need fit before test.
            if model_name.startswith("winclip"):
                result = engine.test(model=model, datamodule=datamodule)
            else:
                engine.fit(model=model, datamodule=datamodule)
                result = engine.test(model=model, datamodule=datamodule)

            metrics = flatten_metrics(result)
            metrics.update({"category": category, "model": model_name})
            rows.append(metrics)
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save a loose CSV with all discovered metric keys.
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    csv_path = out_root / "baseline_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary: {csv_path}")


if __name__ == "__main__":
    main()
