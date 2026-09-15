from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(".").resolve()

DATA_ROOT = ROOT / "datasets" / "MVTec_AD_2_anomalib" / "vial_folder"
RUN_DIR = ROOT / "runs" / "stage10_mvtecad2_patchcore" / "vial_patchcore_pilot"

OUT_DIR = ROOT / "results" / "stage10_dataset_expansion"
DOC_DIR = ROOT / "docs" / "stage10_dataset_expansion"

OUT_METRICS = OUT_DIR / "stage10_c_mvtecad2_vial_patchcore_metrics.csv"
OUT_PREDS = OUT_DIR / "stage10_c_mvtecad2_vial_patchcore_predictions.csv"
OUT_REPORT = DOC_DIR / "stage10_c_mvtecad2_vial_patchcore_report.md"


def filter_kwargs(cls_or_fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls_or_fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def import_patchcore_class():
    try:
        from anomalib.models import Patchcore
        return Patchcore
    except Exception:
        pass

    try:
        from anomalib.models import PatchCore
        return PatchCore
    except Exception:
        pass

    try:
        from anomalib.models.image.patchcore import Patchcore
        return Patchcore
    except Exception as e:
        raise ImportError(f"Cannot import PatchCore/Patchcore from anomalib: {repr(e)}")


def make_folder_datamodule():
    from anomalib.data import Folder

    kwargs = {
        "name": "mvtecad2_vial_folder",
        "root": str(DATA_ROOT),
        "normal_dir": "train/good",
        "abnormal_dir": "test/bad",
        "normal_test_dir": "test/good",
        "mask_dir": "ground_truth/bad",
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_workers": 0,
    }

    dm = Folder(**filter_kwargs(Folder, kwargs))
    return dm


def make_patchcore_model():
    PatchCoreCls = import_patchcore_class()

    kwargs = {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "pre_trained": True,
        "num_neighbors": 9,
    }

    model = PatchCoreCls(**filter_kwargs(PatchCoreCls, kwargs))
    return model


def make_engine():
    from anomalib.engine import Engine

    kwargs = {
        "default_root_dir": str(RUN_DIR),
        "accelerator": "auto",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
    }

    engine = Engine(**filter_kwargs(Engine, kwargs))
    return engine


def to_plain_value(x: Any) -> Any:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.detach().cpu().item()
            return x.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(x, Path):
        return str(x)

    return x


def get_batch_field(batch: Any, field: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(field, None)

    if hasattr(batch, field):
        return getattr(batch, field)

    return None


def get_n_items(batch: Any) -> int:
    image_path = get_batch_field(batch, "image_path")
    if image_path is not None:
        try:
            return len(image_path)
        except Exception:
            pass

    image = get_batch_field(batch, "image")
    if image is not None:
        try:
            return int(image.shape[0])
        except Exception:
            pass

    return 1


def get_item_value(value: Any, index: int) -> Any:
    if value is None:
        return ""

    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return to_plain_value(value)
            return to_plain_value(value[index])
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        if index < len(value):
            return to_plain_value(value[index])
        return ""

    return to_plain_value(value)


def summarize_anomaly_map(value: Any, index: int) -> Dict[str, Any]:
    result = {
        "anomaly_map_min": "",
        "anomaly_map_max": "",
        "anomaly_map_mean": "",
        "anomaly_map_shape": "",
    }

    if value is None:
        return result

    try:
        import torch

        if isinstance(value, torch.Tensor):
            amap = value[index] if value.ndim >= 3 else value
            amap = amap.detach().float().cpu()
            result["anomaly_map_min"] = float(amap.min().item())
            result["anomaly_map_max"] = float(amap.max().item())
            result["anomaly_map_mean"] = float(amap.mean().item())
            result["anomaly_map_shape"] = str(list(amap.shape))
            return result
    except Exception:
        pass

    return result


def extract_prediction_records(predict_outputs: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if predict_outputs is None:
        return records

    for batch_idx, batch in enumerate(predict_outputs):
        n = get_n_items(batch)

        image_path = get_batch_field(batch, "image_path")
        mask_path = get_batch_field(batch, "mask_path")
        gt_label = get_batch_field(batch, "gt_label")
        pred_label = get_batch_field(batch, "pred_label")
        pred_score = get_batch_field(batch, "pred_score")
        anomaly_map = get_batch_field(batch, "anomaly_map")

        for i in range(n):
            row = {
                "dataset": "MVTec AD 2",
                "category": "vial",
                "batch_idx": batch_idx,
                "item_idx": i,
                "image_path": get_item_value(image_path, i),
                "mask_path": get_item_value(mask_path, i),
                "gt_label": get_item_value(gt_label, i),
                "pred_label": get_item_value(pred_label, i),
                "pred_score": get_item_value(pred_score, i),
            }
            row.update(summarize_anomaly_map(anomaly_map, i))
            records.append(row)

    return records


def write_report(metrics_df: pd.DataFrame, pred_df: pd.DataFrame, predict_error: str) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Stage 10-C MVTec AD 2 Vial PatchCore Pilot")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage runs the first PatchCore pilot on MVTec AD 2 vial using the Anomalib Folder adapter.")
    lines.append("It trains PatchCore on normal train/validation images and evaluates on public test normal/anomaly images.")
    lines.append("")
    lines.append("## 2. Input Dataset")
    lines.append("")
    lines.append(f"- Folder root: `{DATA_ROOT.relative_to(ROOT)}`")
    lines.append("- Train normal: `train/good`")
    lines.append("- Test normal: `test/good`")
    lines.append("- Test anomaly: `test/bad`")
    lines.append("- Masks: `ground_truth/bad`")
    lines.append("")
    lines.append("## 3. Output Files")
    lines.append("")
    lines.append(f"- Metrics: `{OUT_METRICS.relative_to(ROOT)}`")
    lines.append(f"- Predictions: `{OUT_PREDS.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append(f"- Run dir: `{RUN_DIR.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 4. Test Metrics")
    lines.append("")

    if metrics_df.empty:
        lines.append("No metric rows were returned by `engine.test`.")
    else:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        row = metrics_df.iloc[0].to_dict()
        for k, v in row.items():
            lines.append(f"| {k} | {v} |")

    lines.append("")
    lines.append("## 5. Prediction Extraction")
    lines.append("")
    lines.append(f"- Prediction rows: `{len(pred_df)}`")
    if predict_error:
        lines.append(f"- Prediction extraction error: `{predict_error}`")
    else:
        lines.append("- Prediction extraction error: ``")
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append("This is the first detector baseline on MVTec AD 2 vial.")
    lines.append("If metrics are valid and prediction rows contain anomaly maps or scores, Stage 10-D should generate candidate crops from these PatchCore outputs.")
    lines.append("If prediction rows are empty, Stage 10-D should first patch prediction extraction before VLM reasoning.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing Folder dataset root: {DATA_ROOT}")

    print("[INFO] Data root:", DATA_ROOT)
    print("[INFO] Run dir:", RUN_DIR)

    dm = make_folder_datamodule()
    model = make_patchcore_model()
    engine = make_engine()

    print("[INFO] Start PatchCore fit ...")
    engine.fit(model=model, datamodule=dm)
    print("[INFO] PatchCore fit finished.")

    print("[INFO] Start PatchCore test ...")
    test_results = engine.test(model=model, datamodule=dm)
    print("[INFO] PatchCore test finished.")

    if isinstance(test_results, list):
        metrics_df = pd.DataFrame(test_results)
    elif isinstance(test_results, dict):
        metrics_df = pd.DataFrame([test_results])
    else:
        metrics_df = pd.DataFrame([{"raw_test_results": repr(test_results)}])

    metrics_df.to_csv(OUT_METRICS, index=False)

    predict_error = ""
    prediction_records: List[Dict[str, Any]] = []

    try:
        print("[INFO] Start PatchCore predict ...")
        predict_outputs = engine.predict(model=model, datamodule=dm)
        prediction_records = extract_prediction_records(predict_outputs)
        print("[INFO] PatchCore predict finished.")
    except Exception as e:
        predict_error = repr(e)
        print("[WARN] Prediction extraction failed:", predict_error)

    pred_df = pd.DataFrame(prediction_records)
    if pred_df.empty:
        pred_df = pd.DataFrame([{
            "dataset": "MVTec AD 2",
            "category": "vial",
            "prediction_error": predict_error,
        }])

    pred_df.to_csv(OUT_PREDS, index=False)

    write_report(metrics_df, pred_df, predict_error)

    print("[DONE]", OUT_METRICS)
    print("[DONE]", OUT_PREDS)
    print("[DONE]", OUT_REPORT)
    print("")
    print("Metrics:")
    print(metrics_df.to_string(index=False))
    print("")
    print("Prediction rows:", len(pred_df))
    if predict_error:
        print("Prediction error:", predict_error)


if __name__ == "__main__":
    main()
