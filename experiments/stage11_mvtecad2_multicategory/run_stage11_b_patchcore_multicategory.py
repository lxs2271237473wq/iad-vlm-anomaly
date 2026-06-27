from __future__ import annotations

import argparse
import gc
import inspect
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(".").resolve()

ADAPTER_ROOT = ROOT / "datasets" / "MVTec_AD_2_anomalib_all"
STAGE11A_SUMMARY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_a_folder_adapter_summary.csv"

RUN_ROOT = ROOT / "runs/stage11_mvtecad2_multicategory/patchcore_baseline"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_METRICS = OUT_DIR / "stage11_b_patchcore_multicategory_metrics.csv"
OUT_PREDS = OUT_DIR / "stage11_b_patchcore_multicategory_predictions.csv"
OUT_STATUS = OUT_DIR / "stage11_b_patchcore_multicategory_status.csv"
OUT_REPORT = DOC_DIR / "stage11_b_patchcore_multicategory_report.md"

DEFAULT_CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


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


def make_folder_datamodule(category: str):
    from anomalib.data import Folder

    data_root = ADAPTER_ROOT / f"{category}_folder"

    kwargs = {
        "name": f"mvtecad2_{category}_folder",
        "root": str(data_root),
        "normal_dir": "train/good",
        "abnormal_dir": "test/bad",
        "normal_test_dir": "test/good",
        "mask_dir": "ground_truth/bad",
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_workers": 0,
    }

    return Folder(**filter_kwargs(Folder, kwargs))


def make_patchcore_model():
    PatchCoreCls = import_patchcore_class()

    kwargs = {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "pre_trained": True,
        "num_neighbors": 9,
    }

    return PatchCoreCls(**filter_kwargs(PatchCoreCls, kwargs))


def make_engine(category: str):
    from anomalib.engine import Engine

    run_dir = RUN_ROOT / category

    kwargs = {
        "default_root_dir": str(run_dir),
        "accelerator": "auto",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
    }

    return Engine(**filter_kwargs(Engine, kwargs))


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


def extract_prediction_records(category: str, predict_outputs: Any) -> List[Dict[str, Any]]:
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
                "category": category,
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


def metrics_to_dataframe(category: str, test_results: Any) -> pd.DataFrame:
    if isinstance(test_results, list):
        df = pd.DataFrame(test_results)
    elif isinstance(test_results, dict):
        df = pd.DataFrame([test_results])
    else:
        df = pd.DataFrame([{"raw_test_results": repr(test_results)}])

    df.insert(0, "category", category)
    df.insert(0, "dataset", "MVTec AD 2")

    return df


def load_category_counts() -> pd.DataFrame:
    if not STAGE11A_SUMMARY.exists():
        return pd.DataFrame()

    df = pd.read_csv(STAGE11A_SUMMARY)

    rows = []
    for category, part in df.groupby("category"):
        def count(subset: str, label: str) -> int:
            m = (part["target_subset"] == subset) & (part["target_label"] == label)
            if not m.any():
                return 0
            return int(part.loc[m, "num_images"].sum())

        rows.append({
            "category": category,
            "train_good": count("train", "good"),
            "test_good": count("test", "good"),
            "test_bad": count("test", "bad"),
            "test_total": count("test", "good") + count("test", "bad"),
        })

    return pd.DataFrame(rows)


def clear_gpu_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()


def run_one_category(category: str) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    start = time.time()
    data_root = ADAPTER_ROOT / f"{category}_folder"
    run_dir = RUN_ROOT / category

    status = {
        "dataset": "MVTec AD 2",
        "category": category,
        "data_root": str(data_root.relative_to(ROOT)),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "success": False,
        "fit_success": False,
        "test_success": False,
        "predict_success": False,
        "num_prediction_rows": 0,
        "elapsed_sec": 0.0,
        "error": "",
    }

    if not data_root.exists():
        raise FileNotFoundError(f"Missing folder adapter: {data_root}")

    print("")
    print(f"========== Stage 11-B PatchCore: {category} ==========")
    print("[INFO] Data root:", data_root)
    print("[INFO] Run dir:", run_dir)

    metrics_df = pd.DataFrame()
    pred_df = pd.DataFrame()

    dm = make_folder_datamodule(category)
    model = make_patchcore_model()
    engine = make_engine(category)

    print("[INFO] fit start")
    engine.fit(model=model, datamodule=dm)
    status["fit_success"] = True
    print("[INFO] fit done")

    print("[INFO] test start")
    test_results = engine.test(model=model, datamodule=dm)
    metrics_df = metrics_to_dataframe(category, test_results)
    status["test_success"] = True
    print("[INFO] test done")

    print("[INFO] predict start")
    predict_outputs = engine.predict(model=model, datamodule=dm)
    prediction_records = extract_prediction_records(category, predict_outputs)
    pred_df = pd.DataFrame(prediction_records)
    status["predict_success"] = True
    status["num_prediction_rows"] = int(len(pred_df))
    print("[INFO] predict done")

    status["success"] = bool(status["fit_success"] and status["test_success"])
    status["elapsed_sec"] = round(time.time() - start, 3)

    return metrics_df, pred_df, status


def write_report(metrics: pd.DataFrame, preds: pd.DataFrame, status: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    counts = load_category_counts()

    lines: List[str] = []

    lines.append("# Stage 11-B MVTec AD 2 Multi-category PatchCore Baseline")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage runs PatchCore baseline experiments over all validated MVTec AD 2 categories using the Stage 11-A Anomalib Folder adapters.")
    lines.append("")
    lines.append("This step trains and evaluates PatchCore. It does not run VLM reasoning, generate candidate crops, or modify the original dataset.")
    lines.append("")
    lines.append("## 2. Input")
    lines.append("")
    lines.append(f"- Folder adapter root: `{ADAPTER_ROOT.relative_to(ROOT)}`")
    lines.append(f"- Stage 11-A summary: `{STAGE11A_SUMMARY.relative_to(ROOT)}`")
    lines.append("- Normal training split: `train/good`")
    lines.append("- Public test normal split: `test/good`")
    lines.append("- Public test anomaly split: `test/bad`")
    lines.append("- Mask split: `ground_truth/bad`")
    lines.append("")
    lines.append("## 3. PatchCore Configuration")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append("| Backbone | wide_resnet50_2 |")
    lines.append("| Layers | layer2, layer3 |")
    lines.append("| Pretrained | True |")
    lines.append("| Neighbors | 9 |")
    lines.append("| Train batch size | 8 |")
    lines.append("| Eval batch size | 8 |")
    lines.append("| Workers | 0 |")
    lines.append("")
    lines.append("## 4. Output Files")
    lines.append("")
    lines.append(f"- Metrics: `{OUT_METRICS.relative_to(ROOT)}`")
    lines.append(f"- Predictions: `{OUT_PREDS.relative_to(ROOT)}`")
    lines.append(f"- Status: `{OUT_STATUS.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append(f"- Run root: `{RUN_ROOT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 5. Dataset Size by Category")
    lines.append("")

    if counts.empty:
        lines.append("Stage 11-A category count file was not available.")
    else:
        lines.append("| Category | Train good | Test good | Test bad | Test total |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in counts.sort_values("category").iterrows():
            lines.append(
                f"| {r['category']} | {int(r['train_good'])} | {int(r['test_good'])} | "
                f"{int(r['test_bad'])} | {int(r['test_total'])} |"
            )

    lines.append("")
    lines.append("## 6. Execution Status")
    lines.append("")
    lines.append("| Category | Success | Fit | Test | Predict | Prediction rows | Time sec | Error |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")

    for _, r in status.sort_values("category").iterrows():
        err = "" if pd.isna(r["error"]) else str(r["error"]).replace("|", "/")
        lines.append(
            f"| {r['category']} | {r['success']} | {r['fit_success']} | {r['test_success']} | "
            f"{r['predict_success']} | {int(r['num_prediction_rows'])} | "
            f"{float(r['elapsed_sec']):.1f} | `{err}` |"
        )

    lines.append("")
    lines.append("## 7. Metrics")
    lines.append("")

    if metrics.empty:
        lines.append("No metrics were produced.")
    else:
        metric_cols = [c for c in metrics.columns if c not in {"dataset"}]
        lines.append("| " + " | ".join(metric_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(metric_cols)) + "|")

        for _, r in metrics.sort_values("category").iterrows():
            vals = []
            for c in metric_cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## 8. Interpretation")
    lines.append("")
    lines.append("This table establishes the detector-side baseline across all MVTec AD 2 categories.")
    lines.append("The next step should not compare VLM methods before confirming detector quality per category, because poor localization can make crop-based VLM reasoning look artificially weak.")
    lines.append("")
    lines.append("## 9. Next Step")
    lines.append("")
    lines.append("Stage 11-C should generate PatchCore candidate regions for the successfully validated categories, then Stage 11-D should evaluate full-image versus context-aware crop VLM reasoning.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def parse_categories(text: str) -> List[str]:
    if not text:
        return DEFAULT_CATEGORIES

    items = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in items if x not in DEFAULT_CATEGORIES]

    if unknown:
        raise ValueError(f"Unknown categories: {unknown}. Valid: {DEFAULT_CATEGORIES}")

    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated categories. Default: all 8 AD2 categories.",
    )
    args = parser.parse_args()

    categories = parse_categories(args.categories)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    metrics_list: List[pd.DataFrame] = []
    pred_list: List[pd.DataFrame] = []
    status_rows: List[Dict[str, Any]] = []

    for category in categories:
        start = time.time()

        try:
            metric_df, pred_df, status = run_one_category(category)

            if not metric_df.empty:
                metrics_list.append(metric_df)

            if not pred_df.empty:
                pred_list.append(pred_df)

            status_rows.append(status)

        except Exception as e:
            err = repr(e)
            print(f"[ERROR] {category}: {err}")
            traceback.print_exc()

            status_rows.append({
                "dataset": "MVTec AD 2",
                "category": category,
                "data_root": str((ADAPTER_ROOT / f"{category}_folder").relative_to(ROOT)),
                "run_dir": str((RUN_ROOT / category).relative_to(ROOT)),
                "success": False,
                "fit_success": False,
                "test_success": False,
                "predict_success": False,
                "num_prediction_rows": 0,
                "elapsed_sec": round(time.time() - start, 3),
                "error": err,
            })

        finally:
            clear_gpu_cache()

    metrics = pd.concat(metrics_list, ignore_index=True) if metrics_list else pd.DataFrame()
    preds = pd.concat(pred_list, ignore_index=True) if pred_list else pd.DataFrame()
    status = pd.DataFrame(status_rows)

    metrics.to_csv(OUT_METRICS, index=False)
    preds.to_csv(OUT_PREDS, index=False)
    status.to_csv(OUT_STATUS, index=False)

    write_report(metrics, preds, status)

    print("")
    print("[DONE]", OUT_METRICS)
    print("[DONE]", OUT_PREDS)
    print("[DONE]", OUT_STATUS)
    print("[DONE]", OUT_REPORT)

    print("")
    print("===== status =====")
    print(status.to_string(index=False))

    print("")
    print("===== metrics =====")
    if metrics.empty:
        print("No metrics.")
    else:
        print(metrics.to_string(index=False))

    failed = status[status["success"] != True]
    if not failed.empty:
        raise SystemExit(f"[ERROR] Failed categories: {failed['category'].tolist()}")


if __name__ == "__main__":
    main()
