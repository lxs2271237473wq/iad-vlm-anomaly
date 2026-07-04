from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(".").resolve()

OUT_DIR = ROOT / "results/stage15_modern_detector_baselines"
DOC_DIR = ROOT / "docs/stage15_modern_detector_baselines"
RUN_ROOT = ROOT / "runs/stage15_modern_detector_baselines/efficientad_primary_fixed_budget"

OUT_CSV = OUT_DIR / "stage15_d_efficientad_primary_fixed_budget.csv"
OUT_JSON = OUT_DIR / "stage15_d_efficientad_primary_fixed_budget_raw.json"
OUT_ERROR = OUT_DIR / "stage15_d_efficientad_primary_fixed_budget_errors.txt"
OUT_REPORT = DOC_DIR / "stage15_d_efficientad_primary_fixed_budget_report.md"

IMAGENETTE_DIR = ROOT / "datasets/imagenette"

CATEGORY_ROOTS = {
    "fruit_jelly": ROOT / "datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder",
    "sheet_metal": ROOT / "datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder",
    "vial": ROOT / "datasets/MVTec_AD_2_anomalib_all/vial_folder",
    "walnuts": ROOT / "datasets/MVTec_AD_2_anomalib_all/walnuts_folder",
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def elapsed(t0: float) -> float:
    return round(time.time() - t0, 3)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def write_state(rows, raw_records, errors) -> None:
    ensure_dirs()
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(
        json.dumps(raw_records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    OUT_ERROR.write_text("\n".join(errors), encoding="utf-8")


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing CPU fallback.")

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    props = torch.cuda.get_device_properties(0)
    print("===== CUDA CHECK =====", flush=True)
    print("cuda:", torch.cuda.is_available(), flush=True)
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    print("vram_gb:", round(props.total_memory / 1024**3, 2), flush=True)
    print("torch_cuda_version:", torch.version.cuda, flush=True)
    print("======================", flush=True)


def check_paths(category: str, data_root: Path) -> None:
    required = [
        data_root / "train/good",
        data_root / "test/good",
        data_root / "test/bad",
        data_root / "ground_truth/bad",
        IMAGENETTE_DIR,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{category} missing required paths:\n" + "\n".join(missing))


def flatten_metrics(metrics) -> dict:
    if isinstance(metrics, list):
        if metrics and isinstance(metrics[0], dict):
            return dict(metrics[0])
        return {"raw_metrics": str(metrics)}
    if isinstance(metrics, dict):
        return dict(metrics)
    return {"raw_metrics": str(metrics)}


def pick_metric(flat: dict, names: list[str]):
    for name in names:
        if name in flat:
            return flat[name]
    return ""


def write_report(rows, args) -> None:
    df = pd.DataFrame(rows)

    lines = []
    lines.append("# Stage 15-D EfficientAD Fixed-Budget Baseline")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This script runs a robust fixed-budget EfficientAD baseline and writes status to CSV before and after each major phase.")
    lines.append("")
    lines.append("EfficientAD is only a modern non-VLM detector baseline. It is not the proposed method.")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    for k, v in vars(args).items():
        lines.append(f"- {k}: `{v}`")
    lines.append("- train_batch_size: `1`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Category | Status | Fit sec | Test sec | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for _, r in df.iterrows():
        err = str(r.get("error", ""))
        if len(err) > 120:
            err = err[:120] + "..."
        lines.append(
            f"| {r.get('category','')} | {r.get('status','')} | "
            f"{r.get('fit_sec','')} | {r.get('test_sec','')} | "
            f"{r.get('image_AUROC','')} | {r.get('image_F1Score','')} | "
            f"{r.get('pixel_AUROC','')} | {r.get('pixel_F1Score','')} | `{err}` |"
        )

    success = df[df["status"] == "success"].copy() if "status" in df else pd.DataFrame()
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    if success.empty:
        lines.append("- successful_categories: `0`")
    else:
        for col in ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]:
            success[col] = pd.to_numeric(success[col], errors="coerce")
        lines.append(f"- successful_categories: `{len(success)}`")
        lines.append(f"- mean_image_AUROC: `{success['image_AUROC'].mean():.4f}`")
        lines.append(f"- mean_image_F1Score: `{success['image_F1Score'].mean():.4f}`")
        lines.append(f"- mean_pixel_AUROC: `{success['pixel_AUROC'].mean():.4f}`")
        lines.append(f"- mean_pixel_F1Score: `{success['pixel_F1Score'].mean():.4f}`")

    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append("Anomalib EfficientAD forces train_batch_size=1 and performs validation quantile/metric computation. RTX 4090 utilization can therefore remain low for this baseline.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_one(category: str, args, rows, raw_records, errors) -> None:
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    data_root = CATEGORY_ROOTS[category]
    check_paths(category, data_root)

    row = {
        "timestamp": now(),
        "category": category,
        "status": "started",
        "max_epochs": args.max_epochs,
        "train_batch_size": 1,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "fit_sec": "",
        "test_sec": "",
        "image_AUROC": "",
        "image_F1Score": "",
        "pixel_AUROC": "",
        "pixel_F1Score": "",
        "error": "",
    }
    rows.append(row)
    write_state(rows, raw_records, errors)

    print("=" * 80, flush=True)
    print(f"[RUN] category={category}", flush=True)
    print("=" * 80, flush=True)

    try:
        print("[BUILD] datamodule", flush=True)
        datamodule = Folder(
            name=f"ad2_{category}_efficientad_fixed_budget",
            root=str(data_root),
            normal_dir="train/good",
            normal_test_dir="test/good",
            abnormal_dir="test/bad",
            mask_dir="ground_truth/bad",
            train_batch_size=1,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )

        print("[BUILD] model", flush=True)
        model = EfficientAd(
            imagenet_dir=IMAGENETTE_DIR,
            model_size=args.model_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        engine_kwargs = dict(
            default_root_dir=RUN_ROOT / category,
            logger=False,
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=1,
            precision=args.precision,
        )

        optional_kwargs = dict(
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            num_sanity_val_steps=0,
            enable_progress_bar=args.enable_progress_bar,
            enable_checkpointing=True,
        )

        try:
            print("[BUILD] engine with optional trainer kwargs", flush=True)
            engine = Engine(**engine_kwargs, **optional_kwargs)
            print("[ENGINE] optional trainer kwargs accepted", flush=True)
        except TypeError as exc:
            print("[WARN] Engine rejected optional trainer kwargs:", repr(exc), flush=True)
            print("[BUILD] engine without optional trainer kwargs", flush=True)
            engine = Engine(**engine_kwargs)

        row["status"] = "fit_running"
        write_state(rows, raw_records, errors)

        t_fit = time.time()
        print(f"[FIT-START] category={category}", flush=True)
        engine.fit(model=model, datamodule=datamodule)
        row["fit_sec"] = elapsed(t_fit)
        row["status"] = "fit_done"
        write_state(rows, raw_records, errors)
        print(f"[FIT-DONE] category={category} fit_sec={row['fit_sec']}", flush=True)

        row["status"] = "test_running"
        write_state(rows, raw_records, errors)

        t_test = time.time()
        print(f"[TEST-START] category={category}", flush=True)
        metrics = engine.test(model=model, datamodule=datamodule)
        row["test_sec"] = elapsed(t_test)

        flat = flatten_metrics(metrics)

        row["image_AUROC"] = pick_metric(flat, ["image_AUROC", "image_auroc", "AUROC"])
        row["image_F1Score"] = pick_metric(flat, ["image_F1Score", "image_f1score", "F1Score"])
        row["pixel_AUROC"] = pick_metric(flat, ["pixel_AUROC", "pixel_auroc"])
        row["pixel_F1Score"] = pick_metric(flat, ["pixel_F1Score", "pixel_f1score"])

        row["status"] = "success"
        raw_records.append({"category": category, "metrics": flat, "raw": metrics})
        write_state(rows, raw_records, errors)

        print(f"[OK] category={category}", flush=True)
        print(flat, flush=True)

    except BaseException as exc:
        err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        row["status"] = "failed"
        row["error"] = err.splitlines()[-1] if err else repr(exc)
        errors.append(f"\n===== {category} =====\n{err}")
        raw_records.append({"category": category, "error": err})
        write_state(rows, raw_records, errors)
        print(f"[ERROR] category={category}", flush=True)
        print(err, flush=True)

    finally:
        try:
            import torch
            print("[CUDA] allocated_mb:", round(torch.cuda.memory_allocated() / 1024**2, 2), flush=True)
            print("[CUDA] reserved_mb:", round(torch.cuda.memory_reserved() / 1024**2, 2), flush=True)
            torch.cuda.empty_cache()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_ROOTS.keys()))
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--enable_progress_bar", action="store_true")
    parser.add_argument("--reset_outputs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    if args.reset_outputs:
        for p in [OUT_CSV, OUT_JSON, OUT_ERROR, OUT_REPORT]:
            if p.exists():
                p.unlink()

    print("[ARGS]", vars(args), flush=True)
    require_cuda()

    rows = []
    raw_records = []
    errors = []

    for category in args.categories:
        if category not in CATEGORY_ROOTS:
            raise ValueError(f"Unknown category: {category}")
        run_one(category, args, rows, raw_records, errors)
        write_report(rows, args)

    write_report(rows, args)
    print("[DONE]", OUT_CSV, flush=True)
    print("[DONE]", OUT_REPORT, flush=True)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False), flush=True)

    if df.empty or not (df["status"] == "success").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
