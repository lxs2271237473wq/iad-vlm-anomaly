from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from io import StringIO
import argparse
import importlib.util
import json
import re
import sys
import pandas as pd


ROOT = Path(".").resolve()

STAGE15_SCRIPT = ROOT / "experiments/stage15_modern_detector_baselines/run_stage15_d_efficientad_primary_fixed_budget.py"
STAGE15_30_CSV = ROOT / "results/stage15_modern_detector_baselines/stage15_d_efficientad_primary_fixed_budget.csv"

OUT_DIR = ROOT / "results/stage17_defensive_sensitivity"
DOC_DIR = ROOT / "docs/stage17_defensive_sensitivity"
RUN_ROOT = ROOT / "runs/stage17_defensive_sensitivity/efficientad100_fruit_jelly"

OUT_CSV = OUT_DIR / "stage17_a_efficientad100_fruit_jelly.csv"
OUT_JSON = OUT_DIR / "stage17_a_efficientad100_fruit_jelly_raw.json"
OUT_ERROR = OUT_DIR / "stage17_a_efficientad100_fruit_jelly_errors.txt"
OUT_TRAINING_REPORT = DOC_DIR / "stage17_a_efficientad100_fruit_jelly_training_report.md"
OUT_DELTA = OUT_DIR / "stage17_a_efficientad100_vs_30_delta.csv"
OUT_REPORT = DOC_DIR / "stage17_a_efficientad100_fruit_jelly_sensitivity_report.md"


def read_csv_robust(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8").strip()
    if "\n" not in raw and raw.startswith("timestamp,category,status,"):
        header = (
            "timestamp,category,status,max_epochs,train_batch_size,eval_batch_size,num_workers,"
            "precision,check_val_every_n_epoch,fit_sec,test_sec,image_AUROC,image_F1Score,"
            "pixel_AUROC,pixel_F1Score,error"
        )
        if raw.startswith(header):
            body = raw[len(header):].strip()
            rows = re.split(r"\s+(?=\d{4}-\d{2}-\d{2}T)", body)
            rows = [r.strip() for r in rows if r.strip()]
            raw = header + "\n" + "\n".join(rows) + "\n"
    df = pd.read_csv(StringIO(raw))
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV line breaks first.")
    return df


def load_stage15_module():
    if not STAGE15_SCRIPT.exists():
        raise FileNotFoundError(STAGE15_SCRIPT)

    spec = importlib.util.spec_from_file_location("stage15_efficientad_fixed_budget", STAGE15_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {STAGE15_SCRIPT}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Redirect all Stage 15 output globals to Stage 17 paths.
    mod.OUT_DIR = OUT_DIR
    mod.DOC_DIR = DOC_DIR
    mod.RUN_ROOT = RUN_ROOT
    mod.OUT_CSV = OUT_CSV
    mod.OUT_JSON = OUT_JSON
    mod.OUT_ERROR = OUT_ERROR
    mod.OUT_REPORT = OUT_TRAINING_REPORT

    return mod


def get_metric(df: pd.DataFrame, category: str, metric: str):
    sub = df[(df["category"] == category) & (df["status"] == "success")].copy()
    if sub.empty or metric not in sub.columns:
        return None
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    return float(sub.iloc[-1][metric])


def build_delta_and_report(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    current = read_csv_robust(OUT_CSV)
    previous = read_csv_robust(STAGE15_30_CSV) if STAGE15_30_CSV.exists() else pd.DataFrame()

    metrics = ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]

    rows = []
    for metric in metrics:
        v30 = get_metric(previous, "fruit_jelly", metric) if not previous.empty else None
        v100 = get_metric(current, "fruit_jelly", metric)
        delta = None if v30 is None or v100 is None else v100 - v30

        rows.append(
            {
                "category": "fruit_jelly",
                "metric": metric,
                "efficientad30_value": v30,
                "efficientad100_value": v100,
                "delta_100_minus_30": delta,
            }
        )

    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(OUT_DELTA, index=False, lineterminator="\n")

    image_delta = delta_df[delta_df["metric"] == "image_AUROC"]["delta_100_minus_30"].iloc[0]
    image_delta = None if pd.isna(image_delta) else float(image_delta)

    if image_delta is None:
        decision = "inconclusive"
        interpretation = "Could not compute image_AUROC delta because either the 30-epoch or 100-epoch value is missing."
        next_action = "Inspect CSV outputs before making a decision."
    elif image_delta >= 0.05:
        decision = "efficientad30_likely_underestimates"
        interpretation = "EfficientAD-100 improves image AUROC by at least +0.05; four-category 100-epoch EfficientAD may be needed for a stronger defense."
        next_action = "Consider running EfficientAD-100 on all four primary categories."
    elif image_delta >= 0.03:
        decision = "moderate_epoch_sensitivity"
        interpretation = "EfficientAD-100 improves image AUROC by +0.03 to +0.05; report EfficientAD-30 as fixed-budget and consider one more category sensitivity check."
        next_action = "Do not immediately run all four categories unless writing needs stronger baseline defense."
    else:
        decision = "efficientad30_not_severely_underestimating"
        interpretation = "EfficientAD-100 does not substantially improve image AUROC over EfficientAD-30 on fruit_jelly."
        next_action = "Keep EfficientAD-30 as fixed-budget baseline and cite this sensitivity check defensively."

    lines = []
    lines += [
        "# Stage 17-A EfficientAD-100 Fruit Jelly Sensitivity",
        "",
        "## 1. Purpose",
        "",
        "This stage checks whether the Stage 15 EfficientAD-30 fixed-budget baseline severely underestimates EfficientAD.",
        "",
        "Only `fruit_jelly` is tested at 100 epochs. This is a defensive sensitivity check, not a new main baseline sweep.",
        "",
        "## 2. Configuration",
        "",
        f"- category: `fruit_jelly`",
        f"- max_epochs: `{args.max_epochs}`",
        f"- eval_batch_size: `{args.eval_batch_size}`",
        f"- num_workers: `{args.num_workers}`",
        f"- check_val_every_n_epoch: `{args.check_val_every_n_epoch}`",
        f"- precision: `{args.precision}`",
        f"- model_size: `{args.model_size}`",
        "- train_batch_size: `1`",
        "",
        "## 3. 100 Epoch vs 30 Epoch",
        "",
        "| Metric | EfficientAD-30 | EfficientAD-100 | Delta 100-30 |",
        "|---|---:|---:|---:|",
    ]

    for _, r in delta_df.iterrows():
        v30 = "" if pd.isna(r["efficientad30_value"]) else f"{float(r['efficientad30_value']):.4f}"
        v100 = "" if pd.isna(r["efficientad100_value"]) else f"{float(r['efficientad100_value']):.4f}"
        d = "" if pd.isna(r["delta_100_minus_30"]) else f"{float(r['delta_100_minus_30']):+.4f}"
        lines.append(f"| {r['metric']} | {v30} | {v100} | {d} |")

    lines += [
        "",
        "## 4. Decision",
        "",
        f"- decision: `{decision}`",
        f"- interpretation: {interpretation}",
        f"- next_action: {next_action}",
        "",
        "## 5. Paper Usage",
        "",
        "Use this result only as baseline-budget sensitivity evidence.",
        "",
        "Do not claim full EfficientAD defeat unless a full-budget multi-category EfficientAD sweep is run.",
        "",
        "Safe wording:",
        "",
        "```text",
        "We report EfficientAD under a fixed 30-epoch budget and include a 100-epoch fruit_jelly sensitivity check to assess whether the fixed budget severely underestimates EfficientAD.",
        "```",
        "",
        "## 6. Outputs",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_DELTA.relative_to(ROOT)}`",
        f"- `{OUT_REPORT.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_DELTA)
    print("[DONE]", OUT_REPORT)
    print()
    print(delta_df.to_string(index=False))
    print()
    print("[DECISION]", decision)
    print("[INTERPRETATION]", interpretation)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=20)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--enable_progress_bar", action="store_true")
    parser.add_argument("--reset_outputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    if args.reset_outputs:
        for p in [OUT_CSV, OUT_JSON, OUT_ERROR, OUT_TRAINING_REPORT, OUT_DELTA, OUT_REPORT]:
            if p.exists():
                p.unlink()

    mod = load_stage15_module()

    stage15_args = SimpleNamespace(
        categories=["fruit_jelly"],
        max_epochs=args.max_epochs,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=args.precision,
        model_size=args.model_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        enable_progress_bar=args.enable_progress_bar,
        reset_outputs=args.reset_outputs,
    )

    print("[STAGE17-A ARGS]", vars(stage15_args), flush=True)

    mod.require_cuda()

    rows = []
    raw_records = []
    errors = []

    mod.run_one("fruit_jelly", stage15_args, rows, raw_records, errors)
    mod.write_report(rows, stage15_args)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False), flush=True)

    if df.empty or not (df["status"] == "success").any():
        print("[ERROR] EfficientAD-100 fruit_jelly did not finish successfully.", flush=True)
        sys.exit(1)

    build_delta_and_report(args)


if __name__ == "__main__":
    main()
