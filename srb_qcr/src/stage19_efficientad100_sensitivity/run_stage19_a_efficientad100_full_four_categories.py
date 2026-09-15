from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(".").resolve()

BACKEND = ROOT / "experiments/stage15_modern_detector_baselines/run_stage15_d_efficientad_primary_fixed_budget.py"

OUT_DIR = ROOT / "results/stage19_efficientad100_sensitivity"
DOC_DIR = ROOT / "docs/stage19_efficientad100_sensitivity"
RUN_ROOT = ROOT / "runs/stage19_efficientad100_sensitivity/efficientad100_full_four_categories"

OUT_CSV = OUT_DIR / "stage19_a_efficientad100_full_four_categories.csv"
OUT_JSON = OUT_DIR / "stage19_a_efficientad100_full_four_categories_raw.json"
OUT_ERROR = OUT_DIR / "stage19_a_efficientad100_full_four_categories_errors.txt"
OUT_DELTA = OUT_DIR / "stage19_a_efficientad100_vs_30_delta.csv"
OUT_SUMMARY = OUT_DIR / "stage19_a_efficientad100_sensitivity_summary.csv"
OUT_REPORT = DOC_DIR / "stage19_a_efficientad100_full_four_categories_report.md"

STAGE15_30_CSV = ROOT / "results/stage15_modern_detector_baselines/stage15_d_efficientad_primary_fixed_budget.csv"
STAGE17_FJ_CSV = ROOT / "results/stage17_defensive_sensitivity/stage17_a_efficientad100_fruit_jelly.csv"

CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

METRICS = ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def load_backend():
    if not BACKEND.exists():
        raise FileNotFoundError(BACKEND)

    spec = importlib.util.spec_from_file_location("stage15_efficientad_backend", BACKEND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import backend: {BACKEND}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Redirect backend outputs to Stage19 paths.
    module.OUT_DIR = OUT_DIR
    module.DOC_DIR = DOC_DIR
    module.RUN_ROOT = RUN_ROOT
    module.OUT_CSV = OUT_CSV
    module.OUT_JSON = OUT_JSON
    module.OUT_ERROR = OUT_ERROR
    module.OUT_REPORT = OUT_REPORT

    return module


def read_existing_rows() -> list[dict]:
    if not OUT_CSV.exists():
        return []

    df = pd.read_csv(OUT_CSV)
    if df.empty:
        return []

    return df.to_dict("records")


def read_existing_raw() -> list[dict]:
    if not OUT_JSON.exists():
        return []

    try:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def read_existing_errors() -> list[str]:
    if not OUT_ERROR.exists():
        return []

    text = OUT_ERROR.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    return [text]


def success_categories(rows: list[dict]) -> set[str]:
    out = set()
    for r in rows:
        if str(r.get("status", "")) == "success":
            out.add(str(r.get("category", "")))
    return out


def seed_from_stage17_fruit_jelly(rows: list[dict], raw_records: list[dict]) -> None:
    """
    Reuse the previous fruit_jelly EfficientAD-100 result if available.
    This avoids rerunning fruit_jelly, but only if the Stage17 file is present and readable.
    """

    if "fruit_jelly" in success_categories(rows):
        return

    if not STAGE17_FJ_CSV.exists():
        print("[SEED] Stage17 fruit_jelly file not found; will run fruit_jelly in Stage19.")
        return

    try:
        df = pd.read_csv(STAGE17_FJ_CSV)
    except Exception as exc:
        print(f"[SEED-WARN] Could not read Stage17 fruit_jelly CSV: {exc}")
        return

    if df.empty:
        print("[SEED-WARN] Stage17 fruit_jelly CSV is empty.")
        return

    # Prefer a success row if available, otherwise the first row.
    if "status" in df.columns and (df["status"].astype(str) == "success").any():
        r = df[df["status"].astype(str) == "success"].iloc[0].to_dict()
    else:
        r = df.iloc[0].to_dict()

    r["category"] = "fruit_jelly"
    r["status"] = "success"
    r["max_epochs"] = 100
    r["stage19_seeded_from"] = str(STAGE17_FJ_CSV.relative_to(ROOT))

    # Make sure expected metric columns exist.
    for m in METRICS:
        r.setdefault(m, "")

    rows.append(r)
    raw_records.append(
        {
            "category": "fruit_jelly",
            "seeded_from": str(STAGE17_FJ_CSV.relative_to(ROOT)),
            "row": r,
        }
    )

    print("[SEED] Reused Stage17 fruit_jelly EfficientAD-100 result.")


def write_state(rows: list[dict], raw_records: list[dict], errors: list[str]) -> None:
    ensure_dirs()
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, lineterminator="\n")
    OUT_JSON.write_text(json.dumps(raw_records, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    OUT_ERROR.write_text("\n".join(errors), encoding="utf-8")


def numeric_metric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def build_delta_and_summary(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not STAGE15_30_CSV.exists():
        raise FileNotFoundError(f"Missing EfficientAD-30 baseline CSV: {STAGE15_30_CSV}")

    df100 = pd.DataFrame(rows)
    df30 = pd.read_csv(STAGE15_30_CSV)

    df100 = df100[df100.get("status", "").astype(str) == "success"].copy()
    df30 = df30[df30.get("status", "").astype(str) == "success"].copy()

    delta_rows = []

    for cat in CATEGORIES:
        r100 = df100[df100["category"].astype(str) == cat]
        r30 = df30[df30["category"].astype(str) == cat]

        if r100.empty or r30.empty:
            for metric in METRICS:
                delta_rows.append(
                    {
                        "category": cat,
                        "metric": metric,
                        "efficientad30_value": pd.NA,
                        "efficientad100_value": pd.NA,
                        "delta_100_minus_30": pd.NA,
                        "status": "missing_30_or_100",
                    }
                )
            continue

        a = r30.iloc[0]
        b = r100.iloc[0]

        for metric in METRICS:
            v30 = pd.to_numeric(pd.Series([a.get(metric, pd.NA)]), errors="coerce").iloc[0]
            v100 = pd.to_numeric(pd.Series([b.get(metric, pd.NA)]), errors="coerce").iloc[0]

            delta_rows.append(
                {
                    "category": cat,
                    "metric": metric,
                    "efficientad30_value": v30,
                    "efficientad100_value": v100,
                    "delta_100_minus_30": v100 - v30 if pd.notna(v30) and pd.notna(v100) else pd.NA,
                    "status": "ok" if pd.notna(v30) and pd.notna(v100) else "missing_metric",
                }
            )

    delta = pd.DataFrame(delta_rows)

    summary_rows = []
    for metric in METRICS:
        sub = delta[(delta["metric"] == metric) & (delta["status"] == "ok")].copy()
        if sub.empty:
            summary_rows.append(
                {
                    "metric": metric,
                    "num_categories": 0,
                    "efficientad30_mean": pd.NA,
                    "efficientad100_mean": pd.NA,
                    "mean_delta_100_minus_30": pd.NA,
                    "num_100_better": 0,
                    "num_100_worse": 0,
                    "num_equal": 0,
                }
            )
            continue

        summary_rows.append(
            {
                "metric": metric,
                "num_categories": int(sub["category"].nunique()),
                "efficientad30_mean": float(sub["efficientad30_value"].mean()),
                "efficientad100_mean": float(sub["efficientad100_value"].mean()),
                "mean_delta_100_minus_30": float(sub["delta_100_minus_30"].mean()),
                "num_100_better": int((sub["delta_100_minus_30"] > 0).sum()),
                "num_100_worse": int((sub["delta_100_minus_30"] < 0).sum()),
                "num_equal": int((sub["delta_100_minus_30"] == 0).sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return delta, summary


def fmt(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.4f}"


def signed(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):+.4f}"


def write_report(rows: list[dict], delta: pd.DataFrame, summary: pd.DataFrame, args) -> None:
    df = pd.DataFrame(rows)

    success = df[df.get("status", "").astype(str) == "success"].copy() if not df.empty else pd.DataFrame()
    success_cats = sorted(success["category"].astype(str).unique().tolist()) if not success.empty else []

    image_summary = summary[summary["metric"] == "image_AUROC"]
    if not image_summary.empty:
        im = image_summary.iloc[0]
        mean_delta = im["mean_delta_100_minus_30"]
        n_better = im["num_100_better"]
        n_worse = im["num_100_worse"]
    else:
        mean_delta = pd.NA
        n_better = 0
        n_worse = 0

    if pd.notna(mean_delta) and mean_delta > 0:
        decision = "EfficientAD-100 improves image AUROC over EfficientAD-30 on average."
    elif pd.notna(mean_delta) and mean_delta <= 0:
        decision = "EfficientAD-100 does not improve image AUROC over EfficientAD-30 on average."
    else:
        decision = "EfficientAD-100 sensitivity is incomplete."

    lines = [
        "# Stage 19-A EfficientAD-100 Full Four-category Sensitivity",
        "",
        "## Purpose",
        "",
        "Run EfficientAD with a 100-epoch budget on the AD2 four-category setting and compare against the existing EfficientAD-30 fixed-budget baseline.",
        "",
        "## Config",
        "",
        f"- categories requested: `{'; '.join(args.categories)}`",
        f"- max_epochs: `{args.max_epochs}`",
        f"- eval_batch_size: `{args.eval_batch_size}`",
        f"- num_workers: `{args.num_workers}`",
        f"- precision: `{args.precision}`",
        f"- seeded fruit_jelly from Stage17: `{args.seed_from_stage17_fruit_jelly}`",
        "",
        "## Completion",
        "",
        f"- successful categories: `{len(success_cats)}/4`",
        f"- success list: `{'; '.join(success_cats)}`",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "## EfficientAD-100 vs EfficientAD-30 summary",
        "",
        "| Metric | Categories | EfficientAD-30 mean | EfficientAD-100 mean | Delta | 100 better | 100 worse |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['metric']} | {int(r['num_categories'])} | "
            f"{fmt(r['efficientad30_mean'])} | {fmt(r['efficientad100_mean'])} | "
            f"{signed(r['mean_delta_100_minus_30'])} | "
            f"{int(r['num_100_better'])} | {int(r['num_100_worse'])} |"
        )

    lines += [
        "",
        "## Per-category delta",
        "",
        "| Category | Metric | EAD-30 | EAD-100 | Delta |",
        "|---|---|---:|---:|---:|",
    ]

    for _, r in delta.iterrows():
        lines.append(
            f"| {r['category']} | {r['metric']} | "
            f"{fmt(r['efficientad30_value'])} | {fmt(r['efficientad100_value'])} | "
            f"{signed(r['delta_100_minus_30'])} |"
        )

    lines += [
        "",
        "## Paper interpretation rule",
        "",
        "- If EfficientAD-100 does not improve image AUROC on average, keep EfficientAD-30 as a fixed-budget baseline and cite Stage19 as sensitivity analysis.",
        "- If EfficientAD-100 improves strongly, update the system baseline table and weaken any comparison against EfficientAD.",
        "- Pixel AUROC improvements are auxiliary unless the paper makes pixel-level localization claims.",
        "",
        "## Outputs",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_DELTA.relative_to(ROOT)}`",
        f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
        f"- `{OUT_JSON.relative_to(ROOT)}`",
        f"- `{OUT_ERROR.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--enable_progress_bar", action="store_true")
    parser.add_argument("--reset_outputs", action="store_true")
    parser.add_argument("--skip_success", action="store_true")
    parser.add_argument("--seed_from_stage17_fruit_jelly", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.max_epochs != 100:
        raise ValueError("Stage19 is reserved for EfficientAD-100. Use --max_epochs 100.")

    if args.reset_outputs:
        for p in [OUT_CSV, OUT_JSON, OUT_ERROR, OUT_DELTA, OUT_SUMMARY, OUT_REPORT]:
            if p.exists():
                p.unlink()

    backend = load_backend()

    rows = read_existing_rows()
    raw_records = read_existing_raw()
    errors = read_existing_errors()

    if args.seed_from_stage17_fruit_jelly:
        seed_from_stage17_fruit_jelly(rows, raw_records)
        write_state(rows, raw_records, errors)

    backend_args = SimpleNamespace(
        categories=args.categories,
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

    print("[STAGE19 ARGS]", vars(args), flush=True)

    backend.require_cuda()

    for category in args.categories:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")

        if args.skip_success and category in success_categories(rows):
            print(f"[SKIP] {category}: already success", flush=True)
            continue

        backend.run_one(category, backend_args, rows, raw_records, errors)

        try:
            delta, summary = build_delta_and_summary(rows)
            delta.to_csv(OUT_DELTA, index=False, lineterminator="\n")
            summary.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")
            write_report(rows, delta, summary, args)
        except Exception as exc:
            print(f"[WARN] Could not build partial Stage19 comparison yet: {exc}", flush=True)

    write_state(rows, raw_records, errors)

    delta, summary = build_delta_and_summary(rows)
    delta.to_csv(OUT_DELTA, index=False, lineterminator="\n")
    summary.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")
    write_report(rows, delta, summary, args)

    print("[DONE]", OUT_CSV, flush=True)
    print("[DONE]", OUT_DELTA, flush=True)
    print("[DONE]", OUT_SUMMARY, flush=True)
    print("[DONE]", OUT_REPORT, flush=True)
    print()
    print("===== EfficientAD-100 rows =====")
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("===== Delta summary =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
