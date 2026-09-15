from __future__ import annotations

from pathlib import Path
from io import StringIO
import re
import pandas as pd


ROOT = Path(".").resolve()

STAGE14_COMPARE = ROOT / "results/stage14_strong_vlm_baselines/stage14_e_primary_external_baseline_comparison.csv"
STAGE15_EFFICIENTAD = ROOT / "results/stage15_modern_detector_baselines/stage15_d_efficientad_primary_fixed_budget.csv"

OUT_CSV = ROOT / "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv"
OUT_REPORT = ROOT / "docs/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison_report.md"

CATEGORIES = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]

METHOD_ORDER = [
    "WinCLIP fixed protocol",
    "full-image VLM",
    "context-aware VLM",
    "PatchCore",
    "EfficientAD-30 fixed-budget",
    "PatchCore + context VLM, LOCO",
    "PatchCore + context VLM, same-set",
]

METHOD_GROUP = {
    "WinCLIP fixed protocol": "external_vlm_baseline",
    "full-image VLM": "vlm_branch",
    "context-aware VLM": "vlm_branch",
    "PatchCore": "classical_detector",
    "EfficientAD-30 fixed-budget": "modern_detector_baseline",
    "PatchCore + context VLM, LOCO": "fusion_loco",
    "PatchCore + context VLM, same-set": "fusion_same_set",
}


def read_stage14_comparison(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8").strip()

    header = "category,method_group,method,auroc,ap,pixel_auroc,pixel_f1,note"
    if raw.startswith(header) and "\n" not in raw:
        body = raw[len(header):].strip()
        rows = re.split(r"\s+(?=(fruit_jelly|sheet_metal|vial|walnuts),)", body)
        # re.split with capturing group inserts category tokens; rebuild safely.
        rebuilt = []
        i = 0
        while i < len(rows):
            part = rows[i].strip()
            if part in CATEGORIES and i + 1 < len(rows):
                rebuilt.append(part + "," + rows[i + 1].strip())
                i += 2
            elif part:
                rebuilt.append(part)
                i += 1
            else:
                i += 1
        raw = header + "\n" + "\n".join(rebuilt) + "\n"

    return pd.read_csv(StringIO(raw))


def read_efficientad(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8").strip()

    header = "timestamp,category,status,max_epochs,train_batch_size,eval_batch_size,num_workers,precision,check_val_every_n_epoch,fit_sec,test_sec,image_AUROC,image_F1Score,pixel_AUROC,pixel_F1Score,error"
    if raw.startswith(header) and "\n" not in raw:
        body = raw[len(header):].strip()
        rows = re.split(r"\s+(?=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2},)", body)
        rows = [r.strip() for r in rows if r.strip()]
        raw = header + "\n" + "\n".join(rows) + "\n"

    df = pd.read_csv(StringIO(raw))
    return df


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    s14 = read_stage14_comparison(STAGE14_COMPARE)
    eff = read_efficientad(STAGE15_EFFICIENTAD)

    rows = []

    for _, r in s14.iterrows():
        method = str(r["method"])
        if method not in METHOD_ORDER:
            continue

        rows.append(
            {
                "category": r["category"],
                "method_group": r["method_group"],
                "method": method,
                "image_auroc": r.get("auroc", ""),
                "image_ap": r.get("ap", ""),
                "pixel_auroc": r.get("pixel_auroc", ""),
                "pixel_f1": r.get("pixel_f1", ""),
                "protocol": r.get("note", ""),
                "fairness_tag": (
                    "upper_bound_diagnostic"
                    if method == "PatchCore + context VLM, same-set"
                    else "primary_or_reference"
                ),
            }
        )

    for _, r in eff.iterrows():
        if r["status"] != "success":
            continue
        rows.append(
            {
                "category": r["category"],
                "method_group": "modern_detector_baseline",
                "method": "EfficientAD-30 fixed-budget",
                "image_auroc": r["image_AUROC"],
                "image_ap": "",
                "pixel_auroc": r["pixel_AUROC"],
                "pixel_f1": r["pixel_F1Score"],
                "protocol": "EfficientAD small, 30 epochs, train_batch_size=1, eval_batch_size=64, num_workers=16",
                "fairness_tag": "fixed_budget_detector_baseline",
            }
        )

    df = pd.DataFrame(rows)

    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["category"] = pd.Categorical(df["category"], categories=CATEGORIES, ordered=True)
    df = df.sort_values(["category", "method"]).reset_index(drop=True)

    for col in ["image_auroc", "image_ap", "pixel_auroc", "pixel_f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add per-method mean rows.
    mean_rows = []
    for method, g in df.groupby("method", observed=True):
        mean_rows.append(
            {
                "category": "MEAN",
                "method_group": METHOD_GROUP.get(str(method), ""),
                "method": str(method),
                "image_auroc": g["image_auroc"].mean(),
                "image_ap": g["image_ap"].mean(),
                "pixel_auroc": g["pixel_auroc"].mean(),
                "pixel_f1": g["pixel_f1"].mean(),
                "protocol": "mean over fruit_jelly, sheet_metal, vial, walnuts",
                "fairness_tag": "mean_summary",
            }
        )

    out = pd.concat([df, pd.DataFrame(mean_rows)], ignore_index=True)

    out["method"] = out["method"].astype(str)
    out.to_csv(OUT_CSV, index=False, lineterminator="\n")

    mean = pd.DataFrame(mean_rows)
    mean = mean.sort_values("image_auroc", ascending=False).reset_index(drop=True)
    mean["rank_by_image_auroc"] = range(1, len(mean) + 1)

    # Useful deltas.
    mean_map = dict(zip(mean["method"], mean["image_auroc"]))
    loco = mean_map.get("PatchCore + context VLM, LOCO")
    ead = mean_map.get("EfficientAD-30 fixed-budget")
    patch = mean_map.get("PatchCore")
    ctx = mean_map.get("context-aware VLM")
    winclip = mean_map.get("WinCLIP fixed protocol")

    lines = []
    lines += [
        "# Stage 15-E Primary Unified Baseline Comparison",
        "",
        "## 1. Purpose",
        "",
        "This stage merges the four-category EfficientAD-30 fixed-budget baseline with the existing Stage 14-E primary external baseline comparison.",
        "",
        "The goal is to check whether the newly added modern detector baseline changes the current research conclusion before running any 100-epoch sensitivity experiment.",
        "",
        "## 2. Included Methods",
        "",
        "- WinCLIP fixed protocol",
        "- full-image VLM",
        "- context-aware VLM",
        "- PatchCore",
        "- EfficientAD-30 fixed-budget",
        "- PatchCore + context VLM, LOCO",
        "- PatchCore + context VLM, same-set",
        "",
        "Important: `same-set` fusion is an upper-bound diagnostic and must not be overclaimed as the final fair protocol.",
        "",
        "## 3. Mean Image AUROC Ranking",
        "",
        "| Rank | Method | Mean Image AUROC | Mean Pixel AUROC | Fairness tag |",
        "|---:|---|---:|---:|---|",
    ]

    for _, r in mean.iterrows():
        lines.append(
            f"| {int(r['rank_by_image_auroc'])} | {r['method']} | "
            f"{r['image_auroc']:.4f} | "
            f"{'' if pd.isna(r['pixel_auroc']) else f'{r['pixel_auroc']:.4f}'} | "
            f"{r['fairness_tag']} |"
        )

    lines += [
        "",
        "## 4. Main Deltas",
        "",
    ]

    if loco is not None and ead is not None:
        lines.append(f"- LOCO fusion minus EfficientAD-30: `{loco - ead:+.4f}` mean image AUROC.")
    if ead is not None and patch is not None:
        lines.append(f"- EfficientAD-30 minus PatchCore: `{ead - patch:+.4f}` mean image AUROC.")
    if ead is not None and ctx is not None:
        lines.append(f"- EfficientAD-30 minus context-aware VLM: `{ead - ctx:+.4f}` mean image AUROC.")
    if ead is not None and winclip is not None:
        lines.append(f"- EfficientAD-30 minus WinCLIP fixed protocol: `{ead - winclip:+.4f}` mean image AUROC.")

    lines += [
        "",
        "## 5. Per-category Result Table",
        "",
        "| Category | Method | Image AUROC | Pixel AUROC | Fairness tag |",
        "|---|---|---:|---:|---|",
    ]

    for _, r in df.iterrows():
        lines.append(
            f"| {r['category']} | {r['method']} | "
            f"{'' if pd.isna(r['image_auroc']) else f'{r['image_auroc']:.4f}'} | "
            f"{'' if pd.isna(r['pixel_auroc']) else f'{r['pixel_auroc']:.4f}'} | "
            f"{r['fairness_tag']} |"
        )

    lines += [
        "",
        "## 6. Interpretation",
        "",
        "EfficientAD-30 is a useful modern non-VLM detector baseline, but it does not invalidate the current localization-guided VLM fusion direction.",
        "",
        "The fairer fusion result, `PatchCore + context VLM, LOCO`, should be compared against EfficientAD-30. The same-set fusion result should remain an upper-bound diagnostic.",
        "",
        "If EfficientAD-30 is close to or below the LOCO fusion mean, the next priority is not immediately a full four-category EfficientAD-100 run. A single-category 100-epoch sensitivity check is enough to test whether the 30-epoch budget severely underestimates EfficientAD.",
        "",
        "## 7. Outputs",
        "",
        f"- CSV: `{OUT_CSV.relative_to(ROOT)}`",
        f"- Report: `{OUT_REPORT.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== Mean ranking =====")
    print(mean[["rank_by_image_auroc", "method", "image_auroc", "pixel_auroc", "fairness_tag"]].to_string(index=False))


if __name__ == "__main__":
    main()
