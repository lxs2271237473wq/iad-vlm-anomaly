from __future__ import annotations

from pathlib import Path
import json
import traceback
from datetime import datetime

import pandas as pd


ROOT = Path(".").resolve()

CATEGORY = "fruit_jelly"
DATA_ROOT = ROOT / "datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder"

OUT_DIR = ROOT / "results/stage14_strong_vlm_baselines"
DOC_DIR = ROOT / "docs/stage14_strong_vlm_baselines"

OUT_CSV = OUT_DIR / "stage14_d_winclip_fruit_jelly_sensitivity.csv"
OUT_JSON = OUT_DIR / "stage14_d_winclip_fruit_jelly_sensitivity_raw.json"
OUT_REPORT = DOC_DIR / "stage14_d_winclip_fruit_jelly_sensitivity_report.md"
OUT_ERROR = OUT_DIR / "stage14_d_winclip_fruit_jelly_sensitivity_errors.txt"

ENGINE_ROOT = ROOT / "runs/stage14_strong_vlm_baselines/winclip_fruit_jelly_sensitivity"

CLASS_NAMES = [
    "fruit jelly",
    "fruit_jelly",
    "jelly",
]

K_SHOTS = [0, 1, 2, 4]

SCALES_LIST = [
    (2, 3),
    (1, 2, 3),
]


def f4(x):
    if x is None or pd.isna(x) or x == "":
        return ""
    return f"{float(x):.4f}"


def flatten_metrics(metrics):
    if isinstance(metrics, list):
        if len(metrics) == 0:
            return {}
        if isinstance(metrics[0], dict):
            return dict(metrics[0])
        return {"metrics": str(metrics)}

    if isinstance(metrics, dict):
        return dict(metrics)

    return {"metrics": str(metrics)}


def run_one_config(class_name: str, k_shot: int, scales: tuple[int, ...], run_id: str):
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import WinClip

    datamodule = Folder(
        name=f"ad2_{CATEGORY}_winclip_{run_id}",
        root=str(DATA_ROOT),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/bad",
        mask_dir="ground_truth/bad",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = WinClip(
        class_name=class_name,
        k_shot=k_shot,
        scales=scales,
    )

    engine = Engine(
        default_root_dir=ENGINE_ROOT / run_id,
        logger=False,
    )

    metrics = engine.test(model=model, datamodule=datamodule)

    return metrics


def write_report(df: pd.DataFrame, errors: list[str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Stage 14-D WinCLIP fruit_jelly Sensitivity")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage tests whether the weak WinCLIP zero-shot result on AD2 fruit_jelly is caused by default configuration choices.")
    lines.append("")
    lines.append("It varies class name, k-shot value, and WinCLIP scales.")
    lines.append("")
    lines.append("## 2. Configuration Space")
    lines.append("")
    lines.append(f"- Class names: `{CLASS_NAMES}`")
    lines.append(f"- k-shot values: `{K_SHOTS}`")
    lines.append(f"- scales: `{SCALES_LIST}`")
    lines.append("")
    lines.append("## 3. Results")
    lines.append("")
    lines.append("| Rank | Status | Class name | k-shot | Scales | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Error |")
    lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---|")

    ranked = df.copy()
    ranked["rank_metric"] = pd.to_numeric(ranked.get("image_AUROC", pd.Series(dtype=float)), errors="coerce")
    ranked = ranked.sort_values("rank_metric", ascending=False, na_position="last").reset_index(drop=True)

    for i, r in ranked.iterrows():
        err = str(r.get("error", ""))
        if len(err) > 120:
            err = err[:120] + "..."
        lines.append(
            f"| {i+1} | {r.get('status', '')} | `{r.get('class_name', '')}` | "
            f"{r.get('k_shot', '')} | `{r.get('scales', '')}` | "
            f"{f4(r.get('image_AUROC', ''))} | {f4(r.get('image_F1Score', ''))} | "
            f"{f4(r.get('pixel_AUROC', ''))} | {f4(r.get('pixel_F1Score', ''))} | `{err}` |"
        )

    lines.append("")
    lines.append("## 4. Best Successful Configuration")
    lines.append("")

    success = df[df["status"] == "success"].copy()
    if success.empty:
        lines.append("No configuration ran successfully.")
    else:
        success["image_AUROC_num"] = pd.to_numeric(success["image_AUROC"], errors="coerce")
        best = success.sort_values("image_AUROC_num", ascending=False).iloc[0]

        lines.append("| Item | Value |")
        lines.append("|---|---:|")
        lines.append(f"| class_name | `{best['class_name']}` |")
        lines.append(f"| k_shot | {best['k_shot']} |")
        lines.append(f"| scales | `{best['scales']}` |")
        lines.append(f"| image AUROC | {f4(best['image_AUROC'])} |")
        lines.append(f"| image F1 | {f4(best['image_F1Score'])} |")
        lines.append(f"| pixel AUROC | {f4(best['pixel_AUROC'])} |")
        lines.append(f"| pixel F1 | {f4(best['pixel_F1Score'])} |")

        lines.append("")
        lines.append("Interpretation:")
        lines.append("")

        if float(best["image_AUROC"]) > 0.7167:
            lines.append("The best WinCLIP configuration exceeds the PatchCore fruit_jelly reference from Stage 14-C3.")
        elif float(best["image_AUROC"]) > 0.4267:
            lines.append("WinCLIP improves over the default zero-shot pilot, but still does not exceed the PatchCore fruit_jelly reference.")
        else:
            lines.append("WinCLIP remains weak even after the tested configuration changes.")

    lines.append("")
    lines.append("## 5. Errors")
    lines.append("")

    if errors:
        lines.append("Some configurations failed. See the error log for full tracebacks.")
    else:
        lines.append("No configuration failed.")

    lines.append("")
    lines.append("## 6. Output")
    lines.append("")
    lines.append(f"- Sensitivity CSV: `{OUT_CSV.relative_to(ROOT)}`")
    lines.append(f"- Raw JSON: `{OUT_JSON.relative_to(ROOT)}`")
    lines.append(f"- Error log: `{OUT_ERROR.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)

    required_paths = [
        DATA_ROOT / "train/good",
        DATA_ROOT / "test/good",
        DATA_ROOT / "test/bad",
        DATA_ROOT / "ground_truth/bad",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset paths:\n" + "\n".join(missing))

    rows = []
    raw_records = []
    errors = []

    for class_name in CLASS_NAMES:
        for k_shot in K_SHOTS:
            for scales in SCALES_LIST:
                scales_str = "_".join(str(x) for x in scales)
                class_tag = class_name.replace(" ", "_").replace("/", "_")
                run_id = f"{class_tag}_k{k_shot}_scales{scales_str}"

                print(f"[RUN] class_name={class_name} k_shot={k_shot} scales={scales}")

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "category": CATEGORY,
                    "class_name": class_name,
                    "k_shot": k_shot,
                    "scales": str(scales),
                    "run_id": run_id,
                    "status": "failed",
                    "image_AUROC": "",
                    "image_F1Score": "",
                    "pixel_AUROC": "",
                    "pixel_F1Score": "",
                    "error": "",
                }

                try:
                    metrics = run_one_config(class_name, k_shot, scales, run_id)
                    flat = flatten_metrics(metrics)

                    row["status"] = "success"
                    row["image_AUROC"] = flat.get("image_AUROC", "")
                    row["image_F1Score"] = flat.get("image_F1Score", "")
                    row["pixel_AUROC"] = flat.get("pixel_AUROC", "")
                    row["pixel_F1Score"] = flat.get("pixel_F1Score", "")
                    row["raw_metrics"] = str(flat)

                    raw_records.append({
                        "run_id": run_id,
                        "class_name": class_name,
                        "k_shot": k_shot,
                        "scales": str(scales),
                        "metrics": flat,
                    })

                    print("[OK]", flat)

                except Exception:
                    err = traceback.format_exc()
                    row["error"] = err.splitlines()[-1] if err else "unknown error"
                    errors.append(f"\n===== {run_id} =====\n{err}")
                    print("[ERROR]", row["error"])

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    OUT_JSON.write_text(
        json.dumps(raw_records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    OUT_ERROR.write_text("\n".join(errors), encoding="utf-8")

    write_report(df, errors)

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_JSON)
    print("[DONE]", OUT_ERROR)
    print("[DONE]", OUT_REPORT)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
