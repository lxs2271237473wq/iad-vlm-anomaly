from __future__ import annotations

from pathlib import Path
import json
import traceback
from datetime import datetime

import pandas as pd


ROOT = Path(".").resolve()

CATEGORY = "fruit_jelly"
DATA_ROOT = ROOT / "datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder"
IMAGENETTE_DIR = ROOT / "datasets/imagenette"

OUT_DIR = ROOT / "results/stage15_modern_detector_baselines"
DOC_DIR = ROOT / "docs/stage15_modern_detector_baselines"

OUT_JSON = OUT_DIR / "stage15_b_efficientad_fruit_jelly_metrics.json"
OUT_CSV = OUT_DIR / "stage15_b_efficientad_fruit_jelly_metrics.csv"
OUT_REPORT = DOC_DIR / "stage15_b_efficientad_fruit_jelly_pilot_report.md"
OUT_ERROR = OUT_DIR / "stage15_b_efficientad_fruit_jelly_error.txt"

ENGINE_ROOT = ROOT / "runs/stage15_modern_detector_baselines/efficientad_fruit_jelly_pilot"

MAX_EPOCHS = 20


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


def write_report(
    *,
    status: str,
    metrics: object | None,
    error: str | None,
    config: dict,
) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Stage 15-B EfficientAD fruit_jelly Pilot")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage runs a one-category EfficientAD pilot on MVTec AD 2 fruit_jelly.")
    lines.append("")
    lines.append("The purpose is to introduce a modern non-VLM anomaly detector baseline, instead of relying only on PatchCore.")
    lines.append("")
    lines.append("## 2. Dataset")
    lines.append("")
    lines.append(f"- Category: `{CATEGORY}`")
    lines.append(f"- Data root: `{DATA_ROOT.relative_to(ROOT)}`")
    lines.append("- Train normal dir: `train/good`")
    lines.append("- Test normal dir: `test/good`")
    lines.append("- Test abnormal dir: `test/bad`")
    lines.append("- Mask dir: `ground_truth/bad`")
    lines.append("")
    lines.append("## 3. Model and Training Config")
    lines.append("")
    for k, v in config.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## 4. Status")
    lines.append("")
    lines.append(f"- Status: `{status}`")
    lines.append(f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## 5. Metrics")
    lines.append("")

    if metrics is None:
        lines.append("No metrics were produced.")
    else:
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
        lines.append("```")

    lines.append("")
    lines.append("## 6. Error")
    lines.append("")

    if error:
        lines.append("```text")
        lines.append(error)
        lines.append("```")
    else:
        lines.append("No error.")

    lines.append("")
    lines.append("## 7. Interpretation")
    lines.append("")

    if status == "success":
        lines.append("EfficientAD successfully completed fit and test on fruit_jelly. The next step is to compare it against PatchCore, WinCLIP, context-aware VLM, and PatchCore+context fusion on the same category.")
    else:
        lines.append("EfficientAD failed during fit or test. The next step is to inspect the error and fix the training configuration or required auxiliary dataset path.")

    lines.append("")
    lines.append("## 8. Output Files")
    lines.append("")
    lines.append(f"- Metrics JSON: `{OUT_JSON.relative_to(ROOT)}`")
    lines.append(f"- Metrics CSV: `{OUT_CSV.relative_to(ROOT)}`")
    lines.append(f"- Error log: `{OUT_ERROR.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append(f"- Engine output root: `{ENGINE_ROOT.relative_to(ROOT)}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
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

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    config = {
        "model": "EfficientAd",
        "category": CATEGORY,
        "imagenet_dir": str(IMAGENETTE_DIR),
        "model_size": 'small',
        "max_epochs": MAX_EPOCHS,
        "train_batch_size": 1,
        "eval_batch_size": 8,
        "num_workers": 0,
        "lr": 0.0001,
        "weight_decay": 0.00001,
    }

    try:
        datamodule = Folder(
            name=f"ad2_{CATEGORY}_efficientad",
            root=str(DATA_ROOT),
            normal_dir="train/good",
            normal_test_dir="test/good",
            abnormal_dir="test/bad",
            mask_dir="ground_truth/bad",
            train_batch_size=config["train_batch_size"],
            eval_batch_size=config["eval_batch_size"],
            num_workers=config["num_workers"],
        )

        model = EfficientAd(
            imagenet_dir=IMAGENETTE_DIR,
            model_size=config["model_size"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        engine = Engine(
            default_root_dir=ENGINE_ROOT,
            logger=False,
            max_epochs=MAX_EPOCHS,
        )

        print("[FIT] EfficientAD fruit_jelly pilot")
        engine.fit(model=model, datamodule=datamodule)

        print("[TEST] EfficientAD fruit_jelly pilot")
        metrics = engine.test(model=model, datamodule=datamodule)
        flat = flatten_metrics(metrics)

        OUT_JSON.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        if isinstance(metrics, list):
            pd.DataFrame(metrics).to_csv(OUT_CSV, index=False)
        elif isinstance(metrics, dict):
            pd.DataFrame([metrics]).to_csv(OUT_CSV, index=False)
        else:
            pd.DataFrame([{"metrics": str(metrics)}]).to_csv(OUT_CSV, index=False)

        write_report(
            status="success",
            metrics=metrics,
            error=None,
            config=config,
        )

        print("[DONE]", OUT_JSON)
        print("[DONE]", OUT_CSV)
        print("[DONE]", OUT_REPORT)
        print(json.dumps(flat, indent=2, ensure_ascii=False, default=str))

    except Exception:
        error = traceback.format_exc()
        OUT_ERROR.write_text(error, encoding="utf-8")

        write_report(
            status="failed",
            metrics=None,
            error=error,
            config=config,
        )

        print("[ERROR] EfficientAD fruit_jelly pilot failed.")
        print(error)
        print("[REPORT]", OUT_REPORT)
        raise


if __name__ == "__main__":
    main()
