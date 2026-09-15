from __future__ import annotations

from pathlib import Path
import json
import traceback
from datetime import datetime

import pandas as pd


ROOT = Path(".").resolve()

CATEGORY = "fruit_jelly"
CLASS_NAME = "fruit jelly"

DATA_ROOT = ROOT / "datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder"

OUT_DIR = ROOT / "results/stage14_strong_vlm_baselines"
DOC_DIR = ROOT / "docs/stage14_strong_vlm_baselines"

OUT_JSON = OUT_DIR / "stage14_c2_winclip_fruit_jelly_metrics.json"
OUT_CSV = OUT_DIR / "stage14_c2_winclip_fruit_jelly_metrics.csv"
OUT_REPORT = DOC_DIR / "stage14_c2_winclip_fruit_jelly_pilot_report.md"
OUT_ERROR = OUT_DIR / "stage14_c2_winclip_fruit_jelly_error.txt"

ENGINE_ROOT = ROOT / "runs/stage14_strong_vlm_baselines/winclip_fruit_jelly_pilot"


def write_report(
    *,
    status: str,
    metrics: object | None,
    error: str | None,
    datamodule_config: dict,
    model_config: dict,
) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Stage 14-C2 WinCLIP fruit_jelly Pilot")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage runs a one-category WinCLIP pilot on MVTec AD 2 fruit_jelly.")
    lines.append("")
    lines.append("The purpose is to introduce an external vision-language anomaly detection baseline, instead of comparing only with full-image VLM or PatchCore.")
    lines.append("")
    lines.append("## 2. Dataset")
    lines.append("")
    lines.append(f"- Category: `{CATEGORY}`")
    lines.append(f"- Class name for WinCLIP: `{CLASS_NAME}`")
    lines.append(f"- Data root: `{DATA_ROOT.relative_to(ROOT)}`")
    lines.append(f"- Train normal dir: `{datamodule_config['normal_dir']}`")
    lines.append(f"- Test normal dir: `{datamodule_config['normal_test_dir']}`")
    lines.append(f"- Test abnormal dir: `{datamodule_config['abnormal_dir']}`")
    lines.append(f"- Mask dir: `{datamodule_config['mask_dir']}`")
    lines.append("")
    lines.append("## 3. Model")
    lines.append("")
    lines.append(f"- Model: `WinClip`")
    lines.append(f"- k-shot: `{model_config['k_shot']}`")
    lines.append(f"- scales: `{model_config['scales']}`")
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
        lines.append("The WinCLIP pilot ran successfully on fruit_jelly. Next step is to parse the reported metrics and compare them with PatchCore, our context VLM score, and PatchCore+context fusion on the same category.")
    else:
        lines.append("The WinCLIP pilot failed. The next step is to inspect the error and adjust the Anomalib Folder or Engine invocation.")

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

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import WinClip

    required_paths = [
        DATA_ROOT / "train/good",
        DATA_ROOT / "test/good",
        DATA_ROOT / "test/bad",
        DATA_ROOT / "ground_truth/bad",
    ]

    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset paths:\n" + "\n".join(missing))

    datamodule_config = {
        "name": f"ad2_{CATEGORY}_winclip",
        "root": str(DATA_ROOT),
        "normal_dir": "train/good",
        "normal_test_dir": "test/good",
        "abnormal_dir": "test/bad",
        "mask_dir": "ground_truth/bad",
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_workers": 0,
    }

    model_config = {
        "class_name": CLASS_NAME,
        "k_shot": 0,
        "scales": (2, 3),
    }

    metrics = None
    error = None

    try:
        datamodule = Folder(**datamodule_config)
        model = WinClip(**model_config)
        engine = Engine(default_root_dir=ENGINE_ROOT, logger=False)

        # WinCLIP is a zero-shot / few-shot method. For k_shot=0, testing directly is expected.
        metrics = engine.test(model=model, datamodule=datamodule)

        OUT_JSON.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Also save a flattened CSV when possible.
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
            datamodule_config=datamodule_config,
            model_config=model_config,
        )

        print("[DONE]", OUT_JSON)
        print("[DONE]", OUT_CSV)
        print("[DONE]", OUT_REPORT)
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))

    except Exception:
        error = traceback.format_exc()
        OUT_ERROR.write_text(error, encoding="utf-8")

        write_report(
            status="failed",
            metrics=None,
            error=error,
            datamodule_config=datamodule_config,
            model_config=model_config,
        )

        print("[ERROR] WinCLIP fruit_jelly pilot failed.")
        print(error)
        print("[REPORT]", OUT_REPORT)
        raise


if __name__ == "__main__":
    main()
