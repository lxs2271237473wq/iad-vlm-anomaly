from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

BASELINE_SCRIPT = (
    ROOT
    / "scripts/legacy_starter"
    / "run_mvtec_baselines.py"
)

CONFIG_PATH = (
    ROOT
    / "configs/first_baselines.yaml"
)

SUMMARY_PATH = (
    ROOT
    / "results/baselines"
    / "patchcore_mvtec_summary.csv"
)

CHECKPOINT_ROOT = (
    ROOT
    / "runs/baselines/patchcore"
    / "MVTecAD/bottle/Patchcore/MVTecAD_bottle"
)

OUT_DIR = ROOT / "results/stage22_selective_qcr"
DOC_DIR = ROOT / "docs/stage22_selective_qcr"

OUT_RESULTS = (
    OUT_DIR
    / "stage22_d4a_bottle_checkpoint_probe.csv"
)

OUT_SELECTION = (
    OUT_DIR
    / "stage22_d4a_bottle_checkpoint_selection.json"
)

OUT_REPORT = (
    DOC_DIR
    / "stage22_d4a_bottle_checkpoint_probe.md"
)

CATEGORY = "bottle"
MATCH_TOLERANCE = 0.002


def load_baseline_module():
    if not BASELINE_SCRIPT.exists():
        raise FileNotFoundError(BASELINE_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "legacy_mvtec_baseline",
        BASELINE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {BASELINE_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def version_number(path: Path) -> int:
    match = re.search(
        r"/v(\d+)/",
        str(path).replace("\\", "/"),
    )

    return int(match.group(1)) if match else 10**9


def checkpoint_candidates() -> list[Path]:
    candidates = sorted(
        CHECKPOINT_ROOT.glob(
            "v*/weights/lightning/model.ckpt"
        ),
        key=version_number,
    )

    if not candidates:
        raise RuntimeError(
            f"No checkpoint found under {CHECKPOINT_ROOT}"
        )

    return candidates


def expected_test_count() -> int:
    test_root = (
        ROOT
        / "datasets/MVTecAD"
        / CATEGORY
        / "test"
    )

    suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    return sum(
        1
        for path in test_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in suffixes
        )
    )


def expected_summary_auroc() -> float:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(SUMMARY_PATH)

    summary = pd.read_csv(SUMMARY_PATH)

    row = summary[
        summary["category"].astype(str)
        == CATEGORY
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected one summary row for {CATEGORY}, "
            f"found {len(row)}"
        )

    value = float(row.iloc[0]["image_AUROC"])

    if value > 1.0:
        value /= 100.0

    return value


def supported_kwargs(
    callable_object,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(
            callable_object
        )
    except Exception:
        return kwargs

    supports_var_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if supports_var_kwargs:
        return kwargs

    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def field(
    batch: Any,
    names: list[str],
) -> Any:
    for name in names:
        if isinstance(batch, dict) and name in batch:
            return batch[name]

        if hasattr(batch, name):
            return getattr(batch, name)

    return None


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)

    return array.reshape(-1)


def collect_labels_scores(
    outputs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[float] = []
    scores: list[float] = []

    if outputs is None:
        raise RuntimeError(
            "Engine.predict returned None."
        )

    for batch in outputs:
        batch_scores = to_numpy(
            field(
                batch,
                [
                    "pred_score",
                    "pred_scores",
                    "anomaly_score",
                    "image_score",
                ],
            )
        )

        batch_labels = to_numpy(
            field(
                batch,
                [
                    "gt_label",
                    "gt_labels",
                    "label",
                    "labels",
                ],
            )
        )

        if len(batch_scores) == 0:
            raise RuntimeError(
                "Prediction batch has no image scores."
            )

        if len(batch_labels) == 1 and len(batch_scores) > 1:
            batch_labels = np.repeat(
                batch_labels,
                len(batch_scores),
            )

        if len(batch_labels) != len(batch_scores):
            raise RuntimeError(
                "Label/score length mismatch: "
                f"{len(batch_labels)} vs "
                f"{len(batch_scores)}"
            )

        labels.extend(
            batch_labels.astype(float).tolist()
        )

        scores.extend(
            batch_scores.astype(float).tolist()
        )

    y = np.asarray(labels, dtype=float)
    score = np.asarray(scores, dtype=float)

    valid = np.isfinite(y) & np.isfinite(score)

    return (
        y[valid].astype(int),
        score[valid],
    )


def build_engine(Engine, output_path: Path):
    kwargs = {
        "default_root_dir": str(output_path),
        "accelerator": (
            "gpu"
            if torch.cuda.is_available()
            else "cpu"
        ),
        "devices": 1,
        "logger": False,
    }

    return Engine(
        **supported_kwargs(Engine, kwargs)
    )



def make_compatible_mvtec_datamodule(
    module,
    MVTecAD,
    data_config: dict,
    category: str,
):
    """
    Preserve the legacy datamodule construction logic while
    filtering arguments unsupported by the installed anomalib.
    """

    captured = {}

    def constructor_adapter(*args, **kwargs):
        filtered = supported_kwargs(
            MVTecAD,
            kwargs,
        )

        removed = sorted(
            set(kwargs) - set(filtered)
        )

        captured["received"] = sorted(
            kwargs.keys()
        )
        captured["accepted"] = sorted(
            filtered.keys()
        )
        captured["removed"] = removed

        print(
            "[MVTecAD compatibility] received:",
            captured["received"],
        )
        print(
            "[MVTecAD compatibility] accepted:",
            captured["accepted"],
        )
        print(
            "[MVTecAD compatibility] removed:",
            captured["removed"],
        )

        return MVTecAD(
            *args,
            **filtered,
        )

    datamodule = module.make_mvtec_datamodule(
        constructor_adapter,
        data_config,
        category,
    )

    if "image_size" not in captured.get(
        "removed",
        [],
    ):
        print(
            "[MVTecAD compatibility] note: "
            "image_size was not present or was accepted."
        )

    return datamodule

def predict_from_checkpoint(
    module,
    config: dict,
    checkpoint: Path,
    output_path: Path,
) -> tuple[Any, str]:
    MVTecAD, model_map, Engine = (
        module.import_anomalib()
    )

    datamodule = make_compatible_mvtec_datamodule(
        module=module,
        MVTecAD=MVTecAD,
        data_config=config["data"],
        category=CATEGORY,
    )

    model = module.make_model(
        "patchcore",
        model_map["patchcore"],
        CATEGORY,
    )

    engine = build_engine(
        Engine,
        output_path,
    )

    predict_signature = inspect.signature(
        engine.predict
    )

    accepts_ckpt_path = (
        "ckpt_path"
        in predict_signature.parameters
        or any(
            parameter.kind
            == inspect.Parameter.VAR_KEYWORD
            for parameter
            in predict_signature.parameters.values()
        )
    )

    if accepts_ckpt_path:
        outputs = engine.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(checkpoint),
            return_predictions=True,
        )

        return outputs, "engine_predict_ckpt_path"

    model_class = model_map["patchcore"]

    if not hasattr(
        model_class,
        "load_from_checkpoint",
    ):
        raise RuntimeError(
            "Neither Engine.predict(ckpt_path=...) nor "
            "Patchcore.load_from_checkpoint is available."
        )

    loaded_model = (
        model_class.load_from_checkpoint(
            str(checkpoint),
            map_location="cpu",
        )
    )

    outputs = engine.predict(
        model=loaded_model,
        datamodule=datamodule,
        return_predictions=True,
    )

    return outputs, "load_from_checkpoint"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    for path in [
        BASELINE_SCRIPT,
        CONFIG_PATH,
        SUMMARY_PATH,
        CHECKPOINT_ROOT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    module = load_baseline_module()

    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    checkpoints = checkpoint_candidates()
    expected_count = expected_test_count()
    expected_auroc = expected_summary_auroc()

    MVTecAD, model_map, Engine = (
        module.import_anomalib()
    )

    model_class = model_map["patchcore"]

    print(
        "===== STAGE 22-D4a VALIDATION ====="
    )
    print("category:", CATEGORY)
    print("expected test images:", expected_count)
    print("summary image AUROC:", expected_auroc)
    print("checkpoint candidates:", len(checkpoints))
    print(
        "Patchcore load_from_checkpoint:",
        hasattr(
            model_class,
            "load_from_checkpoint",
        ),
    )
    print(
        "Engine.predict signature:",
        inspect.signature(Engine.predict),
    )

    for checkpoint in checkpoints:
        print(
            "checkpoint:",
            checkpoint.relative_to(ROOT),
        )

    print()
    print(
        "IMPORTANT: this script never calls engine.fit()."
    )

    if args.validate_only:
        print("[OK] Validation-only run passed.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the checkpoint probe."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for checkpoint in checkpoints:
        version = version_number(checkpoint)

        print()
        print(
            f"===== Testing bottle checkpoint v{version} ====="
        )

        started = time.perf_counter()

        try:
            outputs, load_method = (
                predict_from_checkpoint(
                    module=module,
                    config=config,
                    checkpoint=checkpoint,
                    output_path=(
                        OUT_DIR
                        / "stage22_d4a_runtime"
                        / f"v{version}"
                    ),
                )
            )

            y, score = collect_labels_scores(
                outputs
            )

            if len(np.unique(y)) < 2:
                raise RuntimeError(
                    "Predictions do not contain both "
                    "normal and anomalous labels."
                )

            auroc = float(
                roc_auc_score(y, score)
            )

            delta = auroc - expected_auroc
            sample_count_match = (
                len(y) == expected_count
            )

            status = (
                "match"
                if (
                    sample_count_match
                    and abs(delta)
                    <= MATCH_TOLERANCE
                )
                else "mismatch"
            )

            error = ""

        except Exception as exc:
            load_method = ""
            auroc = float("nan")
            delta = float("nan")
            sample_count_match = False
            status = "error"
            error = (
                f"{type(exc).__name__}: {exc}"
            )
            y = np.asarray([])

        elapsed = (
            time.perf_counter() - started
        )

        row = {
            "category": CATEGORY,
            "version": version,
            "checkpoint": str(
                checkpoint.relative_to(ROOT)
            ),
            "checkpoint_size_mib": (
                checkpoint.stat().st_size
                / 1024**2
            ),
            "load_method": load_method,
            "num_predictions": len(y),
            "expected_num_predictions": (
                expected_count
            ),
            "sample_count_match": (
                sample_count_match
            ),
            "image_auroc": auroc,
            "summary_image_auroc": (
                expected_auroc
            ),
            "delta_vs_summary": delta,
            "tolerance": MATCH_TOLERANCE,
            "status": status,
            "elapsed_sec": elapsed,
            "error": error,
        }

        rows.append(row)

        print(
            json.dumps(
                row,
                indent=2,
                ensure_ascii=False,
            )
        )

        torch.cuda.empty_cache()

    results = pd.DataFrame(rows)

    results.to_csv(
        OUT_RESULTS,
        index=False,
        lineterminator="\n",
    )

    valid_matches = results[
        results["status"] == "match"
    ].copy()

    if valid_matches.empty:
        decision = {
            "decision": (
                "checkpoint_reuse_not_yet_validated"
            ),
            "selected_checkpoint": None,
            "reason": (
                "No bottle checkpoint reproduced "
                "the locked summary AUROC within "
                f"{MATCH_TOLERANCE:.4f}."
            ),
        }
    else:
        valid_matches[
            "absolute_delta"
        ] = (
            valid_matches[
                "delta_vs_summary"
            ].abs()
        )

        selected = valid_matches.sort_values(
            [
                "absolute_delta",
                "version",
            ]
        ).iloc[0]

        decision = {
            "decision": (
                "checkpoint_reuse_validated"
            ),
            "selected_checkpoint": (
                selected["checkpoint"]
            ),
            "selected_version": int(
                selected["version"]
            ),
            "image_auroc": float(
                selected["image_auroc"]
            ),
            "summary_image_auroc": float(
                selected[
                    "summary_image_auroc"
                ]
            ),
            "delta_vs_summary": float(
                selected[
                    "delta_vs_summary"
                ]
            ),
            "load_method": (
                selected["load_method"]
            ),
        }

    OUT_SELECTION.write_text(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Stage 22-D4a: Bottle Checkpoint Probe",
        "",
        "## Restrictions",
        "",
        "- PatchCore fitting: `none`",
        "- category: `bottle`",
        f"- expected test images: `{expected_count}`",
        f"- locked summary AUROC: `{expected_auroc:.6f}`",
        f"- matching tolerance: `{MATCH_TOLERANCE:.4f}`",
        "",
        "## Results",
        "",
        "| Version | Predictions | AUROC | Delta vs summary | Load method | Status |",
        "|---:|---:|---:|---:|---|---|",
    ]

    for _, row in results.iterrows():
        lines.append(
            f"| v{int(row['version'])} | "
            f"{int(row['num_predictions'])} | "
            f"{row['image_auroc']:.6f} | "
            f"{row['delta_vs_summary']:+.6f} | "
            f"{row['load_method']} | "
            f"{row['status']} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(
        "===== STAGE 22-D4a COMPLETE ====="
    )
    print(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print("[DONE]", OUT_RESULTS)
    print("[DONE]", OUT_SELECTION)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
