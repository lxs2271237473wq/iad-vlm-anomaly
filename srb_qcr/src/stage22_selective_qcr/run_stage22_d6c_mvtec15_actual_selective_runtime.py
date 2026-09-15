from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

STAGE7_SCRIPT = (
    ROOT
    / "experiments/stage7_generalization"
    / "visa_binary_prompt_reasoning.py"
)

D6B_PREDICTIONS = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_srb_qcr_transfer"
    / "stage22_d6b_mvtec15_unified_predictions.csv"
)

CANDIDATE_ROOT = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_rerun_patchcore"
    / "MVTecAD"
)

OUTPUT_ROOT = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_actual_selective_runtime"
)

DOC_ROOT = (
    ROOT
    / "docs/stage22_selective_qcr"
)

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

EXPECTED_FULL_ROWS = 1725
EXPECTED_FULL_SELECTIVE_CALLS = 1294

FROZEN = {
    "w_max": 0.35,
    "q_quantile": 0.25,
    "tau_delta": 0.75,
}


def load_python_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import Python module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def canonical_path(value: Any) -> str:
    text = str(value).replace("\\", "/").strip()

    for marker in (
        "/datasets/MVTecAD/",
        "datasets/MVTecAD/",
    ):
        if marker in text:
            return text.split(marker, 1)[1]

    return text.removeprefix("./")


def as_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():
        return numeric.gt(0)

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    positive = {
        "1",
        "true",
        "yes",
        "anomaly",
        "abnormal",
        "bad",
    }

    negative = {
        "0",
        "false",
        "no",
        "normal",
        "good",
    }

    unknown = sorted(
        set(text.unique())
        - positive
        - negative
    )

    if unknown:
        raise RuntimeError(
            f"Unknown Boolean values: {unknown}"
        )

    return text.isin(positive)


def close_images(images: list[Any]) -> None:
    for image in images:
        close = getattr(
            image,
            "close",
            None,
        )

        if callable(close):
            close()


def candidate_csv(category: str) -> Path:
    return (
        CANDIDATE_ROOT
        / category
        / "candidate_regions.csv"
    )


def load_predictions(
    categories: list[str],
) -> pd.DataFrame:
    if not D6B_PREDICTIONS.exists():
        raise FileNotFoundError(
            D6B_PREDICTIONS
        )

    df = pd.read_csv(
        D6B_PREDICTIONS
    )

    required = {
        "category",
        "image_path",
        "path_key",
        "Y",
        "D",
        "M",
        "Q",
        "M_raw",
        "has_candidate_bool",
        "fallback_bool",
        "srb_pre_gate",
        "srb_weight",
        "score_S1",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "D6b unified predictions are missing "
            f"required columns: {missing}"
        )

    df = df[
        df["category"].astype(str).isin(
            categories
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No D6b rows match the requested categories."
        )

    df["category"] = (
        df["category"].astype(str)
    )

    df["path_key"] = (
        df["path_key"]
        .astype(str)
        .map(canonical_path)
    )

    for column in [
        "Y",
        "D",
        "M",
        "Q",
        "M_raw",
        "srb_pre_gate",
        "srb_weight",
        "score_S1",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["has_candidate_bool"] = (
        as_bool(
            df["has_candidate_bool"]
        )
    )

    df["fallback_bool"] = (
        as_bool(
            df["fallback_bool"]
        )
    )

    if df[
        [
            "Y",
            "D",
            "M",
            "Q",
            "M_raw",
            "srb_pre_gate",
            "score_S1",
        ]
    ].isna().any().any():
        bad = df[
            df[
                [
                    "Y",
                    "D",
                    "M",
                    "Q",
                    "M_raw",
                    "srb_pre_gate",
                    "score_S1",
                ]
            ]
            .isna()
            .any(axis=1)
        ].head(20)

        raise RuntimeError(
            "D6b contains missing numeric values:\n"
            + bad.to_string(index=False)
        )

    if df["path_key"].duplicated().any():
        duplicates = df[
            df["path_key"].duplicated(
                keep=False
            )
        ][
            [
                "category",
                "image_path",
                "path_key",
            ]
        ].head(20)

        raise RuntimeError(
            "Duplicate path keys:\n"
            + duplicates.to_string(index=False)
        )

    actual_categories = set(
        df["category"].unique()
    )

    expected_categories = set(
        categories
    )

    if actual_categories != expected_categories:
        raise RuntimeError(
            "Category mismatch. "
            f"Expected={sorted(expected_categories)}, "
            f"actual={sorted(actual_categories)}"
        )

    for category in categories:
        path = candidate_csv(category)

        if not path.exists():
            raise FileNotFoundError(path)

    return (
        df.sort_values(
            [
                "category",
                "path_key",
            ]
        )
        .reset_index(drop=True)
    )


def load_boxes(
    stage7,
    category: str,
    top_k: int,
    map_size: int,
) -> dict[str, list[dict]]:
    path = candidate_csv(category)
    df = pd.read_csv(path)

    required = {
        "image_path",
        "component_rank",
        "candidate_available",
        "map_x1",
        "map_y1",
        "map_x2",
        "map_y2",
        "map_width",
        "map_height",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{category}: candidate CSV missing "
            f"columns {missing}"
        )

    numeric_columns = [
        "component_rank",
        "candidate_available",
        "map_x1",
        "map_y1",
        "map_x2",
        "map_y2",
        "map_width",
        "map_height",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df[
        (df["candidate_available"] == 1)
        & (df["component_rank"] > 0)
    ].copy()

    if df[numeric_columns].isna().any().any():
        raise RuntimeError(
            f"{category}: invalid candidate coordinates"
        )

    boxes: dict[
        str,
        list[dict],
    ] = {}

    for image_path, group in df.groupby(
        "image_path",
        sort=False,
    ):
        group = (
            group.sort_values(
                "component_rank"
            )
            .head(top_k)
        )

        converted = []

        for _, row in group.iterrows():
            source_width = int(
                row["map_width"]
            )

            source_height = int(
                row["map_height"]
            )

            if (
                source_width <= 0
                or source_height <= 0
            ):
                raise RuntimeError(
                    f"{category}: invalid map size "
                    f"{source_width}x{source_height}"
                )

            scale_x = (
                map_size / source_width
            )

            scale_y = (
                map_size / source_height
            )

            x1 = int(
                np.floor(
                    float(row["map_x1"])
                    * scale_x
                )
            )

            y1 = int(
                np.floor(
                    float(row["map_y1"])
                    * scale_y
                )
            )

            # D4b map_x2/map_y2 are exclusive.
            # Stage 7 crop_candidate expects inclusive x2/y2.
            x2 = (
                int(
                    np.ceil(
                        float(row["map_x2"])
                        * scale_x
                    )
                )
                - 1
            )

            y2 = (
                int(
                    np.ceil(
                        float(row["map_y2"])
                        * scale_y
                    )
                )
                - 1
            )

            x1 = max(
                0,
                min(x1, map_size - 1),
            )

            y1 = max(
                0,
                min(y1, map_size - 1),
            )

            x2 = max(
                0,
                min(x2, map_size - 1),
            )

            y2 = max(
                0,
                min(y2, map_size - 1),
            )

            if x2 < x1 or y2 < y1:
                raise RuntimeError(
                    f"{category}: invalid converted box "
                    f"({x1}, {y1}, {x2}, {y2}) "
                    f"for {image_path}"
                )

            converted.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "rank": int(
                        row[
                            "component_rank"
                        ]
                    ),
                }
            )

        key = stage7.canonical_path(
            image_path
        )

        boxes[key] = converted

    return boxes


def build_runtime_assets(
    stage7,
    model,
    tokenizer,
    categories: list[str],
    device: str,
    top_k: int,
    map_size: int,
) -> tuple[
    dict[str, dict[str, list[dict]]],
    dict[str, torch.Tensor],
]:
    boxes_by_category = {}
    text_features = {}

    for category in categories:
        boxes = load_boxes(
            stage7=stage7,
            category=category,
            top_k=top_k,
            map_size=map_size,
        )

        boxes_by_category[
            category
        ] = boxes

        features, _ = (
            stage7.build_text_features(
                model=model,
                tokenizer=tokenizer,
                category=category,
                strategy=(
                    "inspection_binary"
                ),
                device=device,
            )
        )

        text_features[
            category
        ] = features

    return (
        boxes_by_category,
        text_features,
    )


def infer_one(
    stage7,
    model,
    preprocess,
    text_features: torch.Tensor,
    boxes: dict[str, list[dict]],
    row: pd.Series,
    runtime_args: SimpleNamespace,
    device: str,
) -> dict:
    eval_images = []

    try:
        (
            eval_images,
            used_mode,
            fallback,
        ) = stage7.get_eval_images(
            row,
            boxes,
            "crop_topk_ensemble",
            runtime_args,
        )

        features = stage7.encode_images(
            model=model,
            preprocess=preprocess,
            images=eval_images,
            device=device,
        )

        similarities = (
            features
            @ text_features.T
        ).detach().cpu().numpy()

        margins = (
            similarities[:, 1]
            - similarities[:, 0]
        )

        best_index = int(
            np.argmax(margins)
        )

        return {
            "path_key": canonical_path(
                row["path_key"]
            ),
            "category": str(
                row["category"]
            ),
            "M_raw_runtime": float(
                margins[best_index]
            ),
            "best_normal_similarity": float(
                similarities[
                    best_index,
                    0,
                ]
            ),
            "best_anomaly_similarity": float(
                similarities[
                    best_index,
                    1,
                ]
            ),
            "best_crop_index": best_index,
            "num_eval_images": int(
                len(eval_images)
            ),
            "fallback": int(fallback),
            "used_mode": str(
                used_mode
            ),
        }

    finally:
        close_images(eval_images)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def warmup(
    stage7,
    model,
    preprocess,
    text_features_by_category,
    boxes_by_category,
    predictions: pd.DataFrame,
    runtime_args,
    device: str,
    warmup_images: int,
) -> None:
    if warmup_images <= 0:
        return

    gate = predictions[
        predictions[
            "srb_pre_gate"
        ].gt(0)
    ]

    source = (
        gate
        if not gate.empty
        else predictions
    )

    rows = (
        source.head(
            warmup_images
        )
        .copy()
    )

    print(
        f"[WARMUP] images={len(rows)}"
    )

    for _, row in rows.iterrows():
        category = str(
            row["category"]
        )

        infer_one(
            stage7=stage7,
            model=model,
            preprocess=preprocess,
            text_features=(
                text_features_by_category[
                    category
                ]
            ),
            boxes=(
                boxes_by_category[
                    category
                ]
            ),
            row=row,
            runtime_args=runtime_args,
            device=device,
        )

    synchronize(device)


def run_mode(
    stage7,
    model,
    preprocess,
    text_features_by_category,
    boxes_by_category,
    predictions: pd.DataFrame,
    runtime_args,
    device: str,
    mode: str,
    repeat: int,
) -> tuple[dict, pd.DataFrame]:
    if mode not in {
        "full",
        "selective",
    }:
        raise ValueError(mode)

    if mode == "full":
        active = predictions.copy()
    else:
        active = predictions[
            predictions[
                "srb_pre_gate"
            ].gt(0)
        ].copy()

    active = (
        active.sort_values(
            [
                "category",
                "path_key",
            ]
        )
        .reset_index(drop=True)
    )

    gc.collect()

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    synchronize(device)

    started = time.perf_counter()
    rows = []

    for index, row in active.iterrows():
        category = str(
            row["category"]
        )

        result = infer_one(
            stage7=stage7,
            model=model,
            preprocess=preprocess,
            text_features=(
                text_features_by_category[
                    category
                ]
            ),
            boxes=(
                boxes_by_category[
                    category
                ]
            ),
            row=row,
            runtime_args=runtime_args,
            device=device,
        )

        rows.append(result)

        if (
            index == 0
            or (index + 1) % 100 == 0
            or index + 1 == len(active)
        ):
            print(
                f"[{mode} repeat {repeat}] "
                f"{index + 1}/{len(active)}"
            )

    synchronize(device)

    elapsed = (
        time.perf_counter()
        - started
    )

    peak_mib = (
        float(
            torch.cuda.max_memory_allocated()
        )
        / (1024 ** 2)
        if device.startswith("cuda")
        else float("nan")
    )

    output = pd.DataFrame(rows)

    if len(output) != len(active):
        raise RuntimeError(
            f"{mode}: expected {len(active)} "
            f"outputs, got {len(output)}"
        )

    fallback_count = int(
        output["fallback"].sum()
    )

    if fallback_count:
        raise RuntimeError(
            f"{mode}: runtime produced "
            f"{fallback_count} fallbacks"
        )

    total_eval_images = int(
        output[
            "num_eval_images"
        ].sum()
    )

    metrics = {
        "repeat": repeat,
        "mode": mode,
        "vlm_calls": int(
            len(active)
        ),
        "eval_images": (
            total_eval_images
        ),
        "elapsed_sec": elapsed,
        "sec_per_call": (
            elapsed / len(active)
            if len(active)
            else float("nan")
        ),
        "images_per_sec": (
            len(active) / elapsed
            if elapsed > 0
            else float("nan")
        ),
        "peak_gpu_allocated_mib": (
            peak_mib
        ),
        "fallback_count": (
            fallback_count
        ),
    }

    return metrics, output


def normalize_runtime_m(
    base: pd.DataFrame,
    runtime: pd.DataFrame,
) -> pd.DataFrame:
    result = runtime.copy()

    stats = (
        base.groupby(
            "category",
            as_index=False,
        )
        .agg(
            m_raw_min=(
                "M_raw",
                "min",
            ),
            m_raw_max=(
                "M_raw",
                "max",
            ),
        )
    )

    result = result.merge(
        stats,
        on="category",
        how="left",
        validate="many_to_one",
    )

    span = (
        result["m_raw_max"]
        - result["m_raw_min"]
    )

    result["M_runtime"] = (
        (
            result["M_raw_runtime"]
            - result["m_raw_min"]
        )
        / span.where(
            span > 1e-12
        )
    ).fillna(0.0).clip(
        0.0,
        1.0,
    )

    return result


def build_score_audit(
    base: pd.DataFrame,
    runtime: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, dict]:
    runtime = normalize_runtime_m(
        base=base,
        runtime=runtime,
    )

    merged = base.merge(
        runtime[
            [
                "path_key",
                "M_raw_runtime",
                "M_runtime",
                "num_eval_images",
                "fallback",
                "used_mode",
            ]
        ],
        on="path_key",
        how="left",
        validate="one_to_one",
    )

    selected = merged[
        "srb_pre_gate"
    ].gt(0)

    if mode == "full":
        expected_runtime = pd.Series(
            True,
            index=merged.index,
        )
    else:
        expected_runtime = selected

    actual_runtime = merged[
        "M_raw_runtime"
    ].notna()

    if not actual_runtime.equals(
        expected_runtime
    ):
        mismatch = merged[
            actual_runtime
            != expected_runtime
        ][
            [
                "category",
                "path_key",
                "srb_pre_gate",
                "M_raw_runtime",
            ]
        ].head(20)

        raise RuntimeError(
            f"{mode}: runtime coverage mismatch:\n"
            + mismatch.to_string(index=False)
        )

    compared = merged[
        actual_runtime
    ].copy()

    compared[
        "M_raw_abs_diff"
    ] = (
        compared[
            "M_raw_runtime"
        ]
        - compared["M_raw"]
    ).abs()

    compared[
        "M_norm_abs_diff"
    ] = (
        compared[
            "M_runtime"
        ]
        - compared["M"]
    ).abs()

    merged[
        "M_for_runtime_score"
    ] = merged["M_runtime"]

    # For skipped rows M is never used because pre-gate=0.
    merged.loc[
        ~actual_runtime,
        "M_for_runtime_score",
    ] = 0.0

    agreement = (
        1.0
        - (
            (
                merged["D"]
                - merged[
                    "M_for_runtime_score"
                ]
            ).abs()
            / FROZEN["tau_delta"]
        )
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    weight = (
        FROZEN["w_max"]
        * merged[
            "srb_pre_gate"
        ].astype(float)
        * merged["Q"]
        * agreement
    ).clip(
        lower=0.0,
        upper=FROZEN["w_max"],
    )

    merged[
        "score_S1_runtime"
    ] = (
        merged["D"]
        + weight
        * (
            merged[
                "M_for_runtime_score"
            ]
            - merged["D"]
        )
    )

    merged[
        "score_S1_abs_diff"
    ] = (
        merged[
            "score_S1_runtime"
        ]
        - merged["score_S1"]
    ).abs()

    summary = {
        "mode": mode,
        "runtime_rows": int(
            actual_runtime.sum()
        ),
        "max_m_raw_abs_diff": float(
            compared[
                "M_raw_abs_diff"
            ].max()
        ),
        "mean_m_raw_abs_diff": float(
            compared[
                "M_raw_abs_diff"
            ].mean()
        ),
        "max_m_norm_abs_diff": float(
            compared[
                "M_norm_abs_diff"
            ].max()
        ),
        "max_score_s1_abs_diff": float(
            merged[
                "score_S1_abs_diff"
            ].max()
        ),
        "mean_score_s1_abs_diff": float(
            merged[
                "score_S1_abs_diff"
            ].mean()
        ),
    }

    columns = [
        "category",
        "image_path",
        "path_key",
        "Y",
        "D",
        "M",
        "Q",
        "M_raw",
        "M_raw_runtime",
        "M_runtime",
        "srb_pre_gate",
        "score_S1",
        "score_S1_runtime",
        "score_S1_abs_diff",
        "num_eval_images",
        "fallback",
        "used_mode",
    ]

    return (
        merged[columns],
        summary,
    )


def paired_results(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    pivot = raw.pivot(
        index="repeat",
        columns="mode",
        values=[
            "vlm_calls",
            "eval_images",
            "elapsed_sec",
            "peak_gpu_allocated_mib",
        ],
    )

    rows = []

    for repeat in sorted(
        raw["repeat"].unique()
    ):
        full = raw[
            (raw["repeat"] == repeat)
            & (raw["mode"] == "full")
        ].iloc[0]

        selective = raw[
            (raw["repeat"] == repeat)
            & (
                raw["mode"]
                == "selective"
            )
        ].iloc[0]

        rows.append(
            {
                "repeat": int(repeat),
                "full_vlm_calls": int(
                    full["vlm_calls"]
                ),
                "selective_vlm_calls": int(
                    selective[
                        "vlm_calls"
                    ]
                ),
                "actual_calls_saved": int(
                    full["vlm_calls"]
                    - selective[
                        "vlm_calls"
                    ]
                ),
                "actual_call_saving_rate": float(
                    1.0
                    - selective[
                        "vlm_calls"
                    ]
                    / full["vlm_calls"]
                ),
                "full_eval_images": int(
                    full["eval_images"]
                ),
                "selective_eval_images": int(
                    selective[
                        "eval_images"
                    ]
                ),
                "full_elapsed_sec": float(
                    full["elapsed_sec"]
                ),
                "selective_elapsed_sec": float(
                    selective[
                        "elapsed_sec"
                    ]
                ),
                "wall_time_saved_sec": float(
                    full["elapsed_sec"]
                    - selective[
                        "elapsed_sec"
                    ]
                ),
                "wall_time_saving_rate": float(
                    1.0
                    - selective[
                        "elapsed_sec"
                    ]
                    / full["elapsed_sec"]
                ),
                "speedup": float(
                    full["elapsed_sec"]
                    / selective[
                        "elapsed_sec"
                    ]
                ),
                "full_peak_gpu_allocated_mib": float(
                    full[
                        "peak_gpu_allocated_mib"
                    ]
                ),
                "selective_peak_gpu_allocated_mib": float(
                    selective[
                        "peak_gpu_allocated_mib"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def write_report(
    predictions: pd.DataFrame,
    raw: pd.DataFrame,
    paired: pd.DataFrame,
    audit_summaries: dict[str, dict],
    categories: list[str],
    args: argparse.Namespace,
) -> None:
    full_calls = int(
        predictions.shape[0]
    )

    selective_calls = int(
        predictions[
            "srb_pre_gate"
        ].gt(0).sum()
    )

    lines = [
        "# Stage 22-D6c: MVTec AD Actual Selective Runtime",
        "",
        "## Protocol",
        "",
        "- target: `MVTec AD`",
        f"- categories: `{len(categories)}`",
        f"- images: `{len(predictions)}`",
        "- CLIP: `ViT-B-32/openai`",
        "- prompt strategy: `inspection_binary`",
        "- crop mode: `crop_topk_ensemble`",
        "- CLIP model load excluded from timing",
        "- text features and candidate boxes cached before timing",
        "- image loading, crop construction, preprocessing, and CLIP image inference included",
        f"- paired repeats: `{args.repeats}`",
        "- execution order alternates by repeat to reduce order bias",
        "",
        "## Frozen SRB configuration",
        "",
        f"- `w_max = {FROZEN['w_max']}`",
        f"- `q_quantile = {FROZEN['q_quantile']}`",
        f"- `tau_delta = {FROZEN['tau_delta']}`",
        "",
        "## Actual calls",
        "",
        f"- full VLM calls: `{full_calls}`",
        f"- selective VLM calls: `{selective_calls}`",
        f"- actual calls saved: `{full_calls - selective_calls}`",
        f"- actual call saving rate: `{1.0 - selective_calls / full_calls:.6f}`",
        "",
        "## Paired runtime results",
        "",
        "| Repeat | Full sec | Selective sec | Saved sec | Saving rate | Speedup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in paired.iterrows():
        lines.append(
            f"| {int(row['repeat'])} | "
            f"{row['full_elapsed_sec']:.3f} | "
            f"{row['selective_elapsed_sec']:.3f} | "
            f"{row['wall_time_saved_sec']:.3f} | "
            f"{row['wall_time_saving_rate']:.4f} | "
            f"{row['speedup']:.4f}x |"
        )

    lines += [
        "",
        "## Median runtime",
        "",
        f"- full median time: `{paired['full_elapsed_sec'].median():.3f} s`",
        f"- selective median time: `{paired['selective_elapsed_sec'].median():.3f} s`",
        f"- median wall-time saving: `{paired['wall_time_saving_rate'].median():.6f}`",
        f"- median speedup: `{paired['speedup'].median():.6f}x`",
        "",
        "## Output consistency",
        "",
    ]

    for mode in [
        "full",
        "selective",
    ]:
        summary = audit_summaries[mode]

        lines += [
            f"### {mode}",
            f"- runtime rows: `{summary['runtime_rows']}`",
            f"- max raw M difference: `{summary['max_m_raw_abs_diff']:.10f}`",
            f"- max normalized M difference: `{summary['max_m_norm_abs_diff']:.10f}`",
            f"- max SRB score difference: `{summary['max_score_s1_abs_diff']:.10f}`",
            "",
        ]

    lines += [
        "The measured speedup is specific to this hardware, software stack, "
        "image-crop protocol, and per-image CLIP execution path.",
    ]

    DOC_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        DOC_ROOT
        / "stage22_d6c_mvtec15_actual_selective_runtime.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--clip_model",
        default="ViT-B-32",
    )

    parser.add_argument(
        "--clip_pretrained",
        default="openai",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--map_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--crop_padding",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min_crop_size",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--warmup_images",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--score_tolerance",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    categories = list(
        dict.fromkeys(
            args.categories
        )
    )

    invalid = [
        category
        for category in categories
        if category not in CATEGORIES
    ]

    if invalid:
        raise ValueError(
            f"Unknown categories: {invalid}"
        )

    if args.repeats < 1:
        raise ValueError(
            "repeats must be at least 1"
        )

    predictions = load_predictions(
        categories
    )

    full_calls = len(predictions)

    selective_calls = int(
        predictions[
            "srb_pre_gate"
        ].gt(0).sum()
    )

    print(
        "===== STAGE 22-D6c PREFLIGHT ====="
    )

    print(
        "categories:",
        len(categories),
        categories,
    )

    print("images:", full_calls)

    print(
        "full VLM calls:",
        full_calls,
    )

    print(
        "selective VLM calls:",
        selective_calls,
    )

    print(
        "actual calls saved:",
        full_calls - selective_calls,
    )

    print(
        "call saving rate:",
        f"{1.0 - selective_calls / full_calls:.6f}",
    )

    if set(categories) == set(CATEGORIES):
        if full_calls != EXPECTED_FULL_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_FULL_ROWS} "
                f"full rows, got {full_calls}"
            )

        if (
            selective_calls
            != EXPECTED_FULL_SELECTIVE_CALLS
        ):
            raise RuntimeError(
                "Expected "
                f"{EXPECTED_FULL_SELECTIVE_CALLS} "
                "selective calls, got "
                f"{selective_calls}"
            )

    if args.validate_only:
        print()
        print(
            "[OK] Validation-only run passed."
        )
        return

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    stage7 = load_python_module(
        STAGE7_SCRIPT,
        "stage7_binary_prompt_reasoning",
    )

    print()
    print(
        "[INFO] Loading CLIP once; "
        "model load is excluded from timing."
    )

    model, _, preprocess = (
        stage7.open_clip
        .create_model_and_transforms(
            args.clip_model,
            pretrained=(
                args.clip_pretrained
            ),
            device=args.device,
        )
    )

    tokenizer = (
        stage7.open_clip.get_tokenizer(
            args.clip_model
        )
    )

    model.eval()

    (
        boxes_by_category,
        text_features_by_category,
    ) = build_runtime_assets(
        stage7=stage7,
        model=model,
        tokenizer=tokenizer,
        categories=categories,
        device=args.device,
        top_k=args.top_k,
        map_size=args.map_size,
    )

    runtime_args = SimpleNamespace(
        patchcore_root=str(
            CANDIDATE_ROOT.parent
        ),
        top_k=args.top_k,
        map_size=args.map_size,
        crop_padding=args.crop_padding,
        min_crop_size=args.min_crop_size,
    )

    warmup(
        stage7=stage7,
        model=model,
        preprocess=preprocess,
        text_features_by_category=(
            text_features_by_category
        ),
        boxes_by_category=(
            boxes_by_category
        ),
        predictions=predictions,
        runtime_args=runtime_args,
        device=args.device,
        warmup_images=(
            args.warmup_images
        ),
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_records = []
    first_outputs = {}

    for repeat in range(
        1,
        args.repeats + 1,
    ):
        order = (
            ["full", "selective"]
            if repeat % 2 == 1
            else ["selective", "full"]
        )

        print()
        print(
            f"===== PAIRED REPEAT {repeat} "
            f"ORDER={order} ====="
        )

        for mode in order:
            metrics, output = run_mode(
                stage7=stage7,
                model=model,
                preprocess=preprocess,
                text_features_by_category=(
                    text_features_by_category
                ),
                boxes_by_category=(
                    boxes_by_category
                ),
                predictions=predictions,
                runtime_args=runtime_args,
                device=args.device,
                mode=mode,
                repeat=repeat,
            )

            raw_records.append(
                metrics
            )

            if mode not in first_outputs:
                first_outputs[
                    mode
                ] = output

            print(
                f"[DONE] {mode} repeat {repeat}: "
                f"calls={metrics['vlm_calls']}, "
                f"eval_images={metrics['eval_images']}, "
                f"time={metrics['elapsed_sec']:.3f}s, "
                f"peak={metrics['peak_gpu_allocated_mib']:.1f} MiB"
            )

    raw = pd.DataFrame(
        raw_records
    ).sort_values(
        [
            "repeat",
            "mode",
        ]
    )

    paired = paired_results(raw)

    audit_summaries = {}

    for mode in [
        "full",
        "selective",
    ]:
        audit, summary = (
            build_score_audit(
                base=predictions,
                runtime=(
                    first_outputs[mode]
                ),
                mode=mode,
            )
        )

        audit_path = (
            OUTPUT_ROOT
            / (
                "stage22_d6c_"
                f"{mode}_score_audit.csv"
            )
        )

        audit.to_csv(
            audit_path,
            index=False,
            lineterminator="\n",
        )

        audit_summaries[
            mode
        ] = summary

        if (
            summary[
                "max_m_raw_abs_diff"
            ]
            > args.score_tolerance
        ):
            raise RuntimeError(
                f"{mode}: max raw M difference "
                f"{summary['max_m_raw_abs_diff']} "
                "exceeds tolerance "
                f"{args.score_tolerance}"
            )

        if (
            summary[
                "max_score_s1_abs_diff"
            ]
            > args.score_tolerance
        ):
            raise RuntimeError(
                f"{mode}: max SRB score "
                f"difference "
                f"{summary['max_score_s1_abs_diff']} "
                "exceeds tolerance "
                f"{args.score_tolerance}"
            )

    raw_path = (
        OUTPUT_ROOT
        / "stage22_d6c_raw_runtime.csv"
    )

    paired_path = (
        OUTPUT_ROOT
        / "stage22_d6c_paired_runtime.csv"
    )

    summary_path = (
        OUTPUT_ROOT
        / "stage22_d6c_runtime_summary.json"
    )

    raw.to_csv(
        raw_path,
        index=False,
        lineterminator="\n",
    )

    paired.to_csv(
        paired_path,
        index=False,
        lineterminator="\n",
    )

    summary_payload = {
        "protocol_id": (
            "stage22_d6c_mvtec15_"
            "actual_selective_runtime_v1"
        ),
        "categories": categories,
        "num_categories": len(
            categories
        ),
        "num_images": len(
            predictions
        ),
        "full_vlm_calls": full_calls,
        "selective_vlm_calls": (
            selective_calls
        ),
        "actual_calls_saved": (
            full_calls - selective_calls
        ),
        "actual_call_saving_rate": (
            1.0
            - selective_calls
            / full_calls
        ),
        "repeats": args.repeats,
        "full_median_sec": float(
            paired[
                "full_elapsed_sec"
            ].median()
        ),
        "selective_median_sec": float(
            paired[
                "selective_elapsed_sec"
            ].median()
        ),
        "median_wall_time_saving_rate": float(
            paired[
                "wall_time_saving_rate"
            ].median()
        ),
        "median_speedup": float(
            paired[
                "speedup"
            ].median()
        ),
        "mean_wall_time_saving_rate": float(
            paired[
                "wall_time_saving_rate"
            ].mean()
        ),
        "mean_speedup": float(
            paired[
                "speedup"
            ].mean()
        ),
        "full_peak_gpu_allocated_mib_max": float(
            raw.loc[
                raw["mode"] == "full",
                "peak_gpu_allocated_mib",
            ].max()
        ),
        "selective_peak_gpu_allocated_mib_max": float(
            raw.loc[
                raw["mode"]
                == "selective",
                "peak_gpu_allocated_mib",
            ].max()
        ),
        "score_tolerance": (
            args.score_tolerance
        ),
        "audit": audit_summaries,
        "timing_scope": {
            "clip_model_load_included": False,
            "text_encoding_included": False,
            "candidate_box_loading_included": False,
            "image_loading_included": True,
            "crop_construction_included": True,
            "image_preprocessing_included": True,
            "clip_image_inference_included": True,
        },
        "frozen_configuration": (
            FROZEN
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    write_report(
        predictions=predictions,
        raw=raw,
        paired=paired,
        audit_summaries=(
            audit_summaries
        ),
        categories=categories,
        args=args,
    )

    print()
    print(
        "===== STAGE 22-D6c COMPLETE ====="
    )

    print(
        paired.to_string(
            index=False
        )
    )

    print()
    print(
        "median full time:",
        f"{paired['full_elapsed_sec'].median():.3f}s",
    )

    print(
        "median selective time:",
        f"{paired['selective_elapsed_sec'].median():.3f}s",
    )

    print(
        "median wall-time saving:",
        f"{paired['wall_time_saving_rate'].median():.6f}",
    )

    print(
        "median speedup:",
        f"{paired['speedup'].median():.6f}x",
    )

    print()
    print(
        "full max raw M diff:",
        f"{audit_summaries['full']['max_m_raw_abs_diff']:.10f}",
    )

    print(
        "selective max raw M diff:",
        f"{audit_summaries['selective']['max_m_raw_abs_diff']:.10f}",
    )

    print(
        "selective max SRB score diff:",
        f"{audit_summaries['selective']['max_score_s1_abs_diff']:.10f}",
    )

    print()

    for path in [
        raw_path,
        paired_path,
        summary_path,
        OUTPUT_ROOT
        / "stage22_d6c_full_score_audit.csv",
        OUTPUT_ROOT
        / "stage22_d6c_selective_score_audit.csv",
        DOC_ROOT
        / "stage22_d6c_mvtec15_actual_selective_runtime.md",
    ]:
        print("[DONE]", path)


if __name__ == "__main__":
    main()
