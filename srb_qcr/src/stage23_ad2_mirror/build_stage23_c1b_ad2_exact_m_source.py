from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

CATEGORIES = [
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "walnuts",
]

B2B_PATH = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_b2b_ad2_frozen_predictions.csv"
)

IMAGE_PATH = (
    ROOT
    / "results/stage11_mvtecad2_multicategory"
    / "stage11_d_vlm_image_predictions.csv"
)

CANDIDATE_PATH = (
    ROOT
    / "results/stage11_mvtecad2_multicategory"
    / "stage11_d_vlm_candidate_scores.csv"
)

OUT_JSON = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "stage23_c1b_ad2_exact_m_source.json"
)

OUT_TXT = (
    ROOT
    / "docs/stage23_ad2_mirror"
    / "stage23_c1b_ad2_exact_m_source.txt"
)

IMAGE_SCORE_COLUMNS = [
    "full_image_score",
    "tight_top1_score",
    "tight_topk_max_score",
    "tight_topk_mean_score",
    "context_top1_score",
    "context_topk_max_score",
    "context_topk_mean_score",
]


def normalize_path(value: Any) -> str:
    text = str(value).replace("\\", "/").strip()

    for marker in [
        "/datasets/MVTec_AD_2_anomalib_all/",
        "datasets/MVTec_AD_2_anomalib_all/",
        "/datasets/",
        "datasets/",
    ]:
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

    return text.isin(
        ["1", "true", "yes"]
    )


def prepare_b2b() -> pd.DataFrame:
    frame = pd.read_csv(B2B_PATH)

    required = {
        "category",
        "image_path",
        "M_raw_crop_topk",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"B2b file missing columns: {missing}"
        )

    frame = frame[
        frame["category"]
        .astype(str)
        .isin(CATEGORIES)
    ].copy()

    frame["category"] = (
        frame["category"].astype(str)
    )

    frame["path_key"] = (
        frame["image_path"]
        .astype(str)
        .map(normalize_path)
    )

    frame["M_raw_crop_topk"] = pd.to_numeric(
        frame["M_raw_crop_topk"],
        errors="coerce",
    )

    if "M_available" in frame.columns:
        frame["M_available_bool"] = as_bool(
            frame["M_available"]
        )
    else:
        frame["M_available_bool"] = (
            frame["M_raw_crop_topk"].notna()
        )

    if frame["path_key"].duplicated().any():
        raise RuntimeError(
            "Duplicate path keys in B2b predictions."
        )

    return frame


def prepare_image_predictions() -> pd.DataFrame:
    frame = pd.read_csv(IMAGE_PATH)

    required = {
        "category",
        "image_path",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Image prediction file missing "
            f"columns: {missing}"
        )

    frame = frame[
        frame["category"]
        .astype(str)
        .isin(CATEGORIES)
    ].copy()

    frame["category"] = (
        frame["category"].astype(str)
    )

    frame["path_key"] = (
        frame["image_path"]
        .astype(str)
        .map(normalize_path)
    )

    for column in IMAGE_SCORE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    if frame["path_key"].duplicated().any():
        raise RuntimeError(
            "Duplicate path keys in image predictions."
        )

    return frame


def prepare_candidate_aggregates() -> pd.DataFrame:
    frame = pd.read_csv(CANDIDATE_PATH)

    required = {
        "category",
        "image_path",
        "candidate_rank",
        "tight_vlm_margin",
        "context_vlm_margin",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Candidate score file missing "
            f"columns: {missing}"
        )

    frame = frame[
        frame["category"]
        .astype(str)
        .isin(CATEGORIES)
    ].copy()

    frame["category"] = (
        frame["category"].astype(str)
    )

    frame["path_key"] = (
        frame["image_path"]
        .astype(str)
        .map(normalize_path)
    )

    for column in [
        "candidate_rank",
        "tight_vlm_margin",
        "context_vlm_margin",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame[
        frame["candidate_rank"] > 0
    ].copy()

    rows = []

    for (
        category,
        path_key,
    ), group in frame.groupby(
        ["category", "path_key"],
        sort=False,
    ):
        group = group.sort_values(
            "candidate_rank"
        )

        tight = (
            group["tight_vlm_margin"]
            .dropna()
            .to_numpy(dtype=float)
        )

        context = (
            group["context_vlm_margin"]
            .dropna()
            .to_numpy(dtype=float)
        )

        record = {
            "category": category,
            "path_key": path_key,
            "candidate_count": len(group),
        }

        if len(tight):
            record.update(
                {
                    "candidate_tight_top1": float(
                        tight[0]
                    ),
                    "candidate_tight_topk_max": float(
                        tight.max()
                    ),
                    "candidate_tight_topk_mean": float(
                        tight.mean()
                    ),
                }
            )

        if len(context):
            record.update(
                {
                    "candidate_context_top1": float(
                        context[0]
                    ),
                    "candidate_context_topk_max": float(
                        context.max()
                    ),
                    "candidate_context_topk_mean": float(
                        context.mean()
                    ),
                }
            )

        rows.append(record)

    return pd.DataFrame(rows)


def compare_column(
    merged: pd.DataFrame,
    source_name: str,
    source_column: str,
) -> dict:
    valid = merged[
        merged["M_available_bool"]
    ][
        [
            "M_raw_crop_topk",
            source_column,
        ]
    ].dropna()

    if valid.empty:
        return {
            "source": source_name,
            "column": source_column,
            "rows_compared": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "median_abs_diff": None,
            "p99_abs_diff": None,
            "exact_1e_12": False,
            "exact_1e_9": False,
            "exact_1e_6": False,
        }

    difference = (
        valid[source_column]
        - valid["M_raw_crop_topk"]
    ).abs()

    return {
        "source": source_name,
        "column": source_column,
        "rows_compared": int(len(valid)),
        "max_abs_diff": float(
            difference.max()
        ),
        "mean_abs_diff": float(
            difference.mean()
        ),
        "median_abs_diff": float(
            difference.median()
        ),
        "p99_abs_diff": float(
            difference.quantile(0.99)
        ),
        "exact_1e_12": bool(
            difference.max() <= 1e-12
        ),
        "exact_1e_9": bool(
            difference.max() <= 1e-9
        ),
        "exact_1e_6": bool(
            difference.max() <= 1e-6
        ),
    }


def main() -> None:
    for path in [
        B2B_PATH,
        IMAGE_PATH,
        CANDIDATE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    b2b = prepare_b2b()
    image_predictions = (
        prepare_image_predictions()
    )
    candidate_aggregates = (
        prepare_candidate_aggregates()
    )

    merged = b2b.merge(
        image_predictions.drop(
            columns=["image_path"],
            errors="ignore",
        ),
        on=["category", "path_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_image"),
    )

    merged = merged.merge(
        candidate_aggregates,
        on=["category", "path_key"],
        how="left",
        validate="one_to_one",
    )

    comparisons = []

    for column in IMAGE_SCORE_COLUMNS:
        if column in merged.columns:
            comparisons.append(
                compare_column(
                    merged,
                    source_name=(
                        "stage11_d_vlm_"
                        "image_predictions"
                    ),
                    source_column=column,
                )
            )

    for column in [
        "candidate_tight_top1",
        "candidate_tight_topk_max",
        "candidate_tight_topk_mean",
        "candidate_context_top1",
        "candidate_context_topk_max",
        "candidate_context_topk_mean",
    ]:
        if column in merged.columns:
            comparisons.append(
                compare_column(
                    merged,
                    source_name=(
                        "recomputed_from_"
                        "candidate_scores"
                    ),
                    source_column=column,
                )
            )

    comparisons = sorted(
        comparisons,
        key=lambda row: (
            float("inf")
            if row["max_abs_diff"] is None
            else row["max_abs_diff"],
            row["source"],
            row["column"],
        ),
    )

    stage18_source_counts = {}

    for column in [
        "stage18_m_score_source",
        "stage18_q_score_source",
        "stage18_full_image_source",
        "stage18_note",
    ]:
        if column in b2b.columns:
            stage18_source_counts[column] = (
                b2b[column]
                .dropna()
                .astype(str)
                .value_counts()
                .to_dict()
            )

    matched_image_rows = int(
        merged[
            "full_image_score"
        ].notna().sum()
    ) if "full_image_score" in merged.columns else 0

    best = comparisons[0]

    exact = bool(
        best["exact_1e_12"]
    )

    payload = {
        "protocol_id": (
            "stage23_c1b_ad2_"
            "exact_m_source_v1"
        ),
        "rows": int(len(b2b)),
        "available_m_rows": int(
            b2b["M_available_bool"].sum()
        ),
        "matched_image_rows": (
            matched_image_rows
        ),
        "stage18_source_counts": (
            stage18_source_counts
        ),
        "comparisons": comparisons,
        "selected_source": best,
        "exact_m_source_identified": exact,
        "runtime_runner_ready": exact,
    }

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_TXT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        "===== STAGE 23-C1b EXACT M SOURCE =====",
        "",
        "===== INPUT ALIGNMENT =====",
        f"B2b rows: {len(b2b)}",
        (
            "M-available rows: "
            f"{int(b2b['M_available_bool'].sum())}"
        ),
        (
            "matched image-prediction rows: "
            f"{matched_image_rows}"
        ),
        (
            "Stage 18 source counts: "
            f"{stage18_source_counts}"
        ),
        "",
        "===== ALL SOURCE COMPARISONS =====",
    ]

    for row in comparisons:
        lines.append(
            (
                f"{row['source']}::{row['column']}: "
                f"rows={row['rows_compared']}, "
                f"max={row['max_abs_diff']}, "
                f"mean={row['mean_abs_diff']}, "
                f"p99={row['p99_abs_diff']}, "
                f"exact_1e-12={row['exact_1e_12']}, "
                f"exact_1e-9={row['exact_1e_9']}"
            )
        )

    lines += [
        "",
        "===== DECISION =====",
        (
            "selected_source_family: "
            f"{best['source']}"
        ),
        (
            "selected_source_column: "
            f"{best['column']}"
        ),
        (
            "selected_max_abs_diff: "
            f"{best['max_abs_diff']}"
        ),
        (
            "exact_m_source_identified: "
            f"{exact}"
        ),
        (
            "runtime_runner_ready: "
            f"{exact}"
        ),
        "",
        f"[DONE] {OUT_JSON}",
        f"[DONE] {OUT_TXT}",
    ]

    OUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
