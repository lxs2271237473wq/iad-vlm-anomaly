from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

INPUT = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_srb_qcr_transfer"
    / "stage22_d6b_mvtec15_unified_predictions.csv"
)

OUT_DIR = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_case_analysis"
)

OUT_CSV = OUT_DIR / "stage22_d7a_case_inventory.csv"
OUT_SUMMARY = OUT_DIR / "stage22_d7a_case_summary.json"


REQUIRED = [
    "category",
    "image_path",
    "Y",
    "D",
    "M",
    "Q",
    "score_D0",
    "score_V3",
    "score_V4",
    "score_V6",
    "score_S1",
    "srb_pre_gate",
    "srb_weight",
]


def select_cases(
    frame: pd.DataFrame,
    case_type: str,
    sort_column: str,
    ascending: bool,
    count: int = 10,
) -> pd.DataFrame:
    selected = (
        frame.sort_values(
            sort_column,
            ascending=ascending,
        )
        .head(count)
        .copy()
    )

    selected.insert(
        0,
        "case_type",
        case_type,
    )

    selected.insert(
        1,
        "case_rank",
        range(1, len(selected) + 1),
    )

    return selected


if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

missing = [
    column
    for column in REQUIRED
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )

for column in [
    "Y",
    "D",
    "M",
    "Q",
    "score_D0",
    "score_V3",
    "score_V4",
    "score_V6",
    "score_S1",
    "srb_pre_gate",
    "srb_weight",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

if df[REQUIRED[2:]].isna().any().any():
    raise RuntimeError(
        "The unified prediction table contains missing values."
    )

df["delta_srb_detector"] = (
    df["score_S1"] - df["score_D0"]
)

df["delta_srb_naive"] = (
    df["score_S1"] - df["score_V3"]
)

df["delta_srb_old_quality"] = (
    df["score_S1"] - df["score_V4"]
)

df["naive_damage_vs_detector"] = (
    df["score_D0"] - df["score_V3"]
)

df["detector_vlm_disagreement"] = (
    df["D"] - df["M"]
).abs()

df["srb_change_magnitude"] = (
    df["score_S1"] - df["score_D0"]
).abs()

cases = []

cases.append(
    select_cases(
        df,
        case_type="largest_srb_gain_over_detector",
        sort_column="delta_srb_detector",
        ascending=False,
    )
)

cases.append(
    select_cases(
        df,
        case_type="worst_srb_regression_vs_detector",
        sort_column="delta_srb_detector",
        ascending=True,
    )
)

cases.append(
    select_cases(
        df,
        case_type="largest_naive_fusion_damage",
        sort_column="naive_damage_vs_detector",
        ascending=False,
    )
)

cases.append(
    select_cases(
        df,
        case_type="largest_srb_repair_over_naive",
        sort_column="delta_srb_naive",
        ascending=False,
    )
)

cases.append(
    select_cases(
        df,
        case_type="largest_srb_repair_over_old_quality",
        sort_column="delta_srb_old_quality",
        ascending=False,
    )
)

gate_off = df[
    df["srb_pre_gate"] <= 0
].copy()

cases.append(
    select_cases(
        gate_off,
        case_type="gate_off_high_disagreement",
        sort_column="detector_vlm_disagreement",
        ascending=False,
    )
)

gate_on = df[
    df["srb_pre_gate"] > 0
].copy()

cases.append(
    select_cases(
        gate_on,
        case_type="gate_on_high_weight",
        sort_column="srb_weight",
        ascending=False,
    )
)

inventory = pd.concat(
    cases,
    ignore_index=True,
)

output_columns = [
    "case_type",
    "case_rank",
    "category",
    "image_path",
    "Y",
    "D",
    "M",
    "Q",
    "score_D0",
    "score_V3",
    "score_V4",
    "score_V6",
    "score_S1",
    "srb_pre_gate",
    "srb_weight",
    "delta_srb_detector",
    "delta_srb_naive",
    "delta_srb_old_quality",
    "naive_damage_vs_detector",
    "detector_vlm_disagreement",
]

inventory = inventory[
    output_columns
]

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

inventory.to_csv(
    OUT_CSV,
    index=False,
    lineterminator="\n",
)

summary = {
    "source": str(INPUT.relative_to(ROOT)),
    "source_rows": len(df),
    "source_categories": int(
        df["category"].nunique()
    ),
    "inventory_rows": len(inventory),
    "case_types": (
        inventory["case_type"]
        .value_counts()
        .sort_index()
        .to_dict()
    ),
    "category_counts": (
        inventory["category"]
        .value_counts()
        .sort_index()
        .to_dict()
    ),
    "largest_positive_delta_vs_detector": float(
        df["delta_srb_detector"].max()
    ),
    "worst_delta_vs_detector": float(
        df["delta_srb_detector"].min()
    ),
    "largest_naive_damage": float(
        df["naive_damage_vs_detector"].max()
    ),
    "largest_repair_over_old_quality": float(
        df["delta_srb_old_quality"].max()
    ),
}

OUT_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("===== CASE TYPES =====")
print(
    inventory["case_type"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n===== CATEGORY COVERAGE =====")
print(
    inventory["category"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n===== REPRESENTATIVE CASES =====")
print(
    inventory[
        [
            "case_type",
            "case_rank",
            "category",
            "Y",
            "D",
            "M",
            "Q",
            "score_S1",
            "delta_srb_detector",
            "image_path",
        ]
    ].to_string(index=False)
)

print()
print("[DONE]", OUT_CSV)
print("[DONE]", OUT_SUMMARY)
