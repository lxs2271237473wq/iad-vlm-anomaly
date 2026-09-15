from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

CATEGORIES = [
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "walnuts",
]

PREDICTIONS = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "ad2_frozen_mirror"
    / "stage23_b1_ad2_unified_predictions.csv"
)

SCAN_ROOTS = [
    ROOT / "results/stage11_mvtecad2_multicategory",
    ROOT / "results/stage18_ad2_qcr_ablation",
    ROOT / "results/stage22_selective_qcr",
    ROOT / "datasets/MVTec_AD_2_anomalib_all",
]

OUT_JSON = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "stage23_c0_ad2_runtime_preflight.json"
)

OUT_TXT = (
    ROOT
    / "docs/stage23_ad2_mirror"
    / "stage23_c0_ad2_runtime_preflight.txt"
)

PATH_COLUMNS = [
    "image_path",
    "canonical_image_path",
    "path_key",
    "image_key",
]

CATEGORY_COLUMNS = [
    "category",
    "object",
    "objects",
    "class_name",
]

CANDIDATE_TOKENS = {
    "component_rank",
    "candidate_rank",
    "candidate_available",
    "x1",
    "y1",
    "x2",
    "y2",
    "map_x1",
    "map_y1",
    "map_x2",
    "map_y2",
    "tight_crop_path",
    "context_crop_path",
}

VLM_TOKENS = {
    "m",
    "score_m0",
    "vlm_score_norm",
    "tight_vlm_margin",
    "context_vlm_margin",
    "full_image_score",
    "pred_score",
}

QUALITY_TOKENS = {
    "q",
    "candidate_quality",
    "candidate_quality_norm",
    "candidate_score_max",
    "candidate_score_mean",
}

GATE_TOKENS = {
    "srb_pre_gate",
    "srb_weight",
    "score_s1",
}


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


def choose_column(columns, candidates):
    lower = {
        str(column).lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def inspect_csv(path: Path) -> dict | None:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return None

    columns = [str(column) for column in header.columns]
    lower = {column.lower() for column in columns}

    category_column = choose_column(
        columns,
        CATEGORY_COLUMNS,
    )

    path_column = choose_column(
        columns,
        PATH_COLUMNS,
    )

    roles = []

    if lower & CANDIDATE_TOKENS:
        roles.append("candidate")

    if lower & VLM_TOKENS:
        roles.append("vlm")

    if lower & QUALITY_TOKENS:
        roles.append("quality")

    if lower & GATE_TOKENS:
        roles.append("gate")

    if path_column is not None:
        roles.append("path")

    observed_categories = []

    try:
        usecols = []

        if category_column is not None:
            usecols.append(category_column)

        if path_column is not None:
            usecols.append(path_column)

        sample = pd.read_csv(
            path,
            usecols=usecols or None,
            nrows=50000,
        )

        if category_column is not None:
            observed_categories = sorted(
                set(
                    sample[
                        category_column
                    ]
                    .dropna()
                    .astype(str)
                )
                & set(CATEGORIES)
            )

    except Exception:
        sample = None

    score = (
        20 * len(observed_categories)
        + 6 * len(roles)
        + 5 * int("ad2" in str(path).lower())
        + 5 * int("mvtecad2" in str(path).lower())
        + 4 * int("stage11" in str(path).lower())
    )

    if score <= 0:
        return None

    try:
        row_count = max(
            sum(
                1
                for _ in path.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                )
            )
            - 1,
            0,
        )
    except Exception:
        row_count = None

    return {
        "path": str(path.relative_to(ROOT)),
        "score": score,
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
        "observed_categories": observed_categories,
        "roles": roles,
        "category_column": category_column,
        "path_column": path_column,
        "columns": columns,
    }


def resolve_existing_image(
    raw_path: str,
) -> Path | None:
    candidate = Path(str(raw_path))

    candidates = [
        candidate,
        ROOT / candidate,
        ROOT / "datasets" / candidate,
        ROOT
        / "datasets/MVTec_AD_2_anomalib_all"
        / candidate,
    ]

    normalized = normalize_path(raw_path)

    candidates.extend(
        [
            ROOT
            / "datasets/MVTec_AD_2_anomalib_all"
            / normalized,
            ROOT / "datasets" / normalized,
        ]
    )

    for item in candidates:
        item = item.resolve(strict=False)

        if item.exists() and item.is_file():
            return item

    return None


def main() -> None:
    if not PREDICTIONS.exists():
        raise FileNotFoundError(PREDICTIONS)

    pred = pd.read_csv(PREDICTIONS)

    required = {
        "category",
        "path_key",
        "Y",
        "score_D0",
        "score_M0",
        "score_S1",
        "srb_pre_gate",
    }

    missing = sorted(
        required - set(pred.columns)
    )

    if missing:
        raise RuntimeError(
            f"Unified predictions missing {missing}"
        )

    pred = pred[
        pred["category"]
        .astype(str)
        .isin(CATEGORIES)
    ].copy()

    pred["srb_pre_gate"] = pd.to_numeric(
        pred["srb_pre_gate"],
        errors="coerce",
    )

    if pred["srb_pre_gate"].isna().any():
        raise RuntimeError(
            "srb_pre_gate contains missing values."
        )

    pred["path_key"] = (
        pred["path_key"]
        .astype(str)
        .map(normalize_path)
    )

    if pred["path_key"].duplicated().any():
        raise RuntimeError(
            "Duplicate path_key values in unified predictions."
        )

    image_checks = []

    for _, row in pred.iterrows():
        resolved = resolve_existing_image(
            row["path_key"]
        )

        image_checks.append(
            {
                "category": row["category"],
                "path_key": row["path_key"],
                "resolved_path": (
                    str(resolved)
                    if resolved is not None
                    else None
                ),
                "exists": resolved is not None,
                "gate": int(
                    row["srb_pre_gate"] > 0
                ),
            }
        )

    image_checks_df = pd.DataFrame(
        image_checks
    )

    csv_inventory = []

    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue

        for path in scan_root.rglob("*.csv"):
            record = inspect_csv(path)

            if record is not None:
                csv_inventory.append(record)

    csv_inventory = sorted(
        csv_inventory,
        key=lambda record: (
            -record["score"],
            record["path"],
        ),
    )

    per_category = (
        pred.groupby("category")
        .agg(
            images=("path_key", "size"),
            selective_calls=(
                "srb_pre_gate",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
        )
        .reset_index()
    )

    per_category["calls_saved"] = (
        per_category["images"]
        - per_category["selective_calls"]
    )

    per_category["saving_rate"] = (
        per_category["calls_saved"]
        / per_category["images"]
    )

    payload = {
        "protocol_id": (
            "stage23_c0_ad2_runtime_preflight_v1"
        ),
        "categories": CATEGORIES,
        "prediction_csv": str(
            PREDICTIONS.relative_to(ROOT)
        ),
        "num_images": int(len(pred)),
        "full_vlm_calls": int(len(pred)),
        "selective_vlm_calls": int(
            (pred["srb_pre_gate"] > 0).sum()
        ),
        "actual_calls_saved": int(
            (pred["srb_pre_gate"] <= 0).sum()
        ),
        "call_saving_rate": float(
            (pred["srb_pre_gate"] <= 0).mean()
        ),
        "image_paths_resolved": int(
            image_checks_df["exists"].sum()
        ),
        "image_paths_missing": int(
            (~image_checks_df["exists"]).sum()
        ),
        "per_category": (
            per_category.to_dict(
                orient="records"
            )
        ),
        "top_csv_inventory": csv_inventory[:30],
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
    )

    lines = [
        "===== STAGE 23-C0 AD2 RUNTIME PREFLIGHT =====",
        "",
        "===== GATE COUNTS =====",
        f"images: {len(pred)}",
        f"full VLM calls: {len(pred)}",
        (
            "selective VLM calls: "
            f"{int((pred['srb_pre_gate'] > 0).sum())}"
        ),
        (
            "actual calls saved: "
            f"{int((pred['srb_pre_gate'] <= 0).sum())}"
        ),
        (
            "call saving rate: "
            f"{float((pred['srb_pre_gate'] <= 0).mean()):.6f}"
        ),
        "",
        "===== PER-CATEGORY =====",
        per_category.to_string(index=False),
        "",
        "===== IMAGE PATH RESOLUTION =====",
        (
            "resolved: "
            f"{int(image_checks_df['exists'].sum())}"
        ),
        (
            "missing: "
            f"{int((~image_checks_df['exists']).sum())}"
        ),
        "",
        "===== TOP AD2 CSV ASSETS =====",
    ]

    for index, record in enumerate(
        csv_inventory[:20],
        start=1,
    ):
        lines += [
            f"[{index}] {record['path']}",
            (
                f"  score={record['score']} "
                f"rows={record['row_count']} "
                f"roles={record['roles']}"
            ),
            (
                "  categories="
                f"{record['observed_categories']}"
            ),
            (
                "  path_column="
                f"{record['path_column']}"
            ),
            (
                "  columns="
                + ", ".join(
                    record["columns"]
                )
            ),
        ]

    ready = (
        set(pred["category"].unique())
        == set(CATEGORIES)
        and not pred["path_key"].duplicated().any()
        and int(
            image_checks_df["exists"].sum()
        )
        == len(pred)
        and int(
            (pred["srb_pre_gate"] > 0).sum()
        )
        > 0
        and int(
            (pred["srb_pre_gate"] <= 0).sum()
        )
        > 0
    )

    lines += [
        "",
        "===== DECISION =====",
        (
            "all_categories_ready: "
            f"{set(pred['category'].unique()) == set(CATEGORIES)}"
        ),
        (
            "all_image_paths_resolved: "
            f"{int(image_checks_df['exists'].sum()) == len(pred)}"
        ),
        (
            "both_gate_states_present: "
            f"{int((pred['srb_pre_gate'] > 0).sum()) > 0 and int((pred['srb_pre_gate'] <= 0).sum()) > 0}"
        ),
        f"runtime_stage_ready: {ready}",
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
