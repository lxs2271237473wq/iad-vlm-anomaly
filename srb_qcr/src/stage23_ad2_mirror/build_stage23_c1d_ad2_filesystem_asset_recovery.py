from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

B2B = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_b2b_ad2_frozen_predictions.csv"
)

IMAGE = (
    ROOT
    / "results/stage11_mvtecad2_multicategory"
    / "stage11_d_vlm_image_predictions.csv"
)

OUT_DIR = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "ad2_actual_selective_runtime"
)

OUT_MAPPING = (
    OUT_DIR
    / "stage23_c1d_ad2_runtime_asset_mapping.csv"
)

OUT_JSON = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "stage23_c1d_ad2_filesystem_asset_recovery.json"
)

OUT_TXT = (
    ROOT
    / "docs/stage23_ad2_mirror"
    / "stage23_c1d_ad2_filesystem_asset_recovery.txt"
)

CATEGORIES = [
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "walnuts",
]

TOP_K = 3


def canonical_path(value: Any) -> str:
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


def resolve_existing_file(value: Any) -> Path | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    raw = Path(text)

    candidates = [
        raw,
        ROOT / raw,
        ROOT / "results" / raw,
        ROOT
        / "results/stage11_mvtecad2_multicategory"
        / raw,
        ROOT / "datasets" / raw,
        ROOT
        / "datasets/MVTec_AD_2_anomalib_all"
        / raw,
    ]

    normalized = canonical_path(text)

    candidates.extend(
        [
            ROOT
            / "datasets/MVTec_AD_2_anomalib_all"
            / normalized,
            ROOT / "datasets" / normalized,
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve(
            strict=False
        )

        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def as_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():
        return numeric.gt(0)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def candidate_rank(path: Path) -> int:
    match = re.search(
        r"_cand(\d+)_context_1\.50\.png$",
        path.name,
    )

    if match is None:
        raise RuntimeError(
            f"Could not parse candidate rank: {path}"
        )

    return int(match.group(1))


def expected_context_crops(
    best_crop_path: Path,
    image_stem: str,
    num_candidates: int,
    top_k: int,
) -> tuple[list[Path], list[Path]]:
    directory = best_crop_path.parent

    globbed = sorted(
        directory.glob(
            f"{image_stem}_cand*_context_1.50.png"
        ),
        key=candidate_rank,
    )

    required_count = min(
        int(num_candidates),
        int(top_k),
    )

    expected = [
        directory
        / (
            f"{image_stem}_cand"
            f"{rank:02d}_context_1.50.png"
        )
        for rank in range(
            required_count
        )
    ]

    return expected, globbed


def main() -> None:
    for path in [B2B, IMAGE]:
        if not path.exists():
            raise FileNotFoundError(path)

    b2b = pd.read_csv(B2B)
    image = pd.read_csv(IMAGE)

    for frame in [b2b, image]:
        frame["category"] = (
            frame["category"].astype(str)
        )

        frame["path_key"] = (
            frame["image_path"]
            .astype(str)
            .map(canonical_path)
        )

    b2b = b2b[
        b2b["category"].isin(CATEGORIES)
    ].copy()

    image = image[
        image["category"].isin(CATEGORIES)
    ].copy()

    b2b["gate_on"] = (
        pd.to_numeric(
            b2b["srb_pre_gate"],
            errors="raise",
        )
        > 0
    )

    image["num_candidates"] = pd.to_numeric(
        image["num_candidates"],
        errors="raise",
    ).astype(int)

    image_columns = [
        "category",
        "path_key",
        "num_candidates",
        "context_best_crop_path",
        "context_topk_mean_score",
        "full_image_score",
    ]

    image_ref = image[
        image_columns
    ].copy()

    if image_ref[
        ["category", "path_key"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate image-level prediction keys."
        )

    merged = b2b.merge(
        image_ref,
        on=["category", "path_key"],
        how="left",
        validate="one_to_one",
    )

    if merged["num_candidates"].isna().any():
        bad = merged[
            merged["num_candidates"].isna()
        ]

        raise RuntimeError(
            "Missing image-level rows:\n"
            + bad[
                [
                    "category",
                    "image_path",
                    "path_key",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    records = []

    for _, row in merged.iterrows():
        original_image = resolve_existing_file(
            row["image_path"]
        )

        if original_image is None:
            raise RuntimeError(
                "Could not resolve original image: "
                f"{row['image_path']}"
            )

        num_candidates = int(
            row["num_candidates"]
        )

        gate_on = bool(
            row["gate_on"]
        )

        best_crop = resolve_existing_file(
            row["context_best_crop_path"]
        )

        runtime_paths = []
        globbed_paths = []
        missing_expected = []

        if num_candidates > 0:
            if best_crop is None:
                records.append(
                    {
                        "category": row["category"],
                        "path_key": row["path_key"],
                        "image_path": str(
                            original_image
                        ),
                        "gate_on": gate_on,
                        "num_candidates": num_candidates,
                        "asset_mode": (
                            "context_topk_missing_best"
                        ),
                        "runtime_crop_paths": "[]",
                        "num_eval_images": 0,
                        "best_crop_path": None,
                        "globbed_crop_count": 0,
                        "missing_expected_count": (
                            min(
                                num_candidates,
                                TOP_K,
                            )
                        ),
                        "all_assets_ready": False,
                    }
                )

                continue

            expected, globbed = (
                expected_context_crops(
                    best_crop_path=best_crop,
                    image_stem=(
                        original_image.stem
                    ),
                    num_candidates=(
                        num_candidates
                    ),
                    top_k=TOP_K,
                )
            )

            runtime_paths = [
                path.resolve(strict=False)
                for path in expected
                if path.exists()
                and path.is_file()
            ]

            globbed_paths = [
                path.resolve(strict=False)
                for path in globbed
            ]

            missing_expected = [
                path
                for path in expected
                if not (
                    path.exists()
                    and path.is_file()
                )
            ]

            asset_mode = (
                "context_1p50_topk_mean"
            )

            ready = (
                len(runtime_paths)
                == min(
                    num_candidates,
                    TOP_K,
                )
                and not missing_expected
            )

        else:
            runtime_paths = [
                original_image
            ]

            asset_mode = (
                "full_image_fallback"
            )

            ready = True

        records.append(
            {
                "category": row["category"],
                "path_key": row["path_key"],
                "image_path": str(
                    original_image
                ),
                "gate_on": gate_on,
                "num_candidates": (
                    num_candidates
                ),
                "asset_mode": asset_mode,
                "runtime_crop_paths": (
                    json.dumps(
                        [
                            str(path)
                            for path in runtime_paths
                        ],
                        ensure_ascii=False,
                    )
                ),
                "num_eval_images": len(
                    runtime_paths
                ),
                "best_crop_path": (
                    str(best_crop)
                    if best_crop is not None
                    else None
                ),
                "globbed_crop_count": len(
                    globbed_paths
                ),
                "missing_expected_count": len(
                    missing_expected
                ),
                "all_assets_ready": ready,
                "reference_m_raw": float(
                    row[
                        "context_topk_mean_score"
                    ]
                ),
                "reference_full_image_score": float(
                    row["full_image_score"]
                ),
            }
        )

    mapping = pd.DataFrame(records)

    full_calls = int(
        len(mapping)
    )

    selective_calls = int(
        mapping["gate_on"].sum()
    )

    context_rows = int(
        (
            mapping["asset_mode"]
            == "context_1p50_topk_mean"
        ).sum()
    )

    fallback_rows = int(
        (
            mapping["asset_mode"]
            == "full_image_fallback"
        ).sum()
    )

    gate_on_without_context = int(
        (
            mapping["gate_on"]
            & (
                mapping["asset_mode"]
                != "context_1p50_topk_mean"
            )
        ).sum()
    )

    context_rows_missing_assets = int(
        (
            (
                mapping["asset_mode"]
                == "context_1p50_topk_mean"
            )
            & ~mapping["all_assets_ready"]
        ).sum()
    )

    fallback_reference_diff = (
        mapping.loc[
            mapping["asset_mode"]
            == "full_image_fallback",
            "reference_m_raw",
        ]
        - mapping.loc[
            mapping["asset_mode"]
            == "full_image_fallback",
            "reference_full_image_score",
        ]
    ).abs()

    max_fallback_reference_diff = (
        float(
            fallback_reference_diff.max()
        )
        if len(
            fallback_reference_diff
        )
        else 0.0
    )

    ready = (
        full_calls == 243
        and selective_calls == 182
        and gate_on_without_context == 0
        and context_rows_missing_assets == 0
        and mapping[
            "all_assets_ready"
        ].all()
        and max_fallback_reference_diff
        <= 1e-12
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_TXT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping.to_csv(
        OUT_MAPPING,
        index=False,
        lineterminator="\n",
    )

    payload = {
        "protocol_id": (
            "stage23_c1d_ad2_"
            "filesystem_asset_recovery_v1"
        ),
        "top_k": TOP_K,
        "rows": full_calls,
        "full_vlm_calls": full_calls,
        "selective_vlm_calls": (
            selective_calls
        ),
        "calls_saved": (
            full_calls - selective_calls
        ),
        "call_saving_rate": (
            1.0
            - selective_calls
            / full_calls
        ),
        "context_rows": context_rows,
        "fallback_rows": fallback_rows,
        "gate_on_without_context": (
            gate_on_without_context
        ),
        "context_rows_missing_assets": (
            context_rows_missing_assets
        ),
        "max_fallback_reference_diff": (
            max_fallback_reference_diff
        ),
        "mapping_csv": str(
            OUT_MAPPING.relative_to(ROOT)
        ),
        "runtime_runner_ready": ready,
    }

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
        "===== STAGE 23-C1d FILESYSTEM ASSET RECOVERY =====",
        "",
        "===== COUNTS =====",
        f"rows: {full_calls}",
        f"full VLM calls: {full_calls}",
        f"selective VLM calls: {selective_calls}",
        f"calls saved: {full_calls - selective_calls}",
        (
            "call saving rate: "
            f"{1.0 - selective_calls / full_calls:.6f}"
        ),
        f"context rows: {context_rows}",
        f"full-image fallback rows: {fallback_rows}",
        "",
        "===== CONSISTENCY =====",
        (
            "gate-on rows without context assets: "
            f"{gate_on_without_context}"
        ),
        (
            "context rows missing expected crop files: "
            f"{context_rows_missing_assets}"
        ),
        (
            "max fallback context-vs-full score diff: "
            f"{max_fallback_reference_diff}"
        ),
        "",
        "===== ASSET MODE VS GATE =====",
        pd.crosstab(
            mapping["asset_mode"],
            mapping["gate_on"],
            margins=True,
        ).to_string(),
        "",
        "===== PER-CATEGORY =====",
        mapping.groupby("category")
        .agg(
            images=("path_key", "size"),
            gate_on=("gate_on", "sum"),
            context_rows=(
                "asset_mode",
                lambda values: int(
                    (
                        values
                        == "context_1p50_topk_mean"
                    ).sum()
                ),
            ),
            fallback_rows=(
                "asset_mode",
                lambda values: int(
                    (
                        values
                        == "full_image_fallback"
                    ).sum()
                ),
            ),
            eval_images=(
                "num_eval_images",
                "sum",
            ),
            missing_assets=(
                "all_assets_ready",
                lambda values: int(
                    (~values).sum()
                ),
            ),
        )
        .to_string(),
        "",
        "===== MISSING EXAMPLES =====",
    ]

    missing = mapping[
        ~mapping["all_assets_ready"]
    ]

    if missing.empty:
        lines.append("NONE")
    else:
        lines.append(
            missing[
                [
                    "category",
                    "image_path",
                    "num_candidates",
                    "best_crop_path",
                    "globbed_crop_count",
                    "missing_expected_count",
                    "runtime_crop_paths",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    lines += [
        "",
        "===== DECISION =====",
        (
            "all_243_assets_ready: "
            f"{mapping['all_assets_ready'].all()}"
        ),
        (
            "all_182_gate_on_rows_use_context: "
            f"{gate_on_without_context == 0}"
        ),
        (
            "fallback_semantics_exact: "
            f"{max_fallback_reference_diff <= 1e-12}"
        ),
        f"runtime_runner_ready: {ready}",
        "",
        f"[DONE] {OUT_MAPPING}",
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
