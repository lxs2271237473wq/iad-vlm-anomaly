from __future__ import annotations

import hashlib
import json
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

CAND = (
    ROOT
    / "results/stage11_mvtecad2_multicategory"
    / "stage11_d_vlm_candidate_scores.csv"
)

OUT_DIR = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "ad2_actual_selective_runtime"
)

OUT_MAPPING = (
    OUT_DIR
    / "stage23_c1c_ad2_crop_group_mapping.csv"
)

OUT_JSON = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "stage23_c1c_ad2_crop_identity_bridge.json"
)

OUT_TXT = (
    ROOT
    / "docs/stage23_ad2_mirror"
    / "stage23_c1c_ad2_crop_identity_bridge.txt"
)

CATEGORIES = [
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "walnuts",
]


def canonical_image_path(value: Any) -> str:
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
    ]

    for candidate in candidates:
        candidate = candidate.expanduser().resolve(
            strict=False
        )

        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def crop_suffix(path: Path | None) -> str | None:
    if path is None:
        return None

    text = str(path).replace("\\", "/")

    for marker in [
        "/stage11_c_candidate_crops/",
        "stage11_c_candidate_crops/",
    ]:
        if marker in text:
            return text.split(marker, 1)[1]

    return path.name


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def prepare():
    b2b = pd.read_csv(B2B)
    image = pd.read_csv(IMAGE)
    cand = pd.read_csv(CAND)

    for frame in [b2b, image, cand]:
        frame["category"] = (
            frame["category"].astype(str)
        )

        frame["path_key"] = (
            frame["image_path"]
            .astype(str)
            .map(canonical_image_path)
        )

    b2b = b2b[
        b2b["category"].isin(CATEGORIES)
    ].copy()

    image = image[
        image["category"].isin(CATEGORIES)
    ].copy()

    cand = cand[
        cand["category"].isin(CATEGORIES)
    ].copy()

    b2b["gate_on"] = (
        pd.to_numeric(
            b2b["srb_pre_gate"],
            errors="raise",
        )
        > 0
    )

    for column in [
        "M_raw_crop_topk",
        "M",
    ]:
        b2b[column] = pd.to_numeric(
            b2b[column],
            errors="raise",
        )

    image["context_topk_mean_score"] = (
        pd.to_numeric(
            image[
                "context_topk_mean_score"
            ],
            errors="raise",
        )
    )

    cand["candidate_rank"] = pd.to_numeric(
        cand["candidate_rank"],
        errors="coerce",
    )

    cand["batch_idx"] = pd.to_numeric(
        cand["batch_idx"],
        errors="coerce",
    ).astype("Int64")

    cand["item_idx"] = pd.to_numeric(
        cand["item_idx"],
        errors="coerce",
    ).astype("Int64")

    cand["context_vlm_margin"] = (
        pd.to_numeric(
            cand["context_vlm_margin"],
            errors="coerce",
        )
    )

    cand = cand[
        (cand["candidate_rank"] > 0)
        & cand[
            "context_1p50_crop_path"
        ].notna()
        & cand["batch_idx"].notna()
        & cand["item_idx"].notna()
        & cand[
            "context_vlm_margin"
        ].notna()
    ].copy()

    return b2b, image, cand


def build_candidate_identity(
    cand: pd.DataFrame,
):
    identity = cand.copy()

    identity["resolved_crop_path"] = (
        identity[
            "context_1p50_crop_path"
        ]
        .map(resolve_existing_file)
    )

    missing_files = identity[
        identity[
            "resolved_crop_path"
        ].isna()
    ]

    if not missing_files.empty:
        raise RuntimeError(
            "Missing candidate crop files:\n"
            + missing_files[
                [
                    "category",
                    "image_path",
                    "context_1p50_crop_path",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    identity["resolved_crop_path"] = (
        identity[
            "resolved_crop_path"
        ].map(str)
    )

    identity["crop_basename"] = (
        identity[
            "resolved_crop_path"
        ].map(
            lambda value: Path(value).name
        )
    )

    identity["crop_suffix"] = (
        identity[
            "resolved_crop_path"
        ].map(
            lambda value: crop_suffix(
                Path(value)
            )
        )
    )

    print(
        "[HASH] candidate context crops:",
        len(identity),
    )

    identity["crop_sha256"] = (
        identity[
            "resolved_crop_path"
        ].map(
            lambda value: sha256_file(
                Path(value)
            )
        )
    )

    identity["group_id"] = (
        identity["category"]
        + "::"
        + identity[
            "batch_idx"
        ].astype(str)
        + "::"
        + identity[
            "item_idx"
        ].astype(str)
    )

    groups = (
        identity.groupby(
            [
                "category",
                "batch_idx",
                "item_idx",
                "group_id",
            ],
            as_index=False,
        )
        .agg(
            candidate_rows=(
                "candidate_rank",
                "size",
            ),
            candidate_image_path=(
                "image_path",
                "first",
            ),
            candidate_context_mean=(
                "context_vlm_margin",
                "mean",
            ),
            first_context_crop=(
                "resolved_crop_path",
                "first",
            ),
        )
    )

    return identity, groups


def unique_bridge(
    identity: pd.DataFrame,
    key_column: str,
) -> pd.DataFrame:
    bridge = (
        identity.groupby(
            [
                "category",
                key_column,
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            group_count=(
                "group_id",
                "nunique",
            ),
            group_id=(
                "group_id",
                "first",
            ),
            resolved_batch_idx=(
                "batch_idx",
                "first",
            ),
            resolved_item_idx=(
                "item_idx",
                "first",
            ),
        )
    )

    return bridge[
        bridge["group_count"] == 1
    ].copy()


def main() -> None:
    for path in [B2B, IMAGE, CAND]:
        if not path.exists():
            raise FileNotFoundError(path)

    b2b, image, cand = prepare()

    identity, groups = (
        build_candidate_identity(cand)
    )

    direct_keys = set(
        identity["path_key"]
    )

    missing = b2b[
        b2b["gate_on"]
        & ~b2b["path_key"].isin(
            direct_keys
        )
    ].copy()

    image_ref = image[
        [
            "category",
            "path_key",
            "num_candidates",
            "context_best_crop_path",
            "context_topk_mean_score",
        ]
    ].copy()

    if image_ref[
        ["category", "path_key"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate image-level reference keys."
        )

    mapped = missing.merge(
        image_ref,
        on=["category", "path_key"],
        how="left",
        validate="one_to_one",
    )

    mapped["resolved_best_crop_path"] = (
        mapped[
            "context_best_crop_path"
        ].map(resolve_existing_file)
    )

    if mapped[
        "resolved_best_crop_path"
    ].isna().any():
        bad = mapped[
            mapped[
                "resolved_best_crop_path"
            ].isna()
        ]

        raise RuntimeError(
            "Missing best-crop files:\n"
            + bad[
                [
                    "category",
                    "image_path",
                    "context_best_crop_path",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    mapped[
        "resolved_best_crop_path"
    ] = mapped[
        "resolved_best_crop_path"
    ].map(str)

    mapped["best_crop_basename"] = (
        mapped[
            "resolved_best_crop_path"
        ].map(
            lambda value: Path(value).name
        )
    )

    mapped["best_crop_suffix"] = (
        mapped[
            "resolved_best_crop_path"
        ].map(
            lambda value: crop_suffix(
                Path(value)
            )
        )
    )

    print(
        "[HASH] image-level best crops:",
        len(mapped),
    )

    mapped["best_crop_sha256"] = (
        mapped[
            "resolved_best_crop_path"
        ].map(
            lambda value: sha256_file(
                Path(value)
            )
        )
    )

    bridge_specs = [
        (
            "resolved_path",
            "resolved_best_crop_path",
            "resolved_crop_path",
        ),
        (
            "relative_suffix",
            "best_crop_suffix",
            "crop_suffix",
        ),
        (
            "basename",
            "best_crop_basename",
            "crop_basename",
        ),
        (
            "sha256",
            "best_crop_sha256",
            "crop_sha256",
        ),
    ]

    diagnostics = []

    mapped["selected_bridge"] = None
    mapped["resolved_group_id"] = None
    mapped["resolved_batch_idx"] = pd.NA
    mapped["resolved_item_idx"] = pd.NA

    for (
        bridge_name,
        left_column,
        right_column,
    ) in bridge_specs:
        bridge = unique_bridge(
            identity,
            right_column,
        ).rename(
            columns={
                right_column: left_column,
                "group_id": (
                    f"{bridge_name}_group_id"
                ),
                "resolved_batch_idx": (
                    f"{bridge_name}_batch_idx"
                ),
                "resolved_item_idx": (
                    f"{bridge_name}_item_idx"
                ),
            }
        )

        trial = mapped[
            [
                "category",
                left_column,
            ]
        ].merge(
            bridge[
                [
                    "category",
                    left_column,
                    f"{bridge_name}_group_id",
                    f"{bridge_name}_batch_idx",
                    f"{bridge_name}_item_idx",
                ]
            ],
            on=[
                "category",
                left_column,
            ],
            how="left",
            validate="many_to_one",
        )

        group_column = (
            f"{bridge_name}_group_id"
        )

        batch_column = (
            f"{bridge_name}_batch_idx"
        )

        item_column = (
            f"{bridge_name}_item_idx"
        )

        matched = trial[
            group_column
        ].notna()

        unresolved_now = mapped[
            "resolved_group_id"
        ].isna()

        take = matched & unresolved_now

        mapped.loc[
            take,
            "selected_bridge",
        ] = bridge_name

        mapped.loc[
            take,
            "resolved_group_id",
        ] = trial.loc[
            take,
            group_column,
        ].to_numpy()

        mapped.loc[
            take,
            "resolved_batch_idx",
        ] = trial.loc[
            take,
            batch_column,
        ].to_numpy()

        mapped.loc[
            take,
            "resolved_item_idx",
        ] = trial.loc[
            take,
            item_column,
        ].to_numpy()

        diagnostics.append(
            {
                "bridge": bridge_name,
                "unique_candidate_keys": int(
                    len(bridge)
                ),
                "matched_rows": int(
                    matched.sum()
                ),
                "newly_recovered_rows": int(
                    take.sum()
                ),
            }
        )

    mapped["resolved"] = (
        mapped[
            "resolved_group_id"
        ].notna()
    )

    resolved = mapped[
        mapped["resolved"]
    ].copy()

    unresolved = mapped[
        ~mapped["resolved"]
    ].copy()

    resolved["resolved_batch_idx"] = (
        pd.to_numeric(
            resolved[
                "resolved_batch_idx"
            ],
            errors="raise",
        ).astype("Int64")
    )

    resolved["resolved_item_idx"] = (
        pd.to_numeric(
            resolved[
                "resolved_item_idx"
            ],
            errors="raise",
        ).astype("Int64")
    )

    resolved = resolved.merge(
        groups.rename(
            columns={
                "batch_idx":
                    "resolved_batch_idx",
                "item_idx":
                    "resolved_item_idx",
                "group_id":
                    "resolved_group_id",
            }
        ),
        on=[
            "category",
            "resolved_batch_idx",
            "resolved_item_idx",
            "resolved_group_id",
        ],
        how="left",
        validate="one_to_one",
    )

    resolved[
        "image_vs_candidate_mean_abs_diff"
    ] = (
        resolved[
            "context_topk_mean_score"
        ]
        - resolved[
            "candidate_context_mean"
        ]
    ).abs()

    resolved[
        "b2b_vs_candidate_mean_abs_diff"
    ] = (
        resolved[
            "M_raw_crop_topk"
        ]
        - resolved[
            "candidate_context_mean"
        ]
    ).abs()

    mapping_columns = [
        "category",
        "image_path",
        "path_key",
        "context_best_crop_path",
        "resolved_best_crop_path",
        "selected_bridge",
        "resolved_group_id",
        "resolved_batch_idx",
        "resolved_item_idx",
        "candidate_rows",
        "candidate_image_path",
        "first_context_crop",
        "M_raw_crop_topk",
        "context_topk_mean_score",
        "candidate_context_mean",
        "image_vs_candidate_mean_abs_diff",
        "b2b_vs_candidate_mean_abs_diff",
    ]

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

    resolved[
        mapping_columns
    ].to_csv(
        OUT_MAPPING,
        index=False,
        lineterminator="\n",
    )

    max_image_diff = (
        float(
            resolved[
                "image_vs_candidate_mean_abs_diff"
            ].max()
        )
        if not resolved.empty
        else None
    )

    max_b2b_diff = (
        float(
            resolved[
                "b2b_vs_candidate_mean_abs_diff"
            ].max()
        )
        if not resolved.empty
        else None
    )

    ready = (
        len(mapped) == 49
        and len(resolved) == 49
        and unresolved.empty
        and max_image_diff is not None
        and max_image_diff <= 1e-12
        and max_b2b_diff is not None
        and max_b2b_diff <= 1e-12
    )

    payload = {
        "protocol_id": (
            "stage23_c1c_ad2_"
            "crop_identity_bridge_v1"
        ),
        "missing_gate_on_rows": int(
            len(mapped)
        ),
        "recovered_rows": int(
            len(resolved)
        ),
        "unresolved_rows": int(
            len(unresolved)
        ),
        "bridge_diagnostics": diagnostics,
        "selected_bridge_counts": (
            resolved[
                "selected_bridge"
            ]
            .value_counts()
            .to_dict()
        ),
        "max_image_vs_candidate_mean_abs_diff": (
            max_image_diff
        ),
        "max_b2b_vs_candidate_mean_abs_diff": (
            max_b2b_diff
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
        "===== STAGE 23-C1c CROP IDENTITY BRIDGE =====",
        "",
        "===== INPUT =====",
        f"missing gate-on rows: {len(mapped)}",
        f"candidate crop rows: {len(identity)}",
        "",
        "===== BRIDGE DIAGNOSTICS =====",
    ]

    for row in diagnostics:
        lines.append(
            (
                f"{row['bridge']}: "
                f"unique_candidate_keys="
                f"{row['unique_candidate_keys']}, "
                f"matched_rows="
                f"{row['matched_rows']}, "
                f"newly_recovered_rows="
                f"{row['newly_recovered_rows']}"
            )
        )

    lines += [
        "",
        "===== RECOVERY =====",
        f"recovered rows: {len(resolved)} / {len(mapped)}",
        f"unresolved rows: {len(unresolved)}",
        (
            "selected bridge counts: "
            f"{payload['selected_bridge_counts']}"
        ),
        (
            "max image-vs-candidate mean diff: "
            f"{max_image_diff}"
        ),
        (
            "max B2b-vs-candidate mean diff: "
            f"{max_b2b_diff}"
        ),
        "",
        "===== PER-CATEGORY =====",
    ]

    if resolved.empty:
        lines.append("No rows recovered.")
    else:
        lines.append(
            resolved.groupby("category")
            .agg(
                recovered=(
                    "resolved_group_id",
                    "size",
                ),
                max_image_diff=(
                    "image_vs_candidate_mean_abs_diff",
                    "max",
                ),
                max_b2b_diff=(
                    "b2b_vs_candidate_mean_abs_diff",
                    "max",
                ),
            )
            .to_string()
        )

    lines += [
        "",
        "===== UNRESOLVED EXAMPLES =====",
    ]

    if unresolved.empty:
        lines.append("NONE")
    else:
        lines.append(
            unresolved[
                [
                    "category",
                    "image_path",
                    "context_best_crop_path",
                    "resolved_best_crop_path",
                    "best_crop_suffix",
                    "best_crop_basename",
                    "best_crop_sha256",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    lines += [
        "",
        "===== DECISION =====",
        f"all_49_recovered: {len(resolved) == 49}",
        (
            "cached_mean_exact: "
            f"{max_image_diff is not None and max_image_diff <= 1e-12 and max_b2b_diff is not None and max_b2b_diff <= 1e-12}"
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
