from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()
RESULTS = ROOT / "results"
DATASETS = ROOT / "datasets"

OUT_CSV = (
    RESULTS
    / "stage22_selective_qcr"
    / "stage22_d2_mvtec15_asset_gap.csv"
)

OUT_TXT = (
    ROOT
    / "docs/stage22_selective_qcr"
    / "stage22_d2_mvtec15_asset_gap.txt"
)

MVTEC15 = [
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

CATEGORY_COLUMNS = [
    "category",
    "class_name",
    "object",
    "object_name",
    "cls_name",
]

DATASET_COLUMNS = [
    "dataset",
    "dataset_name",
]

PATH_COLUMNS = [
    "image_path",
    "canonical_image_path",
    "image_key",
    "filename",
]

LABEL_COLUMNS = {
    "gt_binary",
    "is_anomaly",
    "is_anomaly_final",
    "gt_label",
    "target",
    "y_true",
}

DETECTOR_COLUMNS = {
    "patchcore_score",
    "detector_score",
    "detector_score_norm",
    "detector_image_score",
    "d",
    "d_raw_patchcore",
    "image_anomaly_score",
}

VLM_COLUMNS = {
    "vlm_score",
    "vlm_score_norm",
    "vlm_anomaly_score",
    "crop_anomaly_score",
    "crop_vlm_margin",
    "context_anomaly_score",
    "context_vlm_margin",
    "tight_anomaly_score",
    "tight_vlm_margin",
    "m",
    "m_raw_crop_topk",
}

QUALITY_COLUMNS = {
    "candidate_quality",
    "candidate_quality_norm",
    "quality_score",
    "candidate_score_max",
    "candidate_score_mean",
    "q",
    "q_raw_candidate_quality",
}

CANDIDATE_COLUMNS = {
    "candidate_rank",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "map_x1",
    "map_y1",
    "map_x2",
    "map_y2",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def normalize(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def choose_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    normalized = {
        normalize(column): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def detect_roles(columns: list[str]) -> set[str]:
    names = {
        normalize(column)
        for column in columns
    }

    roles = set()

    if names.intersection(LABEL_COLUMNS):
        roles.add("label")

    if names.intersection(DETECTOR_COLUMNS):
        roles.add("detector")

    if names.intersection(VLM_COLUMNS):
        roles.add("vlm")

    if names.intersection(QUALITY_COLUMNS):
        roles.add("quality")

    if names.intersection(CANDIDATE_COLUMNS):
        roles.add("candidate")

    if names.intersection(
        {normalize(value) for value in PATH_COLUMNS}
    ):
        roles.add("path")

    return roles


def scan_dataset_directories() -> dict[str, list[str]]:
    found: dict[str, list[str]] = defaultdict(list)

    if not DATASETS.exists():
        return found

    for current, directories, _ in os.walk(DATASETS):
        for directory in directories:
            normalized = normalize(directory)

            if normalized in MVTEC15:
                path = Path(current) / directory
                found[normalized].append(
                    str(path.relative_to(ROOT))
                )

    return found


def count_images(directory_paths: list[str]) -> int:
    total = 0
    seen = set()

    for relative in directory_paths:
        root = ROOT / relative

        if not root.exists():
            continue

        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):
                resolved = str(
                    path.resolve(strict=False)
                )

                if resolved not in seen:
                    seen.add(resolved)
                    total += 1

    return total


def infer_categories_from_file_path(
    path: Path,
) -> set[str]:
    normalized = normalize(
        str(path.relative_to(ROOT))
    )

    return {
        category
        for category in MVTEC15
        if (
            f"/{category}/" in normalized
            or f"_{category}_" in normalized
            or normalized.endswith(
                f"_{category}.csv"
            )
        )
    }


def scan_csv_sources() -> dict[str, list[dict]]:
    sources: dict[str, list[dict]] = defaultdict(list)

    if not RESULTS.exists():
        return sources

    for path in sorted(RESULTS.rglob("*.csv")):
        relative = path.relative_to(ROOT)
        relative_text = normalize(relative)

        # Skip obviously unrelated large protocol tables.
        if not (
            "mvtec" in relative_text
            or "analysis" in relative_text
            or "baseline" in relative_text
            or any(
                category in relative_text
                for category in MVTEC15
            )
        ):
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as handle:
                columns = next(
                    csv.reader(handle),
                    [],
                )
        except Exception:
            continue

        if not columns:
            continue

        category_column = choose_column(
            columns,
            CATEGORY_COLUMNS,
        )

        dataset_column = choose_column(
            columns,
            DATASET_COLUMNS,
        )

        path_column = choose_column(
            columns,
            PATH_COLUMNS,
        )

        roles = detect_roles(columns)
        categories = infer_categories_from_file_path(
            path
        )

        usecols = [
            column
            for column in [
                category_column,
                dataset_column,
                path_column,
            ]
            if column is not None
        ]

        category_counts = defaultdict(int)
        dataset_values = set()

        if usecols:
            try:
                for chunk in pd.read_csv(
                    path,
                    usecols=usecols,
                    chunksize=50_000,
                    low_memory=False,
                ):
                    if dataset_column:
                        dataset_values.update(
                            str(value).strip()
                            for value in chunk[
                                dataset_column
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                        )

                    if category_column:
                        values = (
                            chunk[category_column]
                            .dropna()
                            .astype(str)
                            .map(normalize)
                        )

                        for value, count in (
                            values.value_counts().items()
                        ):
                            if value in MVTEC15:
                                categories.add(value)
                                category_counts[value] += int(
                                    count
                                )

                    if path_column:
                        paths = (
                            chunk[path_column]
                            .dropna()
                            .astype(str)
                            .map(normalize)
                        )

                        for category in MVTEC15:
                            mask = paths.str.contains(
                                rf"(^|/){category}(/|$)",
                                regex=True,
                            )

                            count = int(mask.sum())

                            if count:
                                categories.add(category)
                                category_counts[category] += count

            except Exception:
                pass

        # Reject files explicitly identified as MVTec AD 2.
        if (
            dataset_values
            and all(
                (
                    "ad 2" in value.lower()
                    or "ad2" in value.lower()
                )
                for value in dataset_values
            )
        ):
            continue

        for category in sorted(categories):
            sources[category].append(
                {
                    "path": str(relative),
                    "roles": sorted(roles),
                    "rows_for_category": int(
                        category_counts.get(
                            category,
                            0,
                        )
                    ),
                    "datasets": sorted(
                        dataset_values
                    ),
                    "columns": columns,
                }
            )

    return sources


def source_paths(
    records: list[dict],
    role: str,
    maximum: int = 5,
) -> list[str]:
    matching = [
        record
        for record in records
        if role in record["roles"]
    ]

    matching.sort(
        key=lambda record: (
            -record["rows_for_category"],
            record["path"],
        )
    )

    return [
        record["path"]
        for record in matching[:maximum]
    ]


def find_code_paths() -> list[str]:
    records = []

    roots = [
        ROOT / "experiments",
        ROOT / "scripts",
    ]

    for code_root in roots:
        if not code_root.exists():
            continue

        for path in code_root.rglob("*.py"):
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            lowered = text.lower()

            score = (
                3 * int("mvtec" in lowered)
                + 3 * int(
                    "candidate_regions.csv"
                    in lowered
                )
                + 3 * int(
                    "patchcore_image_predictions"
                    in lowered
                )
                + 2 * int(
                    "open_clip" in lowered
                )
                + 2 * int(
                    "encode_image" in lowered
                )
            )

            if score:
                records.append(
                    (
                        score,
                        str(
                            path.relative_to(ROOT)
                        ),
                    )
                )

    records.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    return [
        path
        for _, path in records[:30]
    ]


def main() -> None:
    dataset_directories = (
        scan_dataset_directories()
    )

    sources = scan_csv_sources()
    rows = []

    for category in MVTEC15:
        category_sources = sources.get(
            category,
            [],
        )

        roles = {
            role
            for record in category_sources
            for role in record["roles"]
        }

        has_label = "label" in roles
        has_detector = "detector" in roles
        has_vlm = "vlm" in roles
        has_quality = "quality" in roles
        has_candidates = "candidate" in roles
        has_path = "path" in roles

        quality_constructible = (
            has_quality or has_candidates
        )

        offline_ready = all(
            [
                has_label,
                has_detector,
                has_vlm,
                quality_constructible,
                has_path,
            ]
        )

        if offline_ready:
            status = "offline_ready"
        elif (
            has_label
            and has_detector
            and quality_constructible
            and has_path
            and not has_vlm
        ):
            status = "needs_vlm_only"
        elif (
            has_label
            and has_detector
            and has_vlm
            and has_path
            and not quality_constructible
        ):
            status = (
                "needs_quality_or_candidates"
            )
        elif not has_detector:
            status = (
                "needs_detector_and_candidates"
            )
        else:
            status = "partial_assets"

        directories = dataset_directories.get(
            category,
            [],
        )

        rows.append(
            {
                "category": category,
                "dataset_directory_found": bool(
                    directories
                ),
                "dataset_directories": " | ".join(
                    directories
                ),
                "dataset_image_count": (
                    count_images(directories)
                    if directories
                    else 0
                ),
                "has_label": has_label,
                "has_detector": has_detector,
                "has_vlm": has_vlm,
                "has_quality": has_quality,
                "has_candidates": has_candidates,
                "has_path": has_path,
                "quality_constructible": (
                    quality_constructible
                ),
                "offline_ready": offline_ready,
                "status": status,
                "detector_sources": " | ".join(
                    source_paths(
                        category_sources,
                        "detector",
                    )
                ),
                "vlm_sources": " | ".join(
                    source_paths(
                        category_sources,
                        "vlm",
                    )
                ),
                "quality_sources": " | ".join(
                    source_paths(
                        category_sources,
                        "quality",
                    )
                ),
                "candidate_sources": " | ".join(
                    source_paths(
                        category_sources,
                        "candidate",
                    )
                ),
                "label_sources": " | ".join(
                    source_paths(
                        category_sources,
                        "label",
                    )
                ),
                "num_relevant_csvs": len(
                    category_sources
                ),
            }
        )

    result = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_TXT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT_CSV,
        index=False,
        lineterminator="\n",
    )

    status_counts = (
        result["status"]
        .value_counts()
        .to_dict()
    )

    lines = [
        "===== STAGE 22-D2 MVTec15 ASSET GAP =====",
        "",
        "STATUS COUNTS",
        json.dumps(
            status_counts,
            indent=2,
            ensure_ascii=False,
        ),
        "",
        "PER CATEGORY",
        "",
    ]

    display_columns = [
        "category",
        "dataset_directory_found",
        "dataset_image_count",
        "has_detector",
        "has_vlm",
        "has_quality",
        "has_candidates",
        "offline_ready",
        "status",
    ]

    lines.append(
        result[display_columns]
        .to_string(index=False)
    )

    lines += [
        "",
        "SOURCE DETAILS",
        "",
    ]

    for _, row in result.iterrows():
        lines += [
            f"[{row['category']}]",
            f"status: {row['status']}",
            (
                "dataset directories: "
                + (
                    row["dataset_directories"]
                    or "none"
                )
            ),
            (
                "detector sources: "
                + (
                    row["detector_sources"]
                    or "none"
                )
            ),
            (
                "VLM sources: "
                + (
                    row["vlm_sources"]
                    or "none"
                )
            ),
            (
                "quality sources: "
                + (
                    row["quality_sources"]
                    or "none"
                )
            ),
            (
                "candidate sources: "
                + (
                    row["candidate_sources"]
                    or "none"
                )
            ),
            "",
        ]

    lines += [
        "LIKELY CODE PATHS",
        "",
        *find_code_paths(),
        "",
    ]

    OUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_TXT)
    print()
    print("STATUS COUNTS")
    print(
        json.dumps(
            status_counts,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print(
        result[display_columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
