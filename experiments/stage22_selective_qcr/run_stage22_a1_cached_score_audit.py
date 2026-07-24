from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

SEARCH_ROOTS = [
    ROOT / "results",
    ROOT / "experiments",
]

OUTPUT_JSON = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_a1_cached_score_audit.json"
)

OUTPUT_REPORT = (
    ROOT
    / "docs/stage22_selective_qcr"
    / "stage22_a1_cached_score_audit.md"
)

SUPPORTED_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".npz",
    ".npy",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
}

PATH_KEYWORDS = {
    "qcr": 6,
    "stage16": 5,
    "stage18": 5,
    "visa": 4,
    "ad2": 4,
    "ablation": 3,
    "score": 3,
    "prediction": 3,
    "sample": 2,
    "cache": 2,
    "raw": 1,
}

FIELD_GROUPS = {
    "detector": [
        "d",
        "detector",
        "detector_score",
        "base_score",
        "patchcore_score",
        "fastflow_score",
        "anomaly_score",
    ],
    "vlm": [
        "m",
        "vlm",
        "vlm_score",
        "crop_vlm",
        "crop_vlm_score",
        "semantic_score",
    ],
    "quality": [
        "q",
        "quality",
        "quality_score",
        "candidate_quality",
        "crop_quality",
    ],
    "consistency": [
        "c",
        "consistency",
        "consistency_score",
        "agreement",
        "agreement_score",
    ],
    "label": [
        "label",
        "target",
        "y",
        "y_true",
        "gt",
        "ground_truth",
        "anomaly",
        "is_anomaly",
    ],
    "category": [
        "category",
        "class",
        "class_name",
        "cls_name",
        "object",
        "object_name",
    ],
    "path": [
        "path",
        "image_path",
        "img_path",
        "filename",
        "file_name",
    ],
    "split": [
        "split",
        "mode",
        "subset",
        "partition",
    ],
}


def normalized(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def path_score(path: Path) -> int:
    text = str(path).lower()
    score = 0

    for keyword, weight in PATH_KEYWORDS.items():
        if keyword in text:
            score += weight

    return score


def detect_field_groups(fields: list[str]) -> dict[str, list[str]]:
    normalized_fields = {
        normalized(field): field
        for field in fields
    }

    detected: dict[str, list[str]] = {}

    for group, aliases in FIELD_GROUPS.items():
        matches = []

        for alias in aliases:
            alias_norm = normalized(alias)

            if alias_norm in normalized_fields:
                matches.append(normalized_fields[alias_norm])

        if matches:
            detected[group] = sorted(set(matches))

    return detected


def inspect_csv(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "csv",
        "columns": [],
        "row_count": None,
        "preview": [],
    }

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        result["columns"] = reader.fieldnames or []

        row_count = 0

        for row in reader:
            if row_count < 3:
                result["preview"].append(
                    {
                        key: value
                        for key, value in list(row.items())[:20]
                    }
                )

            row_count += 1

        result["row_count"] = row_count

    result["field_groups"] = detect_field_groups(
        result["columns"]
    )

    return result


def first_json_record(data: Any) -> Any:
    if isinstance(data, list):
        return data[0] if data else None

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list) and value:
                return value[0]

        return data

    return None


def inspect_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "json",
        "top_level_type": None,
        "top_level_keys": [],
        "record_fields": [],
        "preview": None,
    }

    if path.stat().st_size > 50 * 1024 * 1024:
        result["note"] = "Skipped content inspection: file exceeds 50 MiB."
        return result

    if path.suffix.lower() == ".jsonl":
        records = []

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            for index, line in enumerate(handle):
                if index >= 3:
                    break

                line = line.strip()
                if line:
                    records.append(json.loads(line))

        data: Any = records
    else:
        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

    result["top_level_type"] = type(data).__name__

    if isinstance(data, dict):
        result["top_level_keys"] = list(data.keys())[:50]

    record = first_json_record(data)

    if isinstance(record, dict):
        result["record_fields"] = list(record.keys())[:100]
        result["preview"] = {
            key: value
            for key, value in list(record.items())[:20]
        }

    fields = (
        result["record_fields"]
        or result["top_level_keys"]
    )

    result["field_groups"] = detect_field_groups(fields)

    return result


def inspect_numpy(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        return {
            "type": path.suffix.lower().lstrip("."),
            "note": f"NumPy unavailable: {exc}",
        }

    if path.suffix.lower() == ".npy":
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        return {
            "type": "npy",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    archive = np.load(
        path,
        allow_pickle=False,
    )

    arrays = {}

    for key in archive.files:
        array = archive[key]
        arrays[key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    return {
        "type": "npz",
        "arrays": arrays,
        "field_groups": detect_field_groups(
            list(archive.files)
        ),
    }


def inspect_parquet(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        return {
            "type": "parquet",
            "note": f"PyArrow unavailable: {exc}",
        }

    metadata = pq.read_metadata(path)
    fields = metadata.schema.names

    return {
        "type": "parquet",
        "columns": fields,
        "row_count": metadata.num_rows,
        "field_groups": detect_field_groups(fields),
    }


def inspect_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()

    base = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "path_score": path_score(path),
        "suffix": suffix,
    }

    try:
        if suffix == ".csv":
            details = inspect_csv(path)
        elif suffix in {".json", ".jsonl"}:
            details = inspect_json(path)
        elif suffix in {".npy", ".npz"}:
            details = inspect_numpy(path)
        elif suffix == ".parquet":
            details = inspect_parquet(path)
        else:
            details = {
                "type": suffix.lstrip("."),
                "note": (
                    "Listed but not deserialized to avoid unsafe or "
                    "expensive loading."
                ),
            }

        base.update(details)

    except Exception as exc:
        base["inspection_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    field_groups = base.get("field_groups", {})
    base["field_group_count"] = len(field_groups)

    # 排序优先级：包含 D/M/Q/C/标签字段的文件优先。
    base["relevance_score"] = (
        base["path_score"]
        + 4 * base["field_group_count"]
    )

    return base


def collect_candidates() -> list[Path]:
    candidates: set[Path] = set()

    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue

        for path in search_root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            relative = path.relative_to(ROOT)
            relative_text = str(relative).lower()

            # 聚焦 QCR、VisA、AD2、Stage16、Stage18 附近的文件。
            if not any(
                keyword in relative_text
                for keyword in [
                    "qcr",
                    "stage16",
                    "stage18",
                    "visa",
                    "ad2",
                ]
            ):
                continue

            candidates.add(path.resolve())

    return sorted(candidates)


def format_groups(groups: dict[str, list[str]]) -> str:
    if not groups:
        return "none"

    parts = []

    for group, fields in groups.items():
        parts.append(
            f"{group}={','.join(fields)}"
        )

    return "; ".join(parts)


def write_report(records: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 22-A1: Cached Score Artifact Audit",
        "",
        "## Purpose",
        "",
        "Locate existing per-image cached scores required for",
        "offline Selective QCR experiments without repeating",
        "detector or VLM inference.",
        "",
        "## Required information",
        "",
        "- detector evidence `D`",
        "- crop/VLM evidence `M`",
        "- candidate quality `Q`",
        "- optional consistency `C`",
        "- binary anomaly label",
        "- category and image identifier",
        "",
        f"## Candidate files ({len(records)})",
        "",
        "| Rank | File | Size MiB | Rows | Detected groups |",
        "|---:|---|---:|---:|---|",
    ]

    for rank, record in enumerate(records[:50], start=1):
        size_mib = record["size_bytes"] / 1024**2
        row_count = record.get("row_count", "")
        groups = format_groups(
            record.get("field_groups", {})
        )

        lines.append(
            f"| {rank} | `{record['path']}` | "
            f"{size_mib:.3f} | {row_count} | {groups} |"
        )

    lines += [
        "",
        "## Interpretation rule",
        "",
        "A file is immediately usable only if sample-level detector",
        "evidence, VLM evidence, quality, labels, and sample/category",
        "identifiers can be aligned without reconstructing values",
        "from aggregate metrics.",
        "",
    ]

    OUTPUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = collect_candidates()

    records = [
        inspect_file(path)
        for path in candidates
    ]

    records.sort(
        key=lambda record: (
            -record["relevance_score"],
            -record["path_score"],
            record["path"],
        )
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    write_report(records)

    print("===== STAGE 22-A1 CACHED SCORE AUDIT =====")
    print("candidate files:", len(records))
    print()

    for rank, record in enumerate(records[:30], start=1):
        print(
            f"[{rank:02d}] "
            f"score={record['relevance_score']:2d} "
            f"size={record['size_bytes'] / 1024**2:8.3f} MiB"
        )
        print("     path:", record["path"])
        print(
            "     groups:",
            format_groups(
                record.get("field_groups", {})
            ),
        )

        columns = record.get("columns")
        if columns:
            print(
                "     columns:",
                columns[:40],
            )

        arrays = record.get("arrays")
        if arrays:
            print(
                "     arrays:",
                arrays,
            )

        if record.get("inspection_error"):
            print(
                "     error:",
                record["inspection_error"],
            )

        print()

    print("[DONE]", OUTPUT_JSON)
    print("[DONE]", OUTPUT_REPORT)


if __name__ == "__main__":
    main()
