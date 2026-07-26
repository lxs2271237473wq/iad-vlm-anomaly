from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

CATEGORIES = [
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "walnuts",
]

DATASET_ROOT = (
    ROOT
    / "datasets"
    / "MVTec_AD_2_anomalib_all"
)

SCAN_ROOTS = [
    ROOT / "results/stage11_mvtecad2_multicategory",
    ROOT / "results/stage18_ad2_qcr_ablation",
    ROOT / "results/stage20_anomalyclip_baseline",
    ROOT / "results/stage22_selective_qcr",
]

REQUIRED_PROTOCOL = {
    "w_max": 0.35,
    "q_quantile": 0.25,
    "tau_delta": 0.75,
}

PROTOCOL_PATH = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_a2_srb_qcr_frozen_protocol.json"
)


GLOBAL_CONFIG_PATH = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_b1_visa_patchcore_global_config.json"
)

TRANSFER_METADATA = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_b2b_ad2_transfer_metadata.json"
)

B2B_SCRIPT = (
    ROOT
    / "experiments/stage22_selective_qcr"
    / "run_stage22_b2b_ad2_frozen_transfer.py"
)

OUT_JSON = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "stage23_a1_ad2_input_inventory.json"
)

OUT_REPORT = (
    ROOT
    / "docs/stage23_ad2_mirror"
    / "stage23_a1_ad2_input_inventory.txt"
)


def count_files(path: Path) -> int:
    if not path.exists():
        return -1

    return sum(
        1
        for item in path.rglob("*")
        if item.is_file()
    )


def row_count(path: Path) -> int | None:
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            return max(
                sum(1 for _ in handle) - 1,
                0,
            )
    except Exception:
        return None


def inspect_csv(path: Path) -> dict:
    record = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "row_count": None,
        "columns": [],
        "observed_categories": [],
        "roles": [],
        "relevance_score": 0,
        "error": None,
    }

    try:
        header = pd.read_csv(
            path,
            nrows=0,
        )

        columns = [
            str(column)
            for column in header.columns
        ]

        record["columns"] = columns

        lower = {
            column.lower()
            for column in columns
        }

        category_column = next(
            (
                column
                for column in columns
                if column.lower()
                in {
                    "category",
                    "object",
                    "objects",
                    "class_name",
                }
            ),
            None,
        )

        if category_column is not None:
            sample = pd.read_csv(
                path,
                usecols=[category_column],
                nrows=20000,
            )

            observed = sorted(
                set(
                    sample[
                        category_column
                    ]
                    .dropna()
                    .astype(str)
                )
                & set(CATEGORIES)
            )

            record[
                "observed_categories"
            ] = observed

            record[
                "relevance_score"
            ] += 10 * len(observed)

        path_lower = str(path).lower()

        if (
            "ad2" in path_lower
            or "mvtecad2" in path_lower
        ):
            record[
                "relevance_score"
            ] += 8

        if {
            "image_path",
            "category",
        }.issubset(lower):
            record[
                "relevance_score"
            ] += 5

        detector_tokens = {
            "patchcore_score",
            "pred_score",
            "detector_score",
            "score_d0",
            "d",
        }

        vlm_tokens = {
            "full_image_score",
            "tight_vlm_margin",
            "context_vlm_margin",
            "vlm_anomaly_score",
            "score_m0",
            "m",
        }

        quality_tokens = {
            "candidate_score_max",
            "candidate_score_mean",
            "candidate_quality",
            "candidate_quality_norm",
            "q",
        }

        final_tokens = {
            "score_v3",
            "score_v4",
            "score_v6",
            "score_s1",
            "srb_pre_gate",
            "srb_weight",
        }

        candidate_tokens = {
            "component_rank",
            "candidate_available",
            "x1",
            "y1",
            "x2",
            "y2",
            "map_x1",
            "map_y1",
            "map_x2",
            "map_y2",
        }

        if lower & detector_tokens:
            record["roles"].append(
                "detector"
            )

        if lower & vlm_tokens:
            record["roles"].append(
                "vlm"
            )

        if lower & quality_tokens:
            record["roles"].append(
                "quality"
            )

        if lower & final_tokens:
            record["roles"].append(
                "frozen_scores"
            )

        if len(
            lower & candidate_tokens
        ) >= 4:
            record["roles"].append(
                "candidate_regions"
            )

        record["relevance_score"] += (
            4 * len(record["roles"])
        )

        if (
            record["relevance_score"] > 0
            and path.stat().st_size
            < 100 * 1024 * 1024
        ):
            record[
                "row_count"
            ] = row_count(path)

    except Exception as error:
        record["error"] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    return record


def load_json(path: Path):
    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def find_config(value):
    """
    Recursively locate the frozen SRB configuration.

    Stage 22 protocol files may store configurations inside
    dictionaries, candidate lists, selected records, or nested
    protocol blocks.
    """

    if isinstance(value, dict):
        aliases = {
            "w_max": [
                "w_max",
                "selected_w_max",
            ],
            "q_quantile": [
                "q_quantile",
                "selected_q_quantile",
            ],
            "tau_delta": [
                "tau_delta",
                "selected_tau_delta",
            ],
        }

        resolved = {}

        for canonical, names in aliases.items():
            for name in names:
                if name in value:
                    try:
                        resolved[canonical] = float(
                            value[name]
                        )
                        break
                    except (TypeError, ValueError):
                        pass

        if set(resolved) == set(REQUIRED_PROTOCOL):
            return resolved

        # Prefer semantically likely protocol blocks first.
        preferred_keys = [
            "configuration",
            "global_configuration",
            "selected",
            "selected_configuration",
            "frozen_configuration",
            "protocol",
            "all_candidates",
        ]

        visited = set()

        for key in preferred_keys:
            if key in value:
                visited.add(key)

                result = find_config(
                    value[key]
                )

                if result is not None:
                    return result

        for key, child in value.items():
            if key in visited:
                continue

            result = find_config(child)

            if result is not None:
                return result

    elif isinstance(value, list):
        for child in value:
            result = find_config(child)

            if result is not None:
                return result

    return None


def main():
    dataset_rows = []

    for category in CATEGORIES:
        root = (
            DATASET_ROOT
            / f"{category}_folder"
        )

        dataset_rows.append(
            {
                "category": category,
                "root": str(
                    root.relative_to(ROOT)
                ),
                "root_exists": root.exists(),
                "train_good": count_files(
                    root / "train/good"
                ),
                "test_good": count_files(
                    root / "test/good"
                ),
                "test_bad": count_files(
                    root / "test/bad"
                ),
                "ground_truth_bad": (
                    count_files(
                        root
                        / "ground_truth/bad"
                    )
                ),
            }
        )

    protocol = load_json(
        PROTOCOL_PATH
    )

    global_config_payload = load_json(
        GLOBAL_CONFIG_PATH
    )

    metadata = load_json(
        TRANSFER_METADATA
    )

    protocol_identity_ok = (
        isinstance(protocol, dict)
        and protocol.get("protocol_id")
        == "stage22_a2_srb_qcr_v1"
        and protocol.get("status")
        == "frozen_before_stage22_b_results"
        and protocol.get("short_name")
        == "SRB-QCR"
    )

    if global_config_payload is None:
        protocol_config = None
    else:
        selected = global_config_payload[
            "global_configuration"
        ]["selected"]

        protocol_config = {
            "w_max": float(
                selected["w_max"]
            ),
            "q_quantile": float(
                selected["q_quantile"]
            ),
            "tau_delta": float(
                selected["tau_delta"]
            ),
        }

    metadata_config = (
        find_config(metadata)
        if metadata is not None
        else None
    )

    csv_rows = []

    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue

        for path in scan_root.rglob(
            "*.csv"
        ):
            record = inspect_csv(path)

            if (
                record[
                    "relevance_score"
                ] > 0
            ):
                csv_rows.append(record)

    csv_rows = sorted(
        csv_rows,
        key=lambda item: (
            -item["relevance_score"],
            item["path"],
        ),
    )

    payload = {
        "protocol_id": (
            "stage23_a1_ad2_"
            "input_inventory_v1"
        ),
        "categories": CATEGORIES,
        "required_frozen_configuration": (
            REQUIRED_PROTOCOL
        ),
        "protocol_path": str(
            PROTOCOL_PATH.relative_to(ROOT)
        ),
        "protocol_exists": (
            PROTOCOL_PATH.exists()
        ),
        "protocol_identity_ok": (
            protocol_identity_ok
        ),
        "global_config_path": str(
            GLOBAL_CONFIG_PATH.relative_to(ROOT)
        ),
        "global_config_exists": (
            GLOBAL_CONFIG_PATH.exists()
        ),
        "protocol_configuration": (
            protocol_config
        ),
        "transfer_metadata_path": str(
            TRANSFER_METADATA.relative_to(
                ROOT
            )
        ),
        "transfer_metadata_exists": (
            TRANSFER_METADATA.exists()
        ),
        "transfer_metadata_configuration": (
            metadata_config
        ),
        "b2b_script": str(
            B2B_SCRIPT.relative_to(ROOT)
        ),
        "b2b_script_exists": (
            B2B_SCRIPT.exists()
        ),
        "dataset_inventory": dataset_rows,
        "candidate_csv_inventory": (
            csv_rows[:50]
        ),
    }

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_REPORT.parent.mkdir(
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
        "===== STAGE 23-A1 AD2 INPUT INVENTORY =====",
        "",
        "===== FROZEN PROTOCOL =====",
        f"protocol exists: {PROTOCOL_PATH.exists()}",
        f"protocol identity valid: {protocol_identity_ok}",
        f"global config exists: {GLOBAL_CONFIG_PATH.exists()}",
        f"protocol config: {protocol_config}",
        f"metadata exists: {TRANSFER_METADATA.exists()}",
        f"metadata config: {metadata_config}",
        f"B2b script exists: {B2B_SCRIPT.exists()}",
        f"required config: {REQUIRED_PROTOCOL}",
        "",
        "===== AD2 DATASET =====",
    ]

    for row in dataset_rows:
        lines.append(
            (
                f"{row['category']}: "
                f"root={row['root_exists']}, "
                f"train/good={row['train_good']}, "
                f"test/good={row['test_good']}, "
                f"test/bad={row['test_bad']}, "
                "ground_truth/bad="
                f"{row['ground_truth_bad']}"
            )
        )

    lines += [
        "",
        "===== TOP RELEVANT CSV FILES =====",
    ]

    for index, row in enumerate(
        csv_rows[:30],
        start=1,
    ):
        lines += [
            f"[{index}] {row['path']}",
            (
                f"  score={row['relevance_score']} "
                f"rows={row['row_count']} "
                f"size={row['size_bytes']}"
            ),
            (
                "  categories="
                f"{row['observed_categories']}"
            ),
            f"  roles={row['roles']}",
            (
                "  columns="
                + ", ".join(
                    row["columns"]
                )
            ),
        ]

        if row["error"]:
            lines.append(
                f"  error={row['error']}"
            )

    def config_matches(actual):
        if actual is None:
            return False

        return all(
            key in actual
            and abs(
                float(actual[key])
                - float(expected)
            ) <= 1e-12
            for key, expected
            in REQUIRED_PROTOCOL.items()
        )

    protocol_ok = (
        protocol_identity_ok
        and GLOBAL_CONFIG_PATH.exists()
        and config_matches(
            protocol_config
        )
    )

    metadata_ok = config_matches(
        metadata_config
    )

    dataset_ok = all(
        row["root_exists"]
        and row["test_good"] >= 0
        and row["test_bad"] >= 0
        for row in dataset_rows
    )

    lines += [
        "",
        "===== DECISION =====",
        f"protocol_config_locked: {protocol_ok}",
        f"metadata_config_locked: {metadata_ok}",
        f"all_dataset_roots_ready: {dataset_ok}",
        (
            "next_stage_ready: "
            f"{protocol_ok and metadata_ok and dataset_ok}"
        ),
        "",
        f"[DONE] {OUT_JSON}",
        f"[DONE] {OUT_REPORT}",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
