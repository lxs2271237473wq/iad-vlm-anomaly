from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()
TARGET = ROOT / "experiments/stage23_ad2_mirror/run_stage23_c2_ad2_actual_selective_runtime.py"


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(encoding="utf-8")

    if "ASSET_MAPPING =" not in text:
        marker = '''CANDIDATE_SCORES = (
    ROOT
    / "results/stage11_mvtecad2_multicategory"
    / "stage11_d_vlm_candidate_scores.csv"
)
'''
        insertion = marker + '''

ASSET_MAPPING = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "ad2_actual_selective_runtime"
    / "stage23_c1d_ad2_runtime_asset_mapping.csv"
)
'''
        if marker not in text:
            raise RuntimeError("Could not locate CANDIDATE_SCORES block.")
        text = text.replace(marker, insertion, 1)

    pattern = re.compile(
        r"def load_runtime_crops\(.*?(?=\n\ndef load_image_reference_scores\()",
        flags=re.DOTALL,
    )

    replacement = r'''def load_runtime_crops(
    predictions: pd.DataFrame,
    categories: list[str],
    top_k: int,
) -> dict[str, list[dict]]:
    # Load the filesystem-validated Stage 23-C1d runtime mapping.
    if not ASSET_MAPPING.exists():
        raise FileNotFoundError(ASSET_MAPPING)

    mapping = pd.read_csv(ASSET_MAPPING)

    required = {
        "category",
        "path_key",
        "gate_on",
        "runtime_crop_paths",
        "num_eval_images",
        "all_assets_ready",
        "asset_mode",
    }

    missing = sorted(required - set(mapping.columns))
    if missing:
        raise RuntimeError(
            f"C1d runtime mapping is missing columns: {missing}"
        )

    mapping = mapping[
        mapping["category"].astype(str).isin(categories)
    ].copy()

    mapping["category"] = mapping["category"].astype(str)
    mapping["path_key"] = (
        mapping["path_key"].astype(str).map(canonical_path)
    )
    mapping["gate_on_bool"] = as_bool(mapping["gate_on"])
    mapping["assets_ready_bool"] = as_bool(mapping["all_assets_ready"])
    mapping["num_eval_images"] = pd.to_numeric(
        mapping["num_eval_images"],
        errors="raise",
    ).astype(int)

    if mapping["path_key"].duplicated().any():
        raise RuntimeError("Duplicate path keys in C1d mapping.")

    if not mapping["assets_ready_bool"].all():
        raise RuntimeError("C1d mapping contains unready assets.")

    expected = predictions[
        ["category", "path_key", "srb_pre_gate"]
    ].copy()
    expected["gate_expected"] = expected["srb_pre_gate"] > 0

    aligned = expected.merge(
        mapping[
            [
                "category",
                "path_key",
                "gate_on_bool",
                "runtime_crop_paths",
                "num_eval_images",
                "asset_mode",
            ]
        ],
        on=["category", "path_key"],
        how="left",
        validate="one_to_one",
    )

    if aligned["runtime_crop_paths"].isna().any():
        raise RuntimeError("Predictions are missing from C1d mapping.")

    if not (
        aligned["gate_expected"] == aligned["gate_on_bool"]
    ).all():
        raise RuntimeError("Gate mismatch between B2b and C1d mapping.")

    import json

    crops_by_path: dict[str, list[dict]] = {}

    for _, row in aligned.iterrows():
        raw_paths = json.loads(row["runtime_crop_paths"])

        if not isinstance(raw_paths, list):
            raise RuntimeError(
                f"runtime_crop_paths is not a list for {row['path_key']}"
            )

        if len(raw_paths) != int(row["num_eval_images"]):
            raise RuntimeError(
                f"Crop count mismatch for {row['path_key']}"
            )

        if len(raw_paths) < 1:
            raise RuntimeError(
                f"No runtime crops for {row['path_key']}"
            )

        if len(raw_paths) > top_k:
            raise RuntimeError(
                f"Runtime mapping exceeds top_k for {row['path_key']}"
            )

        records = []

        for rank, raw_path in enumerate(raw_paths, start=1):
            crop_path = resolve_file(raw_path)
            records.append(
                {
                    "candidate_rank": rank,
                    "crop_path": str(crop_path),
                    "cached_margin": float("nan"),
                }
            )

        crops_by_path[row["path_key"]] = records

    if len(crops_by_path) != len(predictions):
        raise RuntimeError(
            f"Runtime mapping row mismatch: "
            f"{len(crops_by_path)} vs {len(predictions)}"
        )

    return crops_by_path
'''

    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not replace load_runtime_crops().")

    backup = TARGET.with_suffix(".py.before_c1d_mapping")
    backup.write_text(text, encoding="utf-8", newline="\n")
    TARGET.write_text(new_text, encoding="utf-8", newline="\n")

    print("[BACKUP]", backup)
    print("[PATCHED]", TARGET)


if __name__ == "__main__":
    main()
