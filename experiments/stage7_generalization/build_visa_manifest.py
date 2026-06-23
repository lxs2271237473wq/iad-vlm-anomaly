import argparse
from pathlib import Path

import pandas as pd


def normalize_path_text(x):
    if pd.isna(x):
        return ""
    return str(x).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visa_root",
        type=str,
        default="datasets/VisA",
        help="Root directory of the extracted VisA dataset.",
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default="datasets/VisA/split_csv/1cls.csv",
        help="VisA split CSV to parse. Default uses 1-class setting.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/stage7_generalization/visa_manifest",
        help="Output directory for VisA manifest files.",
    )
    args = parser.parse_args()

    visa_root = Path(args.visa_root).resolve()
    split_csv = Path(args.split_csv).resolve()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not visa_root.exists():
        raise FileNotFoundError(f"VisA root not found: {visa_root}")

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)

    required_cols = ["object", "split", "label", "image", "mask"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in {split_csv}: {missing_cols}")

    rows = []
    missing_rows = []

    for idx, row in df.iterrows():
        category = str(row["object"])
        split = str(row["split"])
        label = str(row["label"])

        image_rel = normalize_path_text(row["image"])
        mask_rel = normalize_path_text(row["mask"])

        image_path = visa_root / image_rel
        mask_path = visa_root / mask_rel if mask_rel else None

        is_anomaly = label.lower() == "anomaly"
        is_normal = label.lower() == "normal"

        if not is_anomaly and not is_normal:
            raise ValueError(f"Unknown label at row {idx}: {label}")

        image_exists = image_path.exists()
        mask_exists = mask_path.exists() if mask_path is not None else False

        if not image_exists:
            missing_rows.append(
                {
                    "row_index": idx,
                    "category": category,
                    "split": split,
                    "label": label,
                    "missing_type": "image",
                    "path": str(image_path),
                }
            )

        if is_anomaly and not mask_exists:
            missing_rows.append(
                {
                    "row_index": idx,
                    "category": category,
                    "split": split,
                    "label": label,
                    "missing_type": "mask",
                    "path": "" if mask_path is None else str(mask_path),
                }
            )

        rows.append(
            {
                "dataset": "VisA",
                "category": category,
                "split": split,
                "label": label,
                "is_anomaly": int(is_anomaly),
                "image_rel_path": image_rel,
                "image_path": str(image_path),
                "mask_rel_path": mask_rel,
                "mask_path": "" if mask_path is None else str(mask_path),
                "image_exists": int(image_exists),
                "mask_exists": int(mask_exists),
                "has_pixel_mask": int(is_anomaly and mask_exists),
                "source_split_csv": str(split_csv),
            }
        )

    manifest = pd.DataFrame(rows)

    manifest_csv = output_root / "visa_image_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)

    category_summary = (
        manifest.groupby("category")
        .agg(
            num_images=("image_path", "count"),
            num_train=("split", lambda x: int((x == "train").sum())),
            num_test=("split", lambda x: int((x == "test").sum())),
            num_normal=("label", lambda x: int((x == "normal").sum())),
            num_anomaly=("label", lambda x: int((x == "anomaly").sum())),
            num_masks=("has_pixel_mask", "sum"),
            num_missing_images=("image_exists", lambda x: int((x == 0).sum())),
            num_missing_masks=("mask_exists", lambda x: int(((manifest.loc[x.index, "is_anomaly"] == 1) & (x == 0)).sum())),
        )
        .reset_index()
    )

    category_summary_csv = output_root / "visa_category_summary.csv"
    category_summary.to_csv(category_summary_csv, index=False)

    split_summary = (
        manifest.groupby(["category", "split", "label"])
        .agg(
            num_images=("image_path", "count"),
            num_masks=("has_pixel_mask", "sum"),
            num_missing_images=("image_exists", lambda x: int((x == 0).sum())),
        )
        .reset_index()
        .sort_values(["category", "split", "label"])
    )

    split_summary_csv = output_root / "visa_category_split_summary.csv"
    split_summary.to_csv(split_summary_csv, index=False)

    missing_df = pd.DataFrame(missing_rows)
    missing_csv = output_root / "visa_missing_files.csv"
    missing_df.to_csv(missing_csv, index=False)

    overall = {
        "dataset": "VisA",
        "source_split_csv": str(split_csv),
        "num_categories": manifest["category"].nunique(),
        "num_images": len(manifest),
        "num_train": int((manifest["split"] == "train").sum()),
        "num_test": int((manifest["split"] == "test").sum()),
        "num_normal": int((manifest["label"] == "normal").sum()),
        "num_anomaly": int((manifest["label"] == "anomaly").sum()),
        "num_pixel_masks": int(manifest["has_pixel_mask"].sum()),
        "num_missing_records": len(missing_df),
    }

    overall_csv = output_root / "visa_overall_summary.csv"
    pd.DataFrame([overall]).to_csv(overall_csv, index=False)

    print("\n========== VisA Overall Summary ==========")
    for k, v in overall.items():
        print(f"{k}: {v}")

    print("\n========== VisA Category Summary ==========")
    print(category_summary.to_string(index=False))

    print(f"\n[DONE] Manifest saved to: {manifest_csv}")
    print(f"[DONE] Category summary saved to: {category_summary_csv}")
    print(f"[DONE] Split summary saved to: {split_summary_csv}")
    print(f"[DONE] Missing file report saved to: {missing_csv}")
    print(f"[DONE] Overall summary saved to: {overall_csv}")


if __name__ == "__main__":
    main()
