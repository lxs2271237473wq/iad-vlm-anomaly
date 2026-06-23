import argparse
import os
from pathlib import Path

import pandas as pd


def safe_link(src: Path, dst: Path, overwrite: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return "exists"

    os.link(src, dst)
    return "linked"


def normalize_suffix(path_text: str):
    return Path(path_text).suffix.lower()


def build_view_name(row):
    src = Path(row["image_path"])
    stem = src.stem
    suffix = src.suffix

    category = row["category"]
    label = row["label"]
    split = row["split"]

    return f"{category}_{split}_{label}_{stem}{suffix}"


def build_mask_name(row):
    mask_path = str(row["mask_path"])
    if not mask_path:
        return ""

    src = Path(mask_path)
    stem = Path(row["image_path"]).stem
    suffix = src.suffix

    category = row["category"]
    return f"{category}_test_anomaly_{stem}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default="results/stage7_generalization/visa_manifest/visa_image_manifest.csv",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="datasets/VisA_anomalib_1cls",
    )
    parser.add_argument(
        "--summary_root",
        type=str,
        default="results/stage7_generalization/visa_anomalib_view",
    )
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv)
    output_root = Path(args.output_root)
    summary_root = Path(args.summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_csv}")

    df = pd.read_csv(manifest_csv)

    if args.categories:
        df = df[df["category"].isin(args.categories)].copy()

    required = [
        "category",
        "split",
        "label",
        "image_path",
        "mask_path",
        "image_exists",
        "mask_exists",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in manifest: {missing}")

    rows = []
    errors = []

    for _, row in df.iterrows():
        category = row["category"]
        split = row["split"]
        label = row["label"]

        image_src = Path(row["image_path"])
        if not image_src.exists():
            errors.append(
                {
                    "category": category,
                    "split": split,
                    "label": label,
                    "type": "missing_image",
                    "path": str(image_src),
                }
            )
            continue

        if split == "train" and label == "normal":
            image_dst = output_root / category / "train" / "good" / build_view_name(row)
        elif split == "test" and label == "normal":
            image_dst = output_root / category / "test" / "good" / build_view_name(row)
        elif split == "test" and label == "anomaly":
            image_dst = output_root / category / "test" / "anomaly" / build_view_name(row)
        else:
            errors.append(
                {
                    "category": category,
                    "split": split,
                    "label": label,
                    "type": "unsupported_split_label",
                    "path": str(image_src),
                }
            )
            continue

        image_status = safe_link(image_src, image_dst, overwrite=args.overwrite)

        mask_dst = ""
        mask_status = ""

        if split == "test" and label == "anomaly":
            mask_src = Path(str(row["mask_path"]))
            if not mask_src.exists():
                errors.append(
                    {
                        "category": category,
                        "split": split,
                        "label": label,
                        "type": "missing_mask",
                        "path": str(mask_src),
                    }
                )
                mask_status = "missing"
            else:
                mask_dst_path = output_root / category / "ground_truth" / "anomaly" / build_mask_name(row)
                mask_status = safe_link(mask_src, mask_dst_path, overwrite=args.overwrite)
                mask_dst = str(mask_dst_path)

        rows.append(
            {
                "dataset": "VisA",
                "category": category,
                "split": split,
                "label": label,
                "source_image_path": str(image_src),
                "view_image_path": str(image_dst),
                "source_mask_path": str(row["mask_path"]) if str(row["mask_path"]) != "nan" else "",
                "view_mask_path": mask_dst,
                "image_status": image_status,
                "mask_status": mask_status,
            }
        )

    view_df = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors)

    view_csv = summary_root / "visa_anomalib_view_manifest.csv"
    error_csv = summary_root / "visa_anomalib_view_errors.csv"

    view_df.to_csv(view_csv, index=False)
    error_df.to_csv(error_csv, index=False)

    summary_rows = []

    for category, group in view_df.groupby("category"):
        category_root = output_root / category

        train_good = len(list((category_root / "train" / "good").glob("*"))) if (category_root / "train" / "good").exists() else 0
        test_good = len(list((category_root / "test" / "good").glob("*"))) if (category_root / "test" / "good").exists() else 0
        test_anomaly = len(list((category_root / "test" / "anomaly").glob("*"))) if (category_root / "test" / "anomaly").exists() else 0
        masks = len(list((category_root / "ground_truth" / "anomaly").glob("*"))) if (category_root / "ground_truth" / "anomaly").exists() else 0

        summary_rows.append(
            {
                "category": category,
                "train_good": train_good,
                "test_good": test_good,
                "test_anomaly": test_anomaly,
                "masks": masks,
                "total_view_images": train_good + test_good + test_anomaly,
                "category_root": str(category_root),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("category")
    summary_csv = summary_root / "visa_anomalib_view_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n========== VisA Anomalib View Summary ==========")
    print(summary_df.to_string(index=False))

    print("\n========== Errors ==========")
    print(error_df.to_string(index=False) if len(error_df) else "No errors.")

    print(f"\n[DONE] View manifest saved to: {view_csv}")
    print(f"[DONE] View summary saved to: {summary_csv}")
    print(f"[DONE] Error report saved to: {error_csv}")
    print(f"[DONE] Anomalib-compatible view root: {output_root}")


if __name__ == "__main__":
    main()
