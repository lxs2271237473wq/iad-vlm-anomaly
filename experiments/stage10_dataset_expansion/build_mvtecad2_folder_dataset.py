from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(".").resolve()

IN_MANIFEST = ROOT / "results" / "stage10_dataset_expansion" / "stage10_b1_mvtecad2_manifest.csv"

ADAPTER_ROOT = ROOT / "datasets" / "MVTec_AD_2_anomalib" / "vial_folder"

OUT_DIR = ROOT / "results" / "stage10_dataset_expansion"
DOC_DIR = ROOT / "docs" / "stage10_dataset_expansion"

OUT_MAPPING = OUT_DIR / "stage10_b2_mvtecad2_folder_mapping.csv"
OUT_SUMMARY = OUT_DIR / "stage10_b2_mvtecad2_folder_summary.csv"
OUT_REPORT = DOC_DIR / "stage10_b2_mvtecad2_folder_report.md"


def safe_name(index: int, source_path: str) -> str:
    p = Path(source_path)
    stem = p.stem.replace(" ", "_")
    suffix = p.suffix.lower()
    return f"{index:06d}_{stem}{suffix}"


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def reset_adapter_root() -> None:
    if ADAPTER_ROOT.exists():
        shutil.rmtree(ADAPTER_ROOT)

    for rel in [
        "train/good",
        "test/good",
        "test/bad",
        "ground_truth/bad",
    ]:
        (ADAPTER_ROOT / rel).mkdir(parents=True, exist_ok=True)


def load_manifest() -> pd.DataFrame:
    if not IN_MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {IN_MANIFEST}")

    df = pd.read_csv(IN_MANIFEST)

    required = [
        "dataset",
        "category",
        "split",
        "image_path",
        "mask_path",
        "label_available",
        "is_anomaly",
        "anomaly_type",
        "evaluation_scope",
        "has_mask",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    df = df[df["category"] == "vial"].copy()

    if df.empty:
        raise RuntimeError("No rows for category=vial in manifest.")

    return df


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Normal reference set: train + validation.
    train_normal = df[
        (df["split"].isin(["train", "validation"]))
        & (df["is_anomaly"].astype(int) == 0)
    ].copy()
    train_normal["target_subset"] = "train"
    train_normal["target_label"] = "good"
    rows.append(train_normal)

    # Local test normal: test_public/good.
    test_good = df[
        (df["split"] == "test_public")
        & (df["is_anomaly"].astype(int) == 0)
    ].copy()
    test_good["target_subset"] = "test"
    test_good["target_label"] = "good"
    rows.append(test_good)

    # Local test anomaly: test_public/bad.
    test_bad = df[
        (df["split"] == "test_public")
        & (df["is_anomaly"].astype(int) == 1)
    ].copy()
    test_bad["target_subset"] = "test"
    test_bad["target_label"] = "bad"
    rows.append(test_bad)

    selected = pd.concat(rows, ignore_index=True)

    if selected.empty:
        raise RuntimeError("No selected rows for Folder adapter.")

    return selected


def build_folder_dataset(selected: pd.DataFrame) -> pd.DataFrame:
    mapping_rows: List[Dict[str, object]] = []

    for i, row in selected.reset_index(drop=True).iterrows():
        src_image = ROOT / str(row["image_path"])
        target_filename = safe_name(i, str(row["image_path"]))

        if row["target_subset"] == "train":
            dst_image = ADAPTER_ROOT / "train" / "good" / target_filename
            dst_mask = ""
        elif row["target_label"] == "good":
            dst_image = ADAPTER_ROOT / "test" / "good" / target_filename
            dst_mask = ""
        else:
            dst_image = ADAPTER_ROOT / "test" / "bad" / target_filename

            src_mask_text = "" if pd.isna(row["mask_path"]) else str(row["mask_path"])
            if src_mask_text:
                src_mask = ROOT / src_mask_text
                mask_suffix = src_mask.suffix.lower() if src_mask.suffix else ".png"
                dst_mask_path = ADAPTER_ROOT / "ground_truth" / "bad" / f"{Path(target_filename).stem}{mask_suffix}"
                copy_file(src_mask, dst_mask_path)
                dst_mask = str(dst_mask_path.relative_to(ROOT))
            else:
                dst_mask = ""

        copy_file(src_image, dst_image)

        mapping_rows.append({
            "dataset": row["dataset"],
            "category": row["category"],
            "source_split": row["split"],
            "source_image_path": row["image_path"],
            "source_mask_path": "" if pd.isna(row["mask_path"]) else row["mask_path"],
            "is_anomaly": int(row["is_anomaly"]),
            "anomaly_type": row["anomaly_type"],
            "target_subset": row["target_subset"],
            "target_label": row["target_label"],
            "target_image_path": str(dst_image.relative_to(ROOT)),
            "target_mask_path": dst_mask,
            "has_target_mask": bool(dst_mask),
        })

    return pd.DataFrame(mapping_rows)


def summarize(mapping: pd.DataFrame) -> pd.DataFrame:
    summary = (
        mapping
        .groupby(
            [
                "dataset",
                "category",
                "target_subset",
                "target_label",
                "is_anomaly",
                "anomaly_type",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            num_images=("target_image_path", "count"),
            num_masks=("has_target_mask", "sum"),
        )
        .sort_values(["target_subset", "target_label", "is_anomaly"])
    )

    return summary


def validate_files(mapping: pd.DataFrame) -> Dict[str, object]:
    missing_images = []
    missing_masks = []

    for _, row in mapping.iterrows():
        img_path = ROOT / str(row["target_image_path"])
        if not img_path.exists():
            missing_images.append(str(img_path.relative_to(ROOT)))

        mask_text = "" if pd.isna(row["target_mask_path"]) else str(row["target_mask_path"])
        if mask_text:
            mask_path = ROOT / mask_text
            if not mask_path.exists():
                missing_masks.append(mask_text)

    return {
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "missing_image_count": len(missing_images),
        "missing_mask_count": len(missing_masks),
    }


def optional_anomalib_validation() -> Dict[str, object]:
    result: Dict[str, object] = {
        "attempted": True,
        "success": False,
        "error": "",
        "train_batch_keys": "",
        "test_batch_keys": "",
    }

    try:
        from anomalib.data import Folder

        dm = Folder(
            name="mvtecad2_vial_folder",
            root=str(ADAPTER_ROOT),
            normal_dir="train/good",
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            mask_dir="ground_truth/bad",
            task="segmentation",
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=0,
        )

        dm.setup()

        train_batch = next(iter(dm.train_dataloader()))
        test_batch = next(iter(dm.test_dataloader()))

        if isinstance(train_batch, dict):
            result["train_batch_keys"] = ",".join(sorted(train_batch.keys()))
        else:
            result["train_batch_keys"] = str(type(train_batch))

        if isinstance(test_batch, dict):
            result["test_batch_keys"] = ",".join(sorted(test_batch.keys()))
        else:
            result["test_batch_keys"] = str(type(test_batch))

        result["success"] = True

    except Exception as e:
        result["error"] = repr(e)

    return result


def write_report(mapping: pd.DataFrame, summary: pd.DataFrame, validation: Dict[str, object], anomalib_check: Dict[str, object]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Stage 10-B2 MVTec AD 2 Folder Adapter Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage converts the MVTec AD 2 vial manifest into an Anomalib Folder-style dataset.")
    lines.append("It does not train PatchCore, run FastFlow, run VLM reasoning, or modify previous experiment results.")
    lines.append("")
    lines.append("## 2. Generated Folder Structure")
    lines.append("")
    lines.append(f"`{ADAPTER_ROOT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("```text")
    lines.append("train/good/")
    lines.append("test/good/")
    lines.append("test/bad/")
    lines.append("ground_truth/bad/")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Output Files")
    lines.append("")
    lines.append(f"- `{OUT_MAPPING.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 4. Summary")
    lines.append("")
    lines.append("| Dataset | Category | Subset | Label | is_anomaly | Anomaly type | Images | Masks |")
    lines.append("|---|---|---|---|---:|---|---:|---:|")

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['category']} | {r['target_subset']} | {r['target_label']} | "
            f"{int(r['is_anomaly'])} | {r['anomaly_type']} | {int(r['num_images'])} | {int(r['num_masks'])} |"
        )

    lines.append("")
    lines.append("## 5. File Validation")
    lines.append("")
    lines.append(f"- Missing target images: `{validation['missing_image_count']}`")
    lines.append(f"- Missing target masks: `{validation['missing_mask_count']}`")
    lines.append("")
    lines.append("## 6. Optional Anomalib Folder Validation")
    lines.append("")
    lines.append(f"- Attempted: `{anomalib_check['attempted']}`")
    lines.append(f"- Success: `{anomalib_check['success']}`")
    lines.append(f"- Train batch keys: `{anomalib_check['train_batch_keys']}`")
    lines.append(f"- Test batch keys: `{anomalib_check['test_batch_keys']}`")
    if anomalib_check["error"]:
        lines.append(f"- Error: `{anomalib_check['error']}`")
    lines.append("")
    lines.append("## 7. Next Step")
    lines.append("")
    lines.append("If `Success=True`, Stage 10-C should run the first PatchCore pilot on this Folder dataset.")
    lines.append("If Anomalib validation fails, Stage 10-C should first patch the datamodule arguments instead of training.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    reset_adapter_root()

    manifest = load_manifest()
    selected = select_rows(manifest)
    mapping = build_folder_dataset(selected)
    summary = summarize(mapping)
    validation = validate_files(mapping)
    anomalib_check = optional_anomalib_validation()

    mapping.to_csv(OUT_MAPPING, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(mapping, summary, validation, anomalib_check)

    print("[DONE]", OUT_MAPPING)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print("")
    print(summary.to_string(index=False))
    print("")
    print("missing_image_count:", validation["missing_image_count"])
    print("missing_mask_count:", validation["missing_mask_count"])
    print("anomalib_folder_success:", anomalib_check["success"])
    if anomalib_check["error"]:
        print("anomalib_folder_error:", anomalib_check["error"])


if __name__ == "__main__":
    main()
