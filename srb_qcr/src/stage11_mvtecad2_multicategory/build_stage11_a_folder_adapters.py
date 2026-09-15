from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(".").resolve()

IN_MANIFEST = ROOT / "results" / "stage10_dataset_expansion" / "stage10_b1_mvtecad2_manifest.csv"

ADAPTER_ROOT = ROOT / "datasets" / "MVTec_AD_2_anomalib_all"

OUT_DIR = ROOT / "results" / "stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs" / "stage11_mvtecad2_multicategory"

OUT_MAPPING = OUT_DIR / "stage11_a_folder_adapter_mapping.csv"
OUT_SUMMARY = OUT_DIR / "stage11_a_folder_adapter_summary.csv"
OUT_VALIDATION = OUT_DIR / "stage11_a_folder_adapter_validation.csv"
OUT_REPORT = DOC_DIR / "stage11_a_folder_adapter_report.md"


def filter_kwargs(cls_or_fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls_or_fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


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
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)


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

    categories = sorted(df["category"].unique().tolist())
    if len(categories) != 8:
        raise RuntimeError(f"Expected 8 AD2 categories, got {len(categories)}: {categories}")

    return df


def select_rows_for_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    cdf = df[df["category"] == category].copy()

    rows = []

    train_normal = cdf[
        (cdf["split"].isin(["train", "validation"]))
        & (cdf["is_anomaly"].astype(int) == 0)
    ].copy()
    train_normal["target_subset"] = "train"
    train_normal["target_label"] = "good"
    rows.append(train_normal)

    test_good = cdf[
        (cdf["split"] == "test_public")
        & (cdf["is_anomaly"].astype(int) == 0)
    ].copy()
    test_good["target_subset"] = "test"
    test_good["target_label"] = "good"
    rows.append(test_good)

    test_bad = cdf[
        (cdf["split"] == "test_public")
        & (cdf["is_anomaly"].astype(int) == 1)
    ].copy()
    test_bad["target_subset"] = "test"
    test_bad["target_label"] = "bad"
    rows.append(test_bad)

    selected = pd.concat(rows, ignore_index=True)

    if selected.empty:
        raise RuntimeError(f"No selected rows for category={category}")

    return selected


def build_category_folder(selected: pd.DataFrame, category: str) -> pd.DataFrame:
    category_root = ADAPTER_ROOT / f"{category}_folder"

    for rel in [
        "train/good",
        "test/good",
        "test/bad",
        "ground_truth/bad",
    ]:
        (category_root / rel).mkdir(parents=True, exist_ok=True)

    mapping_rows: List[Dict[str, object]] = []

    for i, row in selected.reset_index(drop=True).iterrows():
        src_image = ROOT / str(row["image_path"])
        target_filename = safe_name(i, str(row["image_path"]))

        if row["target_subset"] == "train":
            dst_image = category_root / "train" / "good" / target_filename
            dst_mask = ""
        elif row["target_label"] == "good":
            dst_image = category_root / "test" / "good" / target_filename
            dst_mask = ""
        else:
            dst_image = category_root / "test" / "bad" / target_filename

            src_mask_text = "" if pd.isna(row["mask_path"]) else str(row["mask_path"])
            if src_mask_text:
                src_mask = ROOT / src_mask_text
                mask_suffix = src_mask.suffix.lower() if src_mask.suffix else ".png"
                dst_mask_path = category_root / "ground_truth" / "bad" / f"{Path(target_filename).stem}{mask_suffix}"
                copy_file(src_mask, dst_mask_path)
                dst_mask = str(dst_mask_path.relative_to(ROOT))
            else:
                dst_mask = ""

        copy_file(src_image, dst_image)

        mapping_rows.append({
            "dataset": row["dataset"],
            "category": category,
            "folder_root": str(category_root.relative_to(ROOT)),
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


def build_summary(mapping: pd.DataFrame) -> pd.DataFrame:
    return (
        mapping
        .groupby(
            [
                "dataset",
                "category",
                "folder_root",
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
        .sort_values(["category", "target_subset", "target_label", "is_anomaly"])
    )


def validate_files(mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category, part in mapping.groupby("category"):
        missing_images = 0
        missing_masks = 0

        for _, row in part.iterrows():
            if not (ROOT / str(row["target_image_path"])).exists():
                missing_images += 1

            mask_text = "" if pd.isna(row["target_mask_path"]) else str(row["target_mask_path"])
            if mask_text and not (ROOT / mask_text).exists():
                missing_masks += 1

        rows.append({
            "category": category,
            "folder_root": part["folder_root"].iloc[0],
            "target_rows": int(len(part)),
            "missing_images": int(missing_images),
            "missing_masks": int(missing_masks),
        })

    return pd.DataFrame(rows)


def optional_anomalib_validation(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    try:
        from anomalib.data import Folder
    except Exception as e:
        for category in sorted(summary["category"].unique()):
            folder_root = summary[summary["category"] == category]["folder_root"].iloc[0]
            rows.append({
                "category": category,
                "folder_root": folder_root,
                "anomalib_success": False,
                "train_batch_type": "",
                "test_batch_type": "",
                "error": f"Cannot import anomalib.data.Folder: {repr(e)}",
            })
        return pd.DataFrame(rows)

    for category in sorted(summary["category"].unique()):
        folder_root = summary[summary["category"] == category]["folder_root"].iloc[0]
        root_path = ROOT / folder_root

        result = {
            "category": category,
            "folder_root": folder_root,
            "anomalib_success": False,
            "train_batch_type": "",
            "test_batch_type": "",
            "error": "",
        }

        try:
            kwargs = {
                "name": f"mvtecad2_{category}_folder",
                "root": str(root_path),
                "normal_dir": "train/good",
                "abnormal_dir": "test/bad",
                "normal_test_dir": "test/good",
                "mask_dir": "ground_truth/bad",
                "train_batch_size": 1,
                "eval_batch_size": 1,
                "num_workers": 0,
            }

            dm = Folder(**filter_kwargs(Folder, kwargs))
            dm.setup()

            train_batch = next(iter(dm.train_dataloader()))
            test_batch = next(iter(dm.test_dataloader()))

            result["train_batch_type"] = type(train_batch).__name__
            result["test_batch_type"] = type(test_batch).__name__
            result["anomalib_success"] = True

        except Exception as e:
            result["error"] = repr(e)

        rows.append(result)

    return pd.DataFrame(rows)


def write_report(mapping: pd.DataFrame, summary: pd.DataFrame, validation: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    categories = sorted(mapping["category"].unique().tolist())
    success_count = int(validation["anomalib_success"].sum()) if "anomalib_success" in validation.columns else 0

    lines = []
    lines.append("# Stage 11-A MVTec AD 2 Multi-category Folder Adapter Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage builds Anomalib Folder-style datasets for all MVTec AD 2 categories.")
    lines.append("It does not train PatchCore, run VLM reasoning, generate crops, or modify original datasets.")
    lines.append("")
    lines.append("## 2. Categories")
    lines.append("")
    for c in categories:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## 3. Output Files")
    lines.append("")
    lines.append(f"- Mapping: `{OUT_MAPPING.relative_to(ROOT)}`")
    lines.append(f"- Summary: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Validation: `{OUT_VALIDATION.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append(f"- Generated folder root: `{ADAPTER_ROOT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 4. Summary")
    lines.append("")
    lines.append("| Category | Subset | Label | is_anomaly | Images | Masks | Folder root |")
    lines.append("|---|---|---|---:|---:|---:|---|")

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['category']} | {r['target_subset']} | {r['target_label']} | "
            f"{int(r['is_anomaly'])} | {int(r['num_images'])} | {int(r['num_masks'])} | "
            f"`{r['folder_root']}` |"
        )

    lines.append("")
    lines.append("## 5. Anomalib Validation")
    lines.append("")
    lines.append(f"- Successful categories: `{success_count}` / `{len(categories)}`")
    lines.append("")
    lines.append("| Category | Success | Train batch | Test batch | Error |")
    lines.append("|---|---:|---|---|---|")

    for _, r in validation.iterrows():
        err = "" if pd.isna(r.get("error", "")) else str(r.get("error", ""))
        lines.append(
            f"| {r['category']} | {r['anomalib_success']} | "
            f"{r.get('train_batch_type', '')} | {r.get('test_batch_type', '')} | `{err}` |"
        )

    lines.append("")
    lines.append("## 6. Next Step")
    lines.append("")
    lines.append("Stage 11-B should run PatchCore pilot in batch mode over all validated categories.")
    lines.append("The generated folder datasets are local artifacts and should not be committed to GitHub.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    reset_adapter_root()

    manifest = load_manifest()
    categories = sorted(manifest["category"].unique().tolist())

    mappings = []
    for category in categories:
        print(f"[BUILD] {category}")
        selected = select_rows_for_category(manifest, category)
        mappings.append(build_category_folder(selected, category))

    mapping = pd.concat(mappings, ignore_index=True)
    summary = build_summary(mapping)

    file_validation = validate_files(mapping)
    anomalib_validation = optional_anomalib_validation(summary)

    validation = file_validation.merge(
        anomalib_validation,
        on=["category", "folder_root"],
        how="left",
    )

    mapping.to_csv(OUT_MAPPING, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    validation.to_csv(OUT_VALIDATION, index=False)

    write_report(mapping, summary, validation)

    print("[DONE]", OUT_MAPPING)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_VALIDATION)
    print("[DONE]", OUT_REPORT)

    print("\n===== summary =====")
    print(summary.to_string(index=False))

    print("\n===== validation =====")
    print(validation.to_string(index=False))

    success_count = int(validation["anomalib_success"].sum())
    if success_count != len(categories):
        raise SystemExit(f"[ERROR] Anomalib validation failed for {len(categories) - success_count} categories.")


if __name__ == "__main__":
    main()
