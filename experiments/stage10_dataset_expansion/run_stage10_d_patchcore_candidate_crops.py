from __future__ import annotations

import inspect
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(".").resolve()

DATA_ROOT = ROOT / "datasets" / "MVTec_AD_2_anomalib" / "vial_folder"
RUN_DIR = ROOT / "runs" / "stage10_mvtecad2_patchcore" / "vial_patchcore_candidate_crops"

OUT_DIR = ROOT / "results" / "stage10_dataset_expansion"
DOC_DIR = ROOT / "docs" / "stage10_dataset_expansion"

CROP_DIR = OUT_DIR / "stage10_d_patchcore_candidate_crops"

OUT_CSV = OUT_DIR / "stage10_d_patchcore_candidate_regions.csv"
OUT_SUMMARY = OUT_DIR / "stage10_d_patchcore_candidate_summary.csv"
OUT_REPORT = DOC_DIR / "stage10_d_patchcore_candidate_report.md"

TOP_K = 3
THRESHOLD_QUANTILE = 0.97
MIN_AREA_RATIO = 0.0005
PADDING_RATIO = 0.12


def filter_kwargs(cls_or_fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls_or_fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def import_patchcore_class():
    try:
        from anomalib.models import Patchcore
        return Patchcore
    except Exception:
        pass

    try:
        from anomalib.models import PatchCore
        return PatchCore
    except Exception:
        pass

    try:
        from anomalib.models.image.patchcore import Patchcore
        return Patchcore
    except Exception as e:
        raise ImportError(f"Cannot import PatchCore/Patchcore from anomalib: {repr(e)}")


def make_folder_datamodule():
    from anomalib.data import Folder

    kwargs = {
        "name": "mvtecad2_vial_folder",
        "root": str(DATA_ROOT),
        "normal_dir": "train/good",
        "abnormal_dir": "test/bad",
        "normal_test_dir": "test/good",
        "mask_dir": "ground_truth/bad",
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_workers": 0,
    }

    return Folder(**filter_kwargs(Folder, kwargs))


def make_patchcore_model():
    PatchCoreCls = import_patchcore_class()

    kwargs = {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "pre_trained": True,
        "num_neighbors": 9,
    }

    return PatchCoreCls(**filter_kwargs(PatchCoreCls, kwargs))


def make_engine():
    from anomalib.engine import Engine

    kwargs = {
        "default_root_dir": str(RUN_DIR),
        "accelerator": "auto",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
    }

    return Engine(**filter_kwargs(Engine, kwargs))


def to_plain_value(x: Any) -> Any:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.detach().cpu().item()
            return x.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(x, Path):
        return str(x)

    return x


def get_batch_field(batch: Any, field: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(field, None)

    if hasattr(batch, field):
        return getattr(batch, field)

    return None


def get_n_items(batch: Any) -> int:
    image_path = get_batch_field(batch, "image_path")
    if image_path is not None:
        try:
            return len(image_path)
        except Exception:
            pass

    image = get_batch_field(batch, "image")
    if image is not None:
        try:
            return int(image.shape[0])
        except Exception:
            pass

    anomaly_map = get_batch_field(batch, "anomaly_map")
    if anomaly_map is not None:
        try:
            return int(anomaly_map.shape[0])
        except Exception:
            pass

    return 1


def get_item_value(value: Any, index: int) -> Any:
    if value is None:
        return ""

    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return to_plain_value(value)
            return to_plain_value(value[index])
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        if index < len(value):
            return to_plain_value(value[index])
        return ""

    return to_plain_value(value)


def tensor_to_numpy_map(value: Any, index: int) -> np.ndarray | None:
    if value is None:
        return None

    try:
        import torch

        if isinstance(value, torch.Tensor):
            amap = value[index] if value.ndim >= 3 else value
            amap = amap.detach().float().cpu().numpy()

            while amap.ndim > 2:
                amap = amap.squeeze(0)

            if amap.ndim != 2:
                return None

            amap = np.asarray(amap, dtype=np.float32)
            if not np.isfinite(amap).all():
                amap = np.nan_to_num(amap, nan=0.0, posinf=0.0, neginf=0.0)

            return amap
    except Exception:
        pass

    return None


def resize_map_to_image(amap: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size

    if amap.shape == (height, width):
        return amap.astype(np.float32)

    amin = float(amap.min())
    amax = float(amap.max())

    if math.isclose(amin, amax):
        scaled = np.zeros_like(amap, dtype=np.uint8)
    else:
        scaled = ((amap - amin) / (amax - amin) * 255.0).clip(0, 255).astype(np.uint8)

    pil = Image.fromarray(scaled, mode="L")
    pil = pil.resize((width, height), Image.BILINEAR)

    resized = np.asarray(pil).astype(np.float32) / 255.0
    return resized


def connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            comp: List[Tuple[int, int]] = []
            q = deque([(y, x)])
            visited[y, x] = True

            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))

                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        q.append((ny, nx))

            components.append(comp)

    return components


def extract_candidates(amap: np.ndarray, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    h, w = amap.shape
    min_area = max(4, int(h * w * MIN_AREA_RATIO))

    lo = float(amap.min())
    hi = float(amap.max())

    if math.isclose(lo, hi):
        return []

    threshold = float(np.quantile(amap, THRESHOLD_QUANTILE))
    binary = amap >= threshold

    comps = connected_components(binary)
    rows: List[Dict[str, Any]] = []

    for comp in comps:
        if len(comp) < min_area:
            continue

        ys = np.array([p[0] for p in comp], dtype=int)
        xs = np.array([p[1] for p in comp], dtype=int)

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1

        area = int(len(comp))
        values = amap[ys, xs]

        rows.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "area": area,
            "mean_score": float(values.mean()),
            "max_score": float(values.max()),
            "threshold": threshold,
            "map_min": lo,
            "map_max": hi,
            "map_mean": float(amap.mean()),
        })

    rows = sorted(rows, key=lambda r: (r["mean_score"], r["max_score"], r["area"]), reverse=True)
    return rows[:top_k]


def padded_box(box: Dict[str, Any], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * PADDING_RATIO)

    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(width, x2 + pad)
    py2 = min(height, y2 + pad)

    return px1, py1, px2, py2


def normalize_image_path(path_value: Any) -> Path:
    text = str(path_value)

    if text.startswith("/"):
        return Path(text)

    return ROOT / text


def save_crop(image: Image.Image, box: Tuple[int, int, int, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop(box)
    crop.save(out_path)


def reset_crop_dir() -> None:
    if CROP_DIR.exists():
        shutil.rmtree(CROP_DIR)
    CROP_DIR.mkdir(parents=True, exist_ok=True)


def build_records_from_predictions(outputs: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if outputs is None:
        return records

    for batch_idx, batch in enumerate(outputs):
        n = get_n_items(batch)

        image_path_value = get_batch_field(batch, "image_path")
        mask_path_value = get_batch_field(batch, "mask_path")
        gt_label_value = get_batch_field(batch, "gt_label")
        pred_label_value = get_batch_field(batch, "pred_label")
        pred_score_value = get_batch_field(batch, "pred_score")
        anomaly_map_value = get_batch_field(batch, "anomaly_map")

        for item_idx in range(n):
            image_path_raw = get_item_value(image_path_value, item_idx)
            mask_path_raw = get_item_value(mask_path_value, item_idx)
            gt_label = get_item_value(gt_label_value, item_idx)
            pred_label = get_item_value(pred_label_value, item_idx)
            pred_score = get_item_value(pred_score_value, item_idx)

            image_path = normalize_image_path(image_path_raw)

            base_record = {
                "dataset": "MVTec AD 2",
                "category": "vial",
                "batch_idx": batch_idx,
                "item_idx": item_idx,
                "image_path": str(image_path.relative_to(ROOT)) if image_path.is_absolute() and str(image_path).startswith(str(ROOT)) else str(image_path),
                "mask_path": str(mask_path_raw),
                "gt_label": gt_label,
                "pred_label": pred_label,
                "pred_score": pred_score,
            }

            if not image_path.exists():
                row = dict(base_record)
                row.update({
                    "candidate_rank": -1,
                    "candidate_available": False,
                    "error": f"missing_image_path: {image_path}",
                })
                records.append(row)
                continue

            image = Image.open(image_path).convert("RGB")
            amap = tensor_to_numpy_map(anomaly_map_value, item_idx)

            if amap is None:
                row = dict(base_record)
                row.update({
                    "candidate_rank": -1,
                    "candidate_available": False,
                    "error": "missing_anomaly_map",
                })
                records.append(row)
                continue

            amap_resized = resize_map_to_image(amap, image.size)
            candidates = extract_candidates(amap_resized)

            if not candidates:
                row = dict(base_record)
                row.update({
                    "candidate_rank": -1,
                    "candidate_available": False,
                    "error": "no_candidate_after_threshold",
                    "map_min": float(amap_resized.min()),
                    "map_max": float(amap_resized.max()),
                    "map_mean": float(amap_resized.mean()),
                })
                records.append(row)
                continue

            label_dir = "bad" if str(gt_label).lower() in {"1", "true", "tensor(1)"} or str(gt_label) == "1" else "good"
            image_stem = Path(str(image_path)).stem

            for rank, cand in enumerate(candidates, start=1):
                px1, py1, px2, py2 = padded_box(cand, image.size)

                crop_name = f"{image_stem}_cand{rank}.png"
                crop_path = CROP_DIR / label_dir / crop_name
                save_crop(image, (px1, py1, px2, py2), crop_path)

                row = dict(base_record)
                row.update(cand)
                row.update({
                    "candidate_rank": rank,
                    "candidate_available": True,
                    "crop_path": str(crop_path.relative_to(ROOT)),
                    "crop_x1": px1,
                    "crop_y1": py1,
                    "crop_x2": px2,
                    "crop_y2": py2,
                    "image_width": image.size[0],
                    "image_height": image.size[1],
                    "error": "",
                })
                records.append(row)

    return records


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "dataset": "MVTec AD 2",
            "category": "vial",
            "num_images": 0,
            "num_candidate_rows": 0,
            "num_images_with_candidates": 0,
            "candidate_coverage": 0.0,
        }])

    per_image = (
        df.groupby(["dataset", "category", "image_path"], as_index=False)
        .agg(
            gt_label=("gt_label", "first"),
            pred_score=("pred_score", "first"),
            has_candidate=("candidate_available", "max"),
            num_candidates=("candidate_available", "sum"),
        )
    )

    rows = []

    for keys, part in per_image.groupby(["dataset", "category"], as_index=False):
        dataset, category = keys

        rows.append({
            "dataset": dataset,
            "category": category,
            "num_images": int(len(part)),
            "num_candidate_rows": int(df["candidate_available"].sum()),
            "num_images_with_candidates": int(part["has_candidate"].sum()),
            "candidate_coverage": float(part["has_candidate"].mean()) if len(part) else 0.0,
            "mean_candidates_per_image": float(part["num_candidates"].mean()) if len(part) else 0.0,
            "mean_pred_score": float(pd.to_numeric(part["pred_score"], errors="coerce").mean()),
        })

    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, summary_df: pd.DataFrame, predict_error: str) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Stage 10-D PatchCore Candidate Crop Extraction")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage extracts candidate anomaly regions from PatchCore anomaly maps on MVTec AD 2 vial.")
    lines.append("It reruns PatchCore fit/predict because Stage 10-C stored only metrics and summarized predictions.")
    lines.append("")
    lines.append("## 2. Input")
    lines.append("")
    lines.append(f"- Folder dataset: `{DATA_ROOT.relative_to(ROOT)}`")
    lines.append("- Model: PatchCore / wide_resnet50_2 / layer2+layer3")
    lines.append("")
    lines.append("## 3. Candidate Extraction Rule")
    lines.append("")
    lines.append(f"- Threshold quantile: `{THRESHOLD_QUANTILE}`")
    lines.append(f"- Top-k candidates per image: `{TOP_K}`")
    lines.append(f"- Minimum area ratio: `{MIN_AREA_RATIO}`")
    lines.append(f"- Padding ratio: `{PADDING_RATIO}`")
    lines.append("")
    lines.append("## 4. Output Files")
    lines.append("")
    lines.append(f"- Candidate CSV: `{OUT_CSV.relative_to(ROOT)}`")
    lines.append(f"- Summary CSV: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Crop directory: `{CROP_DIR.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 5. Summary")
    lines.append("")
    lines.append("| Dataset | Category | Images | Candidate rows | Images with candidates | Coverage | Mean candidates/image | Mean pred score |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['category']} | {int(r['num_images'])} | "
            f"{int(r['num_candidate_rows'])} | {int(r['num_images_with_candidates'])} | "
            f"{float(r['candidate_coverage']):.4f} | {float(r['mean_candidates_per_image']):.4f} | "
            f"{float(r['mean_pred_score']):.4f} |"
        )

    lines.append("")
    lines.append("## 6. Error Status")
    lines.append("")
    if predict_error:
        lines.append(f"- Predict error: `{predict_error}`")
    else:
        lines.append("- Predict error: ``")

    if not df.empty and "error" in df.columns:
        err_counts = df["error"].fillna("").value_counts().to_dict()
        lines.append("")
        lines.append("Error counts:")
        lines.append("")
        lines.append("| Error | Count |")
        lines.append("|---|---:|")
        for k, v in err_counts.items():
            label = k if k else "none"
            lines.append(f"| {label} | {int(v)} |")

    lines.append("")
    lines.append("## 7. Next Step")
    lines.append("")
    lines.append("Stage 10-E should run VLM binary prompt reasoning on full images versus PatchCore candidate crops.")
    lines.append("The crop directory is generated locally and should not be committed to GitHub unless a small qualitative subset is selected.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    reset_crop_dir()

    print("[INFO] Data root:", DATA_ROOT)
    print("[INFO] Crop dir:", CROP_DIR)

    dm = make_folder_datamodule()
    model = make_patchcore_model()
    engine = make_engine()

    print("[INFO] Start PatchCore fit ...")
    engine.fit(model=model, datamodule=dm)
    print("[INFO] PatchCore fit finished.")

    predict_error = ""
    records: List[Dict[str, Any]] = []

    try:
        print("[INFO] Start PatchCore predict ...")
        outputs = engine.predict(model=model, datamodule=dm)
        records = build_records_from_predictions(outputs)
        print("[INFO] PatchCore predict finished.")
    except Exception as e:
        predict_error = repr(e)
        print("[WARN] Predict/candidate extraction failed:", predict_error)

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame([{
            "dataset": "MVTec AD 2",
            "category": "vial",
            "candidate_available": False,
            "error": predict_error or "empty_records",
        }])

    summary_df = summarize(df)

    df.to_csv(OUT_CSV, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    write_report(df, summary_df, predict_error)

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print("")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
