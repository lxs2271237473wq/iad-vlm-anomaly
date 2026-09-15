from __future__ import annotations

import argparse
import gc
import inspect
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(".").resolve()

ADAPTER_ROOT = ROOT / "datasets" / "MVTec_AD_2_anomalib_all"
QUALITY_CSV = ROOT / "results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv"

RUN_ROOT = ROOT / "runs/stage11_mvtecad2_multicategory/candidate_region_patchcore"
OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

CROP_ROOT = OUT_DIR / "stage11_c_candidate_crops"

OUT_REGIONS = OUT_DIR / "stage11_c_candidate_regions.csv"
OUT_SUMMARY = OUT_DIR / "stage11_c_candidate_summary.csv"
OUT_STATUS = OUT_DIR / "stage11_c_candidate_status.csv"
OUT_REPORT = DOC_DIR / "stage11_c_candidate_region_analysis.md"


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


def make_folder_datamodule(category: str):
    from anomalib.data import Folder

    data_root = ADAPTER_ROOT / f"{category}_folder"

    kwargs = {
        "name": f"mvtecad2_{category}_folder",
        "root": str(data_root),
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


def make_engine(category: str):
    from anomalib.engine import Engine

    run_dir = RUN_ROOT / category

    kwargs = {
        "default_root_dir": str(run_dir),
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


def get_anomaly_map(value: Any, index: int) -> np.ndarray | None:
    if value is None:
        return None

    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.ndim >= 3:
                amap = value[index]
            else:
                amap = value

            amap = amap.detach().float().cpu().numpy()

            if amap.ndim == 3:
                amap = np.squeeze(amap)

            if amap.ndim != 2:
                return None

            return amap.astype(np.float32)
    except Exception:
        pass

    return None


def normalize_map(amap: np.ndarray) -> np.ndarray:
    amap = np.asarray(amap, dtype=np.float32)
    finite = np.isfinite(amap)

    if not finite.any():
        return np.zeros_like(amap, dtype=np.float32)

    lo = float(np.nanmin(amap[finite]))
    hi = float(np.nanmax(amap[finite]))

    if math.isclose(hi, lo):
        return np.zeros_like(amap, dtype=np.float32)

    out = (amap - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0

    return np.clip(out, 0.0, 1.0)


def connected_components(binary: np.ndarray) -> List[np.ndarray]:
    binary = np.asarray(binary).astype(bool)

    try:
        from scipy import ndimage
        labels, n = ndimage.label(binary)
        comps = []
        for label_id in range(1, n + 1):
            mask = labels == label_id
            if mask.any():
                comps.append(mask)
        return comps
    except Exception:
        ys, xs = np.where(binary)
        if len(xs) == 0:
            return []
        mask = np.zeros_like(binary, dtype=bool)
        mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] = True
        return [mask]


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 1, 1

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def bbox_map_to_image(
    bbox: Tuple[int, int, int, int],
    map_w: int,
    map_h: int,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    ix1 = max(0, min(img_w - 1, int(math.floor(x1 * img_w / map_w))))
    iy1 = max(0, min(img_h - 1, int(math.floor(y1 * img_h / map_h))))
    ix2 = max(ix1 + 1, min(img_w, int(math.ceil(x2 * img_w / map_w))))
    iy2 = max(iy1 + 1, min(img_h, int(math.ceil(y2 * img_h / map_h))))

    return ix1, iy1, ix2, iy2


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    context_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    pad_x = int(round(w * context_ratio))
    pad_y = int(round(h * context_ratio))

    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(img_w, x2 + pad_x)
    ey2 = min(img_h, y2 + pad_y)

    return ex1, ey1, ex2, ey2


def resolve_path(path_text: Any) -> Path | None:
    if path_text is None:
        return None

    text = str(path_text)
    if not text or text.lower() == "nan":
        return None

    p = Path(text)

    if p.is_absolute():
        return p

    return ROOT / p


def load_image(path_text: Any) -> Image.Image:
    p = resolve_path(path_text)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Missing image path: {path_text}")

    return Image.open(p).convert("RGB")


def load_mask_array(path_text: Any, size: Tuple[int, int]) -> np.ndarray | None:
    p = resolve_path(path_text)
    if p is None or not p.exists():
        return None

    mask = Image.open(p).convert("L")
    mask = mask.resize(size, resample=Image.NEAREST)
    arr = np.asarray(mask)

    return arr > 0


def mask_coverage(mask: np.ndarray | None, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
    if mask is None:
        return {
            "gt_mask_pixels": 0.0,
            "candidate_mask_pixels": 0.0,
            "candidate_covers_gt_ratio": "",
            "candidate_mask_density": "",
        }

    x1, y1, x2, y2 = bbox
    gt_pixels = float(mask.sum())
    region = mask[y1:y2, x1:x2]
    candidate_mask_pixels = float(region.sum())
    area = float(max(1, (x2 - x1) * (y2 - y1)))

    if gt_pixels <= 0:
        covers_gt_ratio = 0.0
    else:
        covers_gt_ratio = candidate_mask_pixels / gt_pixels

    return {
        "gt_mask_pixels": gt_pixels,
        "candidate_mask_pixels": candidate_mask_pixels,
        "candidate_covers_gt_ratio": covers_gt_ratio,
        "candidate_mask_density": candidate_mask_pixels / area,
    }


def safe_stem(path_text: Any) -> str:
    p = Path(str(path_text))
    return p.stem.replace(" ", "_").replace("/", "_")


def save_crop(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop(bbox)
    crop.save(out_path)


def propose_candidates(
    amap: np.ndarray,
    img_w: int,
    img_h: int,
    top_k: int,
    min_area_ratio: float,
) -> List[Dict[str, Any]]:
    norm = normalize_map(amap)
    map_h, map_w = norm.shape

    threshold = float(np.quantile(norm, 0.98))
    if threshold <= 0:
        threshold = float(norm.max())

    binary = norm >= threshold

    min_area = max(4, int(round(map_h * map_w * min_area_ratio)))

    comps = connected_components(binary)

    candidates = []
    for comp in comps:
        area = int(comp.sum())
        if area < min_area:
            continue

        x1, y1, x2, y2 = bbox_from_mask(comp)
        region_scores = norm[comp]

        bbox_img = bbox_map_to_image((x1, y1, x2, y2), map_w, map_h, img_w, img_h)

        candidates.append({
            "map_x1": x1,
            "map_y1": y1,
            "map_x2": x2,
            "map_y2": y2,
            "map_area": area,
            "candidate_score_max": float(region_scores.max()),
            "candidate_score_mean": float(region_scores.mean()),
            "candidate_score_sum": float(region_scores.sum()),
            "bbox_x1": bbox_img[0],
            "bbox_y1": bbox_img[1],
            "bbox_x2": bbox_img[2],
            "bbox_y2": bbox_img[3],
        })

    if not candidates:
        y, x = np.unravel_index(int(np.argmax(norm)), norm.shape)
        side = max(8, int(round(min(map_h, map_w) * 0.12)))
        x1 = max(0, int(x - side // 2))
        y1 = max(0, int(y - side // 2))
        x2 = min(map_w, x1 + side)
        y2 = min(map_h, y1 + side)

        bbox_img = bbox_map_to_image((x1, y1, x2, y2), map_w, map_h, img_w, img_h)

        candidates.append({
            "map_x1": x1,
            "map_y1": y1,
            "map_x2": x2,
            "map_y2": y2,
            "map_area": int((y2 - y1) * (x2 - x1)),
            "candidate_score_max": float(norm.max()),
            "candidate_score_mean": float(norm[y1:y2, x1:x2].mean()),
            "candidate_score_sum": float(norm[y1:y2, x1:x2].sum()),
            "bbox_x1": bbox_img[0],
            "bbox_y1": bbox_img[1],
            "bbox_x2": bbox_img[2],
            "bbox_y2": bbox_img[3],
        })

    candidates = sorted(
        candidates,
        key=lambda r: (r["candidate_score_max"], r["candidate_score_sum"], r["map_area"]),
        reverse=True,
    )

    return candidates[:top_k]


def clear_gpu_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()


def run_one_category(category: str, top_k: int, min_area_ratio: float) -> tuple[pd.DataFrame, Dict[str, Any]]:
    start = time.time()

    status = {
        "dataset": "MVTec AD 2",
        "category": category,
        "success": False,
        "fit_success": False,
        "predict_success": False,
        "num_images": 0,
        "num_candidate_rows": 0,
        "elapsed_sec": 0.0,
        "error": "",
    }

    print("")
    print(f"========== Stage 11-C candidate regions: {category} ==========")

    dm = make_folder_datamodule(category)
    model = make_patchcore_model()
    engine = make_engine(category)

    print("[INFO] fit start")
    engine.fit(model=model, datamodule=dm)
    status["fit_success"] = True
    print("[INFO] fit done")

    print("[INFO] predict start")
    predict_outputs = engine.predict(model=model, datamodule=dm)
    status["predict_success"] = True
    print("[INFO] predict done")

    rows: List[Dict[str, Any]] = []
    image_counter = 0

    for batch_idx, batch in enumerate(predict_outputs):
        n = get_n_items(batch)

        image_path = get_batch_field(batch, "image_path")
        mask_path = get_batch_field(batch, "mask_path")
        gt_label = get_batch_field(batch, "gt_label")
        pred_label = get_batch_field(batch, "pred_label")
        pred_score = get_batch_field(batch, "pred_score")
        anomaly_map = get_batch_field(batch, "anomaly_map")

        for item_idx in range(n):
            img_path = get_item_value(image_path, item_idx)
            mask_path_i = get_item_value(mask_path, item_idx)
            gt_label_i = get_item_value(gt_label, item_idx)
            pred_label_i = get_item_value(pred_label, item_idx)
            pred_score_i = get_item_value(pred_score, item_idx)

            amap = get_anomaly_map(anomaly_map, item_idx)
            if amap is None:
                continue

            image = load_image(img_path)
            img_w, img_h = image.size
            mask_arr = load_mask_array(mask_path_i, size=(img_w, img_h))

            candidates = propose_candidates(
                amap=amap,
                img_w=img_w,
                img_h=img_h,
                top_k=top_k,
                min_area_ratio=min_area_ratio,
            )

            image_id = safe_stem(img_path)
            image_counter += 1

            for cand_idx, cand in enumerate(candidates):
                tight_bbox = (
                    int(cand["bbox_x1"]),
                    int(cand["bbox_y1"]),
                    int(cand["bbox_x2"]),
                    int(cand["bbox_y2"]),
                )

                context_bbox = expand_bbox(tight_bbox, img_w=img_w, img_h=img_h, context_ratio=1.50)

                tight_name = f"{image_id}_cand{cand_idx:02d}_tight.png"
                context_name = f"{image_id}_cand{cand_idx:02d}_context_1.50.png"

                tight_path = CROP_ROOT / category / "tight" / tight_name
                context_path = CROP_ROOT / category / "context_1.50" / context_name

                save_crop(image, tight_bbox, tight_path)
                save_crop(image, context_bbox, context_path)

                cov_tight = mask_coverage(mask_arr, tight_bbox)
                cov_context = mask_coverage(mask_arr, context_bbox)

                row = {
                    "dataset": "MVTec AD 2",
                    "category": category,
                    "batch_idx": batch_idx,
                    "item_idx": item_idx,
                    "candidate_rank": cand_idx,
                    "image_path": img_path,
                    "mask_path": mask_path_i,
                    "gt_label": gt_label_i,
                    "pred_label": pred_label_i,
                    "pred_score": pred_score_i,
                    "image_width": img_w,
                    "image_height": img_h,
                    "anomaly_map_height": int(amap.shape[0]),
                    "anomaly_map_width": int(amap.shape[1]),
                    **cand,
                    "context_1p50_x1": context_bbox[0],
                    "context_1p50_y1": context_bbox[1],
                    "context_1p50_x2": context_bbox[2],
                    "context_1p50_y2": context_bbox[3],
                    "tight_crop_path": str(tight_path.relative_to(ROOT)),
                    "context_1p50_crop_path": str(context_path.relative_to(ROOT)),
                    "tight_gt_mask_pixels": cov_tight["gt_mask_pixels"],
                    "tight_candidate_mask_pixels": cov_tight["candidate_mask_pixels"],
                    "tight_candidate_covers_gt_ratio": cov_tight["candidate_covers_gt_ratio"],
                    "tight_candidate_mask_density": cov_tight["candidate_mask_density"],
                    "context_gt_mask_pixels": cov_context["gt_mask_pixels"],
                    "context_candidate_mask_pixels": cov_context["candidate_mask_pixels"],
                    "context_candidate_covers_gt_ratio": cov_context["candidate_covers_gt_ratio"],
                    "context_candidate_mask_density": cov_context["candidate_mask_density"],
                }

                rows.append(row)

    df = pd.DataFrame(rows)

    status["num_images"] = int(image_counter)
    status["num_candidate_rows"] = int(len(df))
    status["success"] = bool(status["fit_success"] and status["predict_success"] and len(df) > 0)
    status["elapsed_sec"] = round(time.time() - start, 3)

    return df, status


def load_target_categories(mode: str, explicit: str) -> List[str]:
    if explicit:
        return [x.strip() for x in explicit.split(",") if x.strip()]

    q = pd.read_csv(QUALITY_CSV)

    if mode == "primary":
        return q[q["stage11_c_priority_group"] == "primary"]["category"].tolist()

    if mode == "primary_secondary":
        return q[q["stage11_c_priority_group"].isin(["primary", "secondary"])]["category"].tolist()

    raise ValueError(f"Unsupported mode: {mode}")


def build_summary(regions: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if regions.empty:
        return pd.DataFrame()

    for category, part in regions.groupby("category"):
        top1 = part[part["candidate_rank"] == 0].copy()

        num_images = int(top1["image_path"].nunique())
        num_candidate_rows = int(len(part))
        images_with_candidates = int(part["image_path"].nunique())

        anomaly_top1 = top1[pd.to_numeric(top1["gt_label"], errors="coerce").fillna(0).astype(int) == 1]
        normal_top1 = top1[pd.to_numeric(top1["gt_label"], errors="coerce").fillna(0).astype(int) == 0]

        def numeric_mean(df: pd.DataFrame, col: str):
            if df.empty or col not in df.columns:
                return ""
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() == 0:
                return ""
            return float(vals.mean())

        rows.append({
            "dataset": "MVTec AD 2",
            "category": category,
            "num_images": num_images,
            "num_candidate_rows": num_candidate_rows,
            "images_with_candidates": images_with_candidates,
            "candidate_coverage": images_with_candidates / max(1, num_images),
            "mean_candidates_per_image": num_candidate_rows / max(1, num_images),
            "num_anomaly_images": int(len(anomaly_top1)),
            "num_normal_images": int(len(normal_top1)),
            "top1_tight_mean_gt_coverage_anomaly": numeric_mean(anomaly_top1, "tight_candidate_covers_gt_ratio"),
            "top1_context_mean_gt_coverage_anomaly": numeric_mean(anomaly_top1, "context_candidate_covers_gt_ratio"),
            "top1_tight_mean_mask_density_anomaly": numeric_mean(anomaly_top1, "tight_candidate_mask_density"),
            "top1_context_mean_mask_density_anomaly": numeric_mean(anomaly_top1, "context_candidate_mask_density"),
        })

    return pd.DataFrame(rows).sort_values("category")


def write_report(regions: pd.DataFrame, summary: pd.DataFrame, status: pd.DataFrame, categories: List[str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append("# Stage 11-C MVTec AD 2 Candidate Region Generation")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage generates PatchCore anomaly-map candidate regions for selected MVTec AD 2 categories.")
    lines.append("It prepares tight crops and context-aware crops for later VLM reasoning.")
    lines.append("")
    lines.append("This stage reruns PatchCore fit and predict to access full anomaly maps. It does not run VLM inference and does not modify the original dataset.")
    lines.append("")
    lines.append("## 2. Selected Categories")
    lines.append("")
    for category in categories:
        lines.append(f"- {category}")
    lines.append("")
    lines.append("## 3. Candidate Construction")
    lines.append("")
    lines.append("| Item | Setting |")
    lines.append("|---|---|")
    lines.append("| Detector | PatchCore |")
    lines.append("| Backbone | wide_resnet50_2 |")
    lines.append("| Layers | layer2, layer3 |")
    lines.append("| Candidate source | anomaly map top-percentile connected components |")
    lines.append("| Max candidates per image | 3 |")
    lines.append("| Tight crop | connected-component bounding box |")
    lines.append("| Context crop | tight box expanded by 1.50 times its width/height on each side |")
    lines.append("")
    lines.append("## 4. Output Files")
    lines.append("")
    lines.append(f"- Candidate regions: `{OUT_REGIONS.relative_to(ROOT)}`")
    lines.append(f"- Summary: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Status: `{OUT_STATUS.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append(f"- Local crop root: `{CROP_ROOT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("The crop root is a generated local artifact and should not be committed to GitHub.")
    lines.append("")
    lines.append("## 5. Execution Status")
    lines.append("")
    lines.append("| Category | Success | Images | Candidate rows | Time sec | Error |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for _, r in status.sort_values("category").iterrows():
        err = "" if pd.isna(r["error"]) else str(r["error"]).replace("|", "/")
        lines.append(
            f"| {r['category']} | {r['success']} | {int(r['num_images'])} | "
            f"{int(r['num_candidate_rows'])} | {float(r['elapsed_sec']):.1f} | `{err}` |"
        )

    lines.append("")
    lines.append("## 6. Candidate Summary")
    lines.append("")

    if summary.empty:
        lines.append("No candidate summary was produced.")
    else:
        lines.append("| Category | Images | Candidate rows | Coverage | Mean cand/img | Anomaly images | Top1 tight GT coverage | Top1 context GT coverage |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        for _, r in summary.sort_values("category").iterrows():
            def fmt(v):
                if v == "" or pd.isna(v):
                    return ""
                return f"{float(v):.4f}"

            lines.append(
                f"| {r['category']} | {int(r['num_images'])} | {int(r['num_candidate_rows'])} | "
                f"{float(r['candidate_coverage']):.4f} | {float(r['mean_candidates_per_image']):.4f} | "
                f"{int(r['num_anomaly_images'])} | {fmt(r['top1_tight_mean_gt_coverage_anomaly'])} | "
                f"{fmt(r['top1_context_mean_gt_coverage_anomaly'])} |"
            )

    lines.append("")
    lines.append("## 7. Interpretation")
    lines.append("")
    lines.append("This stage checks whether PatchCore can provide usable visual candidate regions for the VLM branch.")
    lines.append("The context crop is especially important because Stage 10-G showed that overly tight anomaly crops can hurt VLM reasoning when object-level context is removed.")
    lines.append("")
    lines.append("## 8. Next Step")
    lines.append("")
    lines.append("Stage 11-D should evaluate full-image versus context-aware crop VLM reasoning on these categories.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="primary", choices=["primary", "primary_secondary"])
    parser.add_argument("--categories", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    CROP_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    categories = load_target_categories(args.mode, args.categories)

    print("[INFO] Stage 11-C categories:", categories)

    all_regions = []
    status_rows = []

    for category in categories:
        start = time.time()

        try:
            regions, status = run_one_category(
                category=category,
                top_k=args.top_k,
                min_area_ratio=args.min_area_ratio,
            )

            if not regions.empty:
                all_regions.append(regions)

            status_rows.append(status)

        except Exception as e:
            err = repr(e)
            print(f"[ERROR] {category}: {err}")
            traceback.print_exc()

            status_rows.append({
                "dataset": "MVTec AD 2",
                "category": category,
                "success": False,
                "fit_success": False,
                "predict_success": False,
                "num_images": 0,
                "num_candidate_rows": 0,
                "elapsed_sec": round(time.time() - start, 3),
                "error": err,
            })

        finally:
            clear_gpu_cache()

    regions_df = pd.concat(all_regions, ignore_index=True) if all_regions else pd.DataFrame()
    status_df = pd.DataFrame(status_rows)
    summary_df = build_summary(regions_df, status_df)

    regions_df.to_csv(OUT_REGIONS, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    status_df.to_csv(OUT_STATUS, index=False)

    write_report(regions_df, summary_df, status_df, categories)

    print("[DONE]", OUT_REGIONS)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_STATUS)
    print("[DONE]", OUT_REPORT)

    print("\n===== status =====")
    print(status_df.to_string(index=False))

    print("\n===== summary =====")
    print(summary_df.to_string(index=False))

    failed = status_df[status_df["success"] != True]
    if not failed.empty:
        raise SystemExit(f"[ERROR] Failed categories: {failed['category'].tolist()}")


if __name__ == "__main__":
    main()
