from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly"
).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

try:
    from lightning.pytorch import Trainer
except ImportError:
    from pytorch_lightning import Trainer

from experiments.baselines.patchcore_mvtec import MVTEC_DEFECTS


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

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


def canonical_path(value: Any) -> str:
    return str(
        Path(str(value))
        .expanduser()
        .resolve(strict=False)
    ).replace("\\", "/")


def get_field(batch: Any, name: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(name)

    return getattr(batch, name, None)


def take_item(value: Any, index: int) -> Any:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return value[index]

    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        return value[index]

    try:
        return value[index]
    except Exception:
        return value


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def scalar(value: Any) -> float:
    array = to_numpy(value).reshape(-1)

    if len(array) == 0:
        return float("nan")

    return float(array[0])


def map_to_2d(value: Any) -> np.ndarray | None:
    array = to_numpy(value)

    if array.size == 0:
        return None

    array = np.squeeze(array)

    if array.ndim != 2:
        return None

    array = array.astype(np.float32)

    if not np.isfinite(array).any():
        return None

    finite = array[np.isfinite(array)]

    fill_value = (
        float(np.nanmin(finite))
        if len(finite)
        else 0.0
    )

    array = np.nan_to_num(
        array,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )

    return array


def normalize_map(array: np.ndarray) -> np.ndarray:
    minimum = float(array.min())
    maximum = float(array.max())

    if maximum <= minimum:
        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    return (
        (array - minimum)
        / (maximum - minimum)
    ).astype(np.float32)


def infer_binary_label(
    image_path: str,
    gt_label: Any,
) -> int:
    label = scalar(gt_label)

    if math.isfinite(label):
        return int(label > 0)

    normalized = image_path.replace("\\", "/")

    if "/test/good/" in normalized:
        return 0

    if "/test/" in normalized:
        return 1

    raise RuntimeError(
        f"Cannot infer label from path: {image_path}"
    )


def safe_auroc(
    y_true: list[int],
    y_score: list[float],
) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)

    valid = np.isfinite(score)

    y = y[valid]
    score = score[valid]

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        roc_auc_score(y, score)
    )


def safe_ap(
    y_true: list[int],
    y_score: list[float],
) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)

    valid = np.isfinite(score)

    y = y[valid]
    score = score[valid]

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        average_precision_score(y, score)
    )


def expected_test_images(
    data_root: Path,
    category: str,
) -> int:
    suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    return sum(
        1
        for path in (
            data_root
            / category
            / "test"
        ).rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in suffixes
        )
    )


def build_datamodule(
    args: argparse.Namespace,
    category: str,
) -> Folder:
    category_root = (
        Path(args.data_root).resolve()
        / category
    )

    if category not in MVTEC_DEFECTS:
        raise ValueError(
            f"Unknown MVTec category: {category}"
        )

    defects = MVTEC_DEFECTS[category]

    abnormal_dirs = [
        f"test/{defect}"
        for defect in defects
    ]

    mask_dirs = [
        f"ground_truth/{defect}"
        for defect in defects
    ]

    required = [
        category_root / "train/good",
        category_root / "test/good",
        *[
            category_root / path
            for path in abnormal_dirs
        ],
        *[
            category_root / path
            for path in mask_dirs
        ],
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing MVTec paths:\n"
            + "\n".join(missing)
        )

    return Folder(
        name=f"MVTecAD_{category}_stage22",
        root=str(category_root),
        normal_dir="train/good",
        abnormal_dir=abnormal_dirs,
        normal_test_dir="test/good",
        mask_dir=mask_dirs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split_mode="same_as_test",
    )


def build_model() -> Patchcore:
    pre_processor = (
        Patchcore.configure_pre_processor(
            image_size=(256, 256),
            center_crop_size=(224, 224),
        )
    )

    return Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_processor=pre_processor,
    )


def fit_and_predict(
    args: argparse.Namespace,
    category: str,
):
    datamodule = build_datamodule(
        args,
        category,
    )

    model = build_model()

    work_dir = (
        Path(args.work_root)
        / "MVTecAD"
        / category
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = Engine(
        default_root_dir=str(work_dir),
        accelerator="gpu",
        devices=1,
        logger=False,
    )

    print(
        f"[INFO] Fitting PatchCore: {category}"
    )

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print(
        f"[INFO] Predicting full test set: {category}"
    )

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=True,
    )

    if predictions is not None:
        return predictions

    # Compatibility fallback for anomalib/Lightning
    # combinations that suppress Engine.predict returns.
    print(
        "[WARN] Engine.predict returned None; "
        "using Lightning Trainer.predict fallback."
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=True,
    )

    if predictions is None:
        raise RuntimeError(
            "Prediction returned None from both "
            "Engine and Lightning Trainer."
        )

    return predictions


def component_records(
    normalized_map: np.ndarray,
    threshold: float,
    image_width: int,
    image_height: int,
    min_area: int,
    top_components: int,
) -> list[dict]:
    binary = (
        normalized_map >= threshold
    ).astype(np.uint8)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    map_height, map_width = (
        normalized_map.shape
    )

    records = []

    for component_id in range(1, count):
        x = int(
            stats[
                component_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                component_id,
                cv2.CC_STAT_TOP,
            ]
        )

        width = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        map_area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if map_area < min_area:
            continue

        component_mask = (
            labels == component_id
        )

        values = normalized_map[
            component_mask
        ]

        map_x2 = x + width
        map_y2 = y + height

        image_x1 = int(
            np.floor(
                x / map_width
                * image_width
            )
        )

        image_y1 = int(
            np.floor(
                y / map_height
                * image_height
            )
        )

        image_x2 = int(
            np.ceil(
                map_x2 / map_width
                * image_width
            )
        )

        image_y2 = int(
            np.ceil(
                map_y2 / map_height
                * image_height
            )
        )

        image_x1 = max(
            0,
            min(image_x1, image_width - 1),
        )

        image_y1 = max(
            0,
            min(image_y1, image_height - 1),
        )

        image_x2 = max(
            image_x1 + 1,
            min(image_x2, image_width),
        )

        image_y2 = max(
            image_y1 + 1,
            min(image_y2, image_height),
        )

        box_area = (
            (image_x2 - image_x1)
            * (image_y2 - image_y1)
        )

        records.append(
            {
                "x1": image_x1,
                "y1": image_y1,
                "x2": image_x2,
                "y2": image_y2,
                "map_x1": x,
                "map_y1": y,
                "map_x2": map_x2,
                "map_y2": map_y2,
                "map_area": map_area,
                "box_area": box_area,
                "box_area_ratio": (
                    box_area
                    / (image_width * image_height)
                ),
                "candidate_score_mean": (
                    float(values.mean())
                ),
                "candidate_score_max": (
                    float(values.max())
                ),
                "candidate_score_sum": (
                    float(values.sum())
                ),
            }
        )

    # Preserve the ordering used by the old
    # candidate script: area, then mean score.
    records.sort(
        key=lambda row: (
            row["map_area"],
            row["candidate_score_mean"],
        ),
        reverse=True,
    )

    return records[:top_components]


def process_predictions(
    args: argparse.Namespace,
    category: str,
    predictions,
) -> dict:
    image_rows = []
    candidate_rows = []

    for batch_index, batch in enumerate(
        predictions
    ):
        image_paths = get_field(
            batch,
            "image_path",
        )

        anomaly_maps = get_field(
            batch,
            "anomaly_map",
        )

        pred_scores = get_field(
            batch,
            "pred_score",
        )

        gt_labels = get_field(
            batch,
            "gt_label",
        )

        if image_paths is None:
            raise RuntimeError(
                "Prediction batch has no image_path."
            )

        batch_size = (
            len(image_paths)
            if isinstance(
                image_paths,
                (list, tuple),
            )
            else len(to_numpy(image_paths))
            if to_numpy(image_paths).ndim > 0
            else 1
        )

        for item_index in range(batch_size):
            image_path = str(
                take_item(
                    image_paths,
                    item_index,
                )
            )

            anomaly_map_raw = map_to_2d(
                take_item(
                    anomaly_maps,
                    item_index,
                )
            )

            if anomaly_map_raw is None:
                raise RuntimeError(
                    "Missing anomaly map for "
                    f"{image_path}"
                )

            normalized_map = normalize_map(
                anomaly_map_raw
            )

            detector_score = scalar(
                take_item(
                    pred_scores,
                    item_index,
                )
            )

            if not math.isfinite(
                detector_score
            ):
                detector_score = float(
                    anomaly_map_raw.max()
                )

            gt_binary = infer_binary_label(
                image_path=image_path,
                gt_label=take_item(
                    gt_labels,
                    item_index,
                ),
            )

            with Image.open(image_path) as image:
                image_width, image_height = (
                    image.size
                )

            threshold = float(
                np.quantile(
                    normalized_map,
                    args.threshold_quantile,
                )
            )

            candidates = component_records(
                normalized_map=normalized_map,
                threshold=threshold,
                image_width=image_width,
                image_height=image_height,
                min_area=args.min_area,
                top_components=(
                    args.top_components
                ),
            )

            canonical = canonical_path(
                image_path
            )

            image_rows.append(
                {
                    "dataset": "MVTec AD",
                    "category": category,
                    "batch_idx": batch_index,
                    "item_idx": item_index,
                    "image_path": image_path,
                    "canonical_image_path": (
                        canonical
                    ),
                    "label": (
                        "anomaly"
                        if gt_binary
                        else "normal"
                    ),
                    "gt_binary": gt_binary,
                    "patchcore_score": (
                        detector_score
                    ),
                    "anomaly_map_min": float(
                        anomaly_map_raw.min()
                    ),
                    "anomaly_map_max": float(
                        anomaly_map_raw.max()
                    ),
                    "anomaly_map_mean": float(
                        anomaly_map_raw.mean()
                    ),
                    "candidate_threshold": (
                        threshold
                    ),
                    "threshold_quantile": (
                        args.threshold_quantile
                    ),
                    "num_candidates": len(
                        candidates
                    ),
                    "image_width": image_width,
                    "image_height": image_height,
                    "map_width": int(
                        normalized_map.shape[1]
                    ),
                    "map_height": int(
                        normalized_map.shape[0]
                    ),
                }
            )

            if not candidates:
                candidate_rows.append(
                    {
                        "dataset": "MVTec AD",
                        "category": category,
                        "image_path": (
                            image_path
                        ),
                        "canonical_image_path": (
                            canonical
                        ),
                        "gt_binary": gt_binary,
                        "patchcore_score": (
                            detector_score
                        ),
                        "component_rank": 0,
                        "candidate_available": 0,
                        "threshold": threshold,
                        "threshold_quantile": (
                            args.threshold_quantile
                        ),
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                    }
                )

            for rank, candidate in enumerate(
                candidates,
                start=1,
            ):
                candidate_rows.append(
                    {
                        "dataset": "MVTec AD",
                        "category": category,
                        "image_path": (
                            image_path
                        ),
                        "canonical_image_path": (
                            canonical
                        ),
                        "gt_binary": gt_binary,
                        "patchcore_score": (
                            detector_score
                        ),
                        "component_rank": rank,
                        "candidate_available": 1,
                        "threshold": threshold,
                        "threshold_quantile": (
                            args.threshold_quantile
                        ),
                        "image_width": (
                            image_width
                        ),
                        "image_height": (
                            image_height
                        ),
                        "map_width": int(
                            normalized_map.shape[1]
                        ),
                        "map_height": int(
                            normalized_map.shape[0]
                        ),
                        **candidate,
                    }
                )

    image_df = pd.DataFrame(image_rows)
    candidate_df = pd.DataFrame(
        candidate_rows
    )

    expected = expected_test_images(
        Path(args.data_root).resolve(),
        category,
    )

    if len(image_df) != expected:
        raise RuntimeError(
            f"{category}: expected {expected} "
            f"test images, produced "
            f"{len(image_df)}."
        )

    if image_df[
        "canonical_image_path"
    ].duplicated().any():
        raise RuntimeError(
            f"{category}: duplicate image paths."
        )

    output_dir = (
        Path(args.output_root)
        / "MVTecAD"
        / category
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_csv = (
        output_dir
        / "patchcore_image_predictions.csv"
    )

    candidate_csv = (
        output_dir
        / "candidate_regions.csv"
    )

    image_df.to_csv(
        image_csv,
        index=False,
        lineterminator="\n",
    )

    candidate_df.to_csv(
        candidate_csv,
        index=False,
        lineterminator="\n",
    )

    y_true = (
        image_df["gt_binary"]
        .astype(int)
        .tolist()
    )

    y_score = (
        image_df["patchcore_score"]
        .astype(float)
        .tolist()
    )

    summary = {
        "category": category,
        "num_test_images": len(image_df),
        "num_normal": int(
            (image_df["gt_binary"] == 0).sum()
        ),
        "num_anomaly": int(
            (image_df["gt_binary"] == 1).sum()
        ),
        "image_auroc": safe_auroc(
            y_true,
            y_score,
        ),
        "image_ap": safe_ap(
            y_true,
            y_score,
        ),
        "num_candidate_rows": int(
            (
                candidate_df[
                    "candidate_available"
                ] == 1
            ).sum()
        ),
        "num_images_with_candidates": int(
            (
                image_df[
                    "num_candidates"
                ] > 0
            ).sum()
        ),
        "candidate_coverage_rate": float(
            (
                image_df[
                    "num_candidates"
                ] > 0
            ).mean()
        ),
        "threshold_quantile": (
            args.threshold_quantile
        ),
        "top_components": (
            args.top_components
        ),
        "min_area": args.min_area,
        "image_csv": str(image_csv),
        "candidate_csv": str(
            candidate_csv
        ),
    }

    summary_path = (
        output_dir
        / "patchcore_candidate_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"[DONE] {category}: "
        f"images={len(image_df)}, "
        f"AUROC={summary['image_auroc']:.6f}, "
        f"candidate coverage="
        f"{summary['candidate_coverage_rate']:.4f}"
    )
    print("[DONE]", image_csv)
    print("[DONE]", candidate_csv)
    print("[DONE]", summary_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        default="datasets/MVTecAD",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle"],
    )

    parser.add_argument(
        "--output_root",
        default=(
            "results/stage22_selective_qcr/"
            "mvtec15_rerun_patchcore"
        ),
    )

    parser.add_argument(
        "--work_root",
        default=(
            "runs/stage22_selective_qcr/"
            "mvtec15_rerun_patchcore"
        ),
    )

    parser.add_argument(
        "--threshold_quantile",
        type=float,
        default=0.97,
    )

    parser.add_argument(
        "--top_components",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    invalid = [
        category
        for category in args.categories
        if category not in MVTEC15
    ]

    if invalid:
        raise ValueError(
            f"Unknown categories: {invalid}"
        )

    if not (
        0.0
        < args.threshold_quantile
        < 1.0
    ):
        raise ValueError(
            "threshold_quantile must be "
            "between 0 and 1."
        )

    print(
        "===== STAGE 22-D4b PROTOCOL ====="
    )
    print("categories:", args.categories)
    print(
        "threshold quantile:",
        args.threshold_quantile,
    )
    print(
        "top components:",
        args.top_components,
    )
    print("minimum map area:", args.min_area)
    print("uses GT masks for candidates: False")
    print("uses test labels for threshold: False")
    print()

    for category in args.categories:
        count = expected_test_images(
            Path(args.data_root).resolve(),
            category,
        )

        print(
            f"{category}: "
            f"expected test images={count}"
        )

        build_datamodule(
            args,
            category,
        )

    if args.validate_only:
        print()
        print(
            "[OK] Validation-only run passed."
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    torch.set_float32_matmul_precision(
        "high"
    )

    started = time.perf_counter()
    summaries = []

    for category in args.categories:
        predictions = fit_and_predict(
            args,
            category,
        )

        summaries.append(
            process_predictions(
                args,
                category,
                predictions,
            )
        )

        torch.cuda.empty_cache()

    output_root = Path(args.output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_csv = (
        output_root
        / "mvtec15_patchcore_summary.csv"
    )

    pd.DataFrame(summaries).to_csv(
        summary_csv,
        index=False,
        lineterminator="\n",
    )

    print()
    print(
        "total elapsed sec:",
        f"{time.perf_counter() - started:.3f}",
    )
    print("[DONE]", summary_csv)


if __name__ == "__main__":
    main()
