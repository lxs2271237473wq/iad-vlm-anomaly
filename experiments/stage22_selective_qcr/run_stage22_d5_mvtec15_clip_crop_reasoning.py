from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STAGE7_SCRIPT = (
    ROOT
    / "experiments/stage7_generalization"
    / "visa_binary_prompt_reasoning.py"
)

INPUT_ROOT = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_rerun_patchcore"
)

OUTPUT_ROOT = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_clip_crop_reasoning"
)

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


def load_stage7_module():
    if not STAGE7_SCRIPT.exists():
        raise FileNotFoundError(STAGE7_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "stage7_binary_prompt_reasoning",
        STAGE7_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {STAGE7_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "open_clip",
        "canonical_path",
        "build_text_features",
        "get_eval_images",
        "encode_images",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            f"Stage 7 module missing: {missing}"
        )

    return module


def category_paths(category: str):
    input_dir = (
        INPUT_ROOT
        / "MVTecAD"
        / category
    )

    output_dir = (
        OUTPUT_ROOT
        / "MVTecAD"
        / category
    )

    return {
        "image_csv": (
            input_dir
            / "patchcore_image_predictions.csv"
        ),
        "candidate_csv": (
            input_dir
            / "candidate_regions.csv"
        ),
        "output_dir": output_dir,
        "prediction_csv": (
            output_dir
            / "clip_crop_predictions.csv"
        ),
        "summary_json": (
            output_dir
            / "clip_crop_summary.json"
        ),
    }


def load_boxes(
    module,
    candidate_csv: Path,
    top_k: int,
):
    df = pd.read_csv(candidate_csv)

    df["component_rank"] = pd.to_numeric(
        df["component_rank"],
        errors="coerce",
    )

    df["candidate_available"] = pd.to_numeric(
        df["candidate_available"],
        errors="coerce",
    ).fillna(0)

    df = df[
        (df["candidate_available"] == 1)
        & (df["component_rank"] > 0)
    ].copy()

    boxes = {}

    for image_path, group in df.groupby(
        "image_path",
        sort=False,
    ):
        group = group.sort_values(
            "component_rank"
        ).head(top_k)

        key = module.canonical_path(image_path)

        boxes[key] = [
            {
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "rank": int(row["component_rank"]),
            }
            for _, row in group.iterrows()
        ]

    return boxes


def validate_category(category: str):
    paths = category_paths(category)

    for key in [
        "image_csv",
        "candidate_csv",
    ]:
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])

    images = pd.read_csv(paths["image_csv"])
    candidates = pd.read_csv(
        paths["candidate_csv"]
    )

    required_images = {
        "image_path",
        "canonical_image_path",
        "label",
        "gt_binary",
        "patchcore_score",
    }

    required_candidates = {
        "image_path",
        "component_rank",
        "candidate_available",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing_images = (
        required_images
        - set(images.columns)
    )

    missing_candidates = (
        required_candidates
        - set(candidates.columns)
    )

    if missing_images:
        raise RuntimeError(
            f"{category}: missing image columns "
            f"{sorted(missing_images)}"
        )

    if missing_candidates:
        raise RuntimeError(
            f"{category}: missing candidate columns "
            f"{sorted(missing_candidates)}"
        )

    images = images[
        images["label"].isin(
            ["normal", "anomaly"]
        )
    ].copy()

    if images[
        "canonical_image_path"
    ].duplicated().any():
        raise RuntimeError(
            f"{category}: duplicate image paths"
        )

    available = pd.to_numeric(
        candidates["candidate_available"],
        errors="coerce",
    ).fillna(0)

    candidates = candidates[
        available == 1
    ]

    covered = candidates[
        "image_path"
    ].nunique()

    return {
        "paths": paths,
        "num_images": len(images),
        "num_candidates": len(candidates),
        "covered_images": covered,
    }


def safe_metric(function, y, score):
    valid = (
        np.isfinite(y)
        & np.isfinite(score)
    )

    y = y[valid].astype(int)
    score = score[valid].astype(float)

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(function(y, score))


def close_images(images):
    for image in images:
        close = getattr(image, "close", None)

        if callable(close):
            close()


def run_category(
    module,
    model,
    preprocess,
    tokenizer,
    device,
    args,
    category,
):
    inventory = validate_category(category)
    paths = inventory["paths"]

    paths["output_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    images = pd.read_csv(paths["image_csv"])

    images = images[
        images["label"].isin(
            ["normal", "anomaly"]
        )
    ].copy().reset_index(drop=True)

    boxes = load_boxes(
        module=module,
        candidate_csv=paths["candidate_csv"],
        top_k=args.top_k,
    )

    runtime_args = SimpleNamespace(
        patchcore_root=str(INPUT_ROOT),
        top_k=args.top_k,
        map_size=args.map_size,
        crop_padding=args.crop_padding,
        min_crop_size=args.min_crop_size,
    )

    text_features, prompt_row = (
        module.build_text_features(
            model=model,
            tokenizer=tokenizer,
            category=category,
            strategy="inspection_binary",
            device=device,
        )
    )

    rows = []
    started = time.perf_counter()

    for index, row in images.iterrows():
        eval_images = []

        try:
            (
                eval_images,
                used_mode,
                fallback,
            ) = module.get_eval_images(
                row,
                boxes,
                "crop_topk_ensemble",
                runtime_args,
            )

            features = module.encode_images(
                model,
                preprocess,
                eval_images,
                device,
            )

            similarities = (
                features
                @ text_features.T
            ).detach().cpu().numpy()

            margins = (
                similarities[:, 1]
                - similarities[:, 0]
            )

            best_index = int(
                np.argmax(margins)
            )

            rows.append(
                {
                    "dataset": "MVTec AD",
                    "category": category,
                    "image_path": row["image_path"],
                    "canonical_image_path": (
                        module.canonical_path(
                            row["image_path"]
                        )
                    ),
                    "label": row["label"],
                    "gt_binary": int(
                        row["gt_binary"]
                    ),
                    "patchcore_score": float(
                        row["patchcore_score"]
                    ),
                    "vlm_anomaly_score": float(
                        margins[best_index]
                    ),
                    "best_normal_similarity": float(
                        similarities[
                            best_index,
                            0,
                        ]
                    ),
                    "best_anomaly_similarity": float(
                        similarities[
                            best_index,
                            1,
                        ]
                    ),
                    "best_crop_index": best_index,
                    "fallback": int(fallback),
                    "used_mode": used_mode,
                    "num_eval_images": len(
                        eval_images
                    ),
                    "strategy": (
                        "inspection_binary"
                    ),
                    "eval_mode": (
                        "crop_topk_ensemble"
                    ),
                    "clip_model": args.clip_model,
                    "clip_pretrained": (
                        args.clip_pretrained
                    ),
                }
            )

        finally:
            close_images(eval_images)

        if (
            index == 0
            or (index + 1) % 25 == 0
            or index + 1 == len(images)
        ):
            print(
                f"[{category}] "
                f"{index + 1}/{len(images)}"
            )

    result = pd.DataFrame(rows)

    if len(result) != len(images):
        raise RuntimeError(
            f"{category}: expected {len(images)} "
            f"predictions, got {len(result)}"
        )

    if result[
        "canonical_image_path"
    ].duplicated().any():
        raise RuntimeError(
            f"{category}: duplicate outputs"
        )

    y = result[
        "gt_binary"
    ].to_numpy(dtype=int)

    score = result[
        "vlm_anomaly_score"
    ].to_numpy(dtype=float)

    auroc = safe_metric(
        roc_auc_score,
        y,
        score,
    )

    ap = safe_metric(
        average_precision_score,
        y,
        score,
    )

    elapsed = (
        time.perf_counter() - started
    )

    summary = {
        "category": category,
        "num_images": len(result),
        "num_normal": int(
            (result["gt_binary"] == 0).sum()
        ),
        "num_anomaly": int(
            (result["gt_binary"] == 1).sum()
        ),
        "vlm_image_auroc": auroc,
        "vlm_image_ap": ap,
        "fallback_count": int(
            result["fallback"].sum()
        ),
        "fallback_rate": float(
            result["fallback"].mean()
        ),
        "mean_eval_images": float(
            result["num_eval_images"].mean()
        ),
        "elapsed_sec": elapsed,
        "sec_per_image": (
            elapsed / len(result)
        ),
        "strategy": "inspection_binary",
        "eval_mode": "crop_topk_ensemble",
        "clip_model": args.clip_model,
        "clip_pretrained": (
            args.clip_pretrained
        ),
        "normal_prompts": (
            prompt_row["normal_prompts"]
        ),
        "anomaly_prompts": (
            prompt_row["anomaly_prompts"]
        ),
    }

    result.to_csv(
        paths["prediction_csv"],
        index=False,
        lineterminator="\n",
    )

    paths["summary_json"].write_text(
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
        f"AUROC={auroc:.6f}, "
        f"AP={ap:.6f}, "
        f"fallback={summary['fallback_rate']:.4f}, "
        f"time={elapsed:.2f}s"
    )

    print("[DONE]", paths["prediction_csv"])
    print("[DONE]", paths["summary_json"])


def rebuild_summary():
    rows = []

    for category in MVTEC15:
        path = (
            OUTPUT_ROOT
            / "MVTecAD"
            / category
            / "clip_crop_summary.json"
        )

        if path.exists():
            rows.append(
                json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            )

    output = (
        OUTPUT_ROOT
        / "mvtec15_clip_crop_summary_all.csv"
    )

    if rows:
        pd.DataFrame(rows).sort_values(
            "category"
        ).to_csv(
            output,
            index=False,
            lineterminator="\n",
        )

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle"],
    )

    parser.add_argument(
        "--clip_model",
        default="ViT-B-32",
    )

    parser.add_argument(
        "--clip_pretrained",
        default="openai",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--map_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--crop_padding",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min_crop_size",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--force",
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

    module = load_stage7_module()

    print("===== STAGE 22-D5 PROTOCOL =====")
    print("categories:", args.categories)
    print("strategy: inspection_binary")
    print("eval mode: crop_topk_ensemble")
    print(
        "CLIP:",
        f"{args.clip_model}/{args.clip_pretrained}",
    )
    print()

    inventories = {}

    for category in args.categories:
        inventory = validate_category(
            category
        )

        inventories[category] = inventory

        print(
            f"{category}: "
            f"images={inventory['num_images']}, "
            f"candidates="
            f"{inventory['num_candidates']}, "
            f"covered="
            f"{inventory['covered_images']}"
        )

    if args.validate_only:
        print()
        print(
            "[OK] Validation-only run passed."
        )
        return

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    print()
    print("[INFO] Loading CLIP once...")

    model, _, preprocess = (
        module.open_clip
        .create_model_and_transforms(
            args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
        )
    )

    tokenizer = (
        module.open_clip.get_tokenizer(
            args.clip_model
        )
    )

    model.eval()

    for category in args.categories:
        paths = inventories[
            category
        ]["paths"]

        if (
            paths["prediction_csv"].exists()
            and paths["summary_json"].exists()
            and not args.force
        ):
            print(
                f"[SKIP] {category}: "
                "outputs already exist"
            )
            continue

        run_category(
            module=module,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=args.device,
            args=args,
            category=category,
        )

        torch.cuda.empty_cache()

    output = rebuild_summary()

    print()
    print("[DONE]", output)


if __name__ == "__main__":
    main()
