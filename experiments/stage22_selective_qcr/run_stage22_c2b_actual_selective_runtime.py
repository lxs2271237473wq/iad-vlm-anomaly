from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

REASONING_SCRIPT = (
    ROOT
    / "experiments/stage7_generalization"
    / "visa_binary_prompt_reasoning.py"
)

PATCHCORE_ROOT = (
    ROOT
    / "results/stage7_generalization"
    / "visa_patchcore"
)

GATE_PREDICTIONS = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_b1_visa_patchcore_loco_predictions.csv"
)

VIEW_ROOT = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_c2b_selective_runtime_view"
)

OUT_DIR = ROOT / "results/stage22_selective_qcr"
DOC_DIR = ROOT / "docs/stage22_selective_qcr"

OUT_INVENTORY = (
    OUT_DIR
    / "stage22_c2b_actual_selective_runtime_inventory.csv"
)

OUT_RUNS = (
    OUT_DIR
    / "stage22_c2b_actual_selective_runtime_runs.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "stage22_c2b_actual_selective_runtime_summary.json"
)

OUT_REPORT = (
    DOC_DIR
    / "stage22_c2b_actual_selective_runtime_report.md"
)

TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "t",
}


def load_reasoning_module():
    if not REASONING_SCRIPT.exists():
        raise FileNotFoundError(REASONING_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "stage7_visa_binary_prompt_reasoning",
        REASONING_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {REASONING_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "VISA_CATEGORIES",
        "canonical_path",
        "load_candidate_boxes",
        "get_eval_images",
        "build_text_features",
        "encode_images",
        "evaluate_category",
        "open_clip",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            f"Reasoning module missing functions: {missing}"
        )

    return module


def normalize_path(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")

    if not text:
        return ""

    try:
        return str(
            Path(text)
            .expanduser()
            .resolve(strict=False)
        ).replace("\\", "/")
    except Exception:
        return text


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(TRUE_VALUES)
    )


def load_gate_lookup() -> tuple[dict, pd.DataFrame]:
    if not GATE_PREDICTIONS.exists():
        raise FileNotFoundError(GATE_PREDICTIONS)

    gate = pd.read_csv(GATE_PREDICTIONS)

    filters = {
        "dataset": "VisA",
        "strategy": "inspection_binary",
        "eval_mode": "crop_topk_ensemble",
    }

    for column, value in filters.items():
        if column in gate.columns:
            gate = gate[
                gate[column].astype(str) == value
            ]

    if "backbone" in gate.columns:
        gate = gate[
            gate["backbone"]
            .astype(str)
            .str.contains(
                "patchcore",
                case=False,
                regex=False,
            )
        ]

    required = [
        "category",
        "image_key",
        "srb_pre_gate",
    ]

    missing = [
        column
        for column in required
        if column not in gate.columns
    ]

    if missing:
        raise RuntimeError(
            f"Gate prediction table missing: {missing}"
        )

    gate = gate.drop_duplicates(
        subset=["category", "image_key"]
    ).reset_index(drop=True)

    gate["gate_bool"] = as_bool(
        gate["srb_pre_gate"]
    )

    lookup: dict[
        tuple[str, str],
        bool,
    ] = {}

    for _, row in gate.iterrows():
        category = str(row["category"])
        passed = bool(row["gate_bool"])

        candidate_paths = [
            row.get("image_key", ""),
            row.get("image_path", ""),
        ]

        for candidate_path in candidate_paths:
            key = normalize_path(candidate_path)

            if not key:
                continue

            lookup[(category, key)] = passed

    return lookup, gate


def lookup_gate(
    module,
    lookup: dict,
    category: str,
    row: pd.Series,
) -> bool | None:
    candidates = [
        row.get("canonical_image_path", ""),
        row.get("image_path", ""),
    ]

    try:
        candidates.append(
            module.canonical_path(
                row["image_path"]
            )
        )
    except Exception:
        pass

    for candidate in candidates:
        key = normalize_path(candidate)

        if (category, key) in lookup:
            return bool(
                lookup[(category, key)]
            )

    return None


def build_selective_view(
    module,
) -> pd.DataFrame:
    lookup, gate = load_gate_lookup()

    if VIEW_ROOT.exists():
        shutil.rmtree(VIEW_ROOT)

    inventory_rows = []
    unmatched_examples = []

    categories = list(
        module.VISA_CATEGORIES
    )

    for category in categories:
        source_category = (
            PATCHCORE_ROOT
            / "VisA"
            / category
        )

        source_predictions = (
            source_category
            / "patchcore_image_predictions.csv"
        )

        source_candidates = (
            source_category
            / "candidate_regions.csv"
        )

        if not source_predictions.exists():
            raise FileNotFoundError(
                source_predictions
            )

        if not source_candidates.exists():
            raise FileNotFoundError(
                source_candidates
            )

        df = pd.read_csv(
            source_predictions
        )

        df = df[
            df["label"].isin(
                ["normal", "anomaly"]
            )
        ].copy().reset_index(drop=True)

        decisions = []
        unmatched = 0

        for _, row in df.iterrows():
            decision = lookup_gate(
                module=module,
                lookup=lookup,
                category=category,
                row=row,
            )

            if decision is None:
                unmatched += 1

                if len(unmatched_examples) < 20:
                    unmatched_examples.append(
                        {
                            "category": category,
                            "image_path": row.get(
                                "image_path",
                                "",
                            ),
                            "canonical_image_path": (
                                row.get(
                                    "canonical_image_path",
                                    "",
                                )
                            ),
                        }
                    )

                decisions.append(False)
            else:
                decisions.append(decision)

        df["stage22_srb_pre_gate"] = decisions

        selected = df[
            df["stage22_srb_pre_gate"]
        ].copy()

        destination_category = (
            VIEW_ROOT
            / "VisA"
            / category
        )

        destination_category.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_predictions = (
            destination_category
            / "patchcore_image_predictions.csv"
        )

        destination_candidates = (
            destination_category
            / "candidate_regions.csv"
        )

        selected.to_csv(
            destination_predictions,
            index=False,
            lineterminator="\n",
        )

        shutil.copy2(
            source_candidates,
            destination_candidates,
        )

        inventory_rows.append(
            {
                "category": category,
                "full_images": len(df),
                "selective_images": len(
                    selected
                ),
                "calls_saved": (
                    len(df) - len(selected)
                ),
                "selective_call_rate": (
                    len(selected) / len(df)
                    if len(df)
                    else float("nan")
                ),
                "unmatched_images": unmatched,
                "source_predictions": str(
                    source_predictions
                ),
                "selective_predictions": str(
                    destination_predictions
                ),
            }
        )

    inventory = pd.DataFrame(
        inventory_rows
    )

    total_unmatched = int(
        inventory[
            "unmatched_images"
        ].sum()
    )

    if total_unmatched:
        raise RuntimeError(
            "Some PatchCore prediction rows could not "
            "be matched to Stage 22-B1 gate decisions.\n"
            f"Total unmatched: {total_unmatched}\n"
            f"Examples: {unmatched_examples}"
        )

    expected_gate_rows = len(gate)
    full_images = int(
        inventory["full_images"].sum()
    )

    if expected_gate_rows != full_images:
        raise RuntimeError(
            "Gate and PatchCore image counts differ.\n"
            f"Gate rows: {expected_gate_rows}\n"
            f"PatchCore rows: {full_images}"
        )

    return inventory


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def load_clip_model(
    module,
    device: str,
    clip_model: str,
    clip_pretrained: str,
):
    synchronize(device)

    started = time.perf_counter()

    model, _, preprocess = (
        module.open_clip
        .create_model_and_transforms(
            clip_model,
            pretrained=clip_pretrained,
            device=device,
        )
    )

    tokenizer = (
        module.open_clip
        .get_tokenizer(clip_model)
    )

    model.eval()

    synchronize(device)

    elapsed = (
        time.perf_counter() - started
    )

    return (
        model,
        preprocess,
        tokenizer,
        elapsed,
    )


def warm_up(
    module,
    model,
    preprocess,
    tokenizer,
    device: str,
    category: str,
    top_k: int,
    crop_padding: int,
    min_crop_size: int,
) -> None:
    pred_csv = (
        PATCHCORE_ROOT
        / "VisA"
        / category
        / "patchcore_image_predictions.csv"
    )

    df = pd.read_csv(pred_csv)

    df = df[
        df["label"].isin(
            ["normal", "anomaly"]
        )
    ].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            f"No warm-up images for {category}"
        )

    boxes = module.load_candidate_boxes(
        str(PATCHCORE_ROOT),
        category,
        top_k,
    )

    runtime_args = SimpleNamespace(
        patchcore_root=str(
            PATCHCORE_ROOT
        ),
        top_k=top_k,
        map_size=224,
        crop_padding=crop_padding,
        min_crop_size=min_crop_size,
    )

    text_features, _ = (
        module.build_text_features(
            model=model,
            tokenizer=tokenizer,
            category=category,
            strategy="inspection_binary",
            device=device,
        )
    )

    row = df.iloc[0]

    eval_images, _, _ = (
        module.get_eval_images(
            row,
            boxes,
            "crop_topk_ensemble",
            runtime_args,
        )
    )

    for _ in range(5):
        image_features = (
            module.encode_images(
                model,
                preprocess,
                eval_images,
                device,
            )
        )

        _ = (
            image_features
            @ text_features.T
        ).detach()

    synchronize(device)

    for image in eval_images:
        close = getattr(
            image,
            "close",
            None,
        )

        if callable(close):
            close()


def benchmark_mode(
    module,
    model,
    preprocess,
    tokenizer,
    device: str,
    patchcore_root: Path,
    mode: str,
    repeat: int,
    top_k: int,
    crop_padding: int,
    min_crop_size: int,
) -> dict:
    runtime_args = SimpleNamespace(
        patchcore_root=str(
            patchcore_root
        ),
        top_k=top_k,
        map_size=224,
        crop_padding=crop_padding,
        min_crop_size=min_crop_size,
    )

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    synchronize(device)

    started = time.perf_counter()

    total_calls = 0
    total_eval_images = 0
    total_fallbacks = 0
    score_checksum = 0.0

    for category_index, category in enumerate(
        module.VISA_CATEGORIES,
        start=1,
    ):
        print(
            f"[{mode} repeat={repeat}] "
            f"{category_index}/"
            f"{len(module.VISA_CATEGORIES)} "
            f"{category}"
        )

        summary, details, _ = (
            module.evaluate_category(
                args=runtime_args,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                category=category,
                strategy="inspection_binary",
                eval_mode="crop_topk_ensemble",
            )
        )

        total_calls += len(details)

        total_eval_images += int(
            sum(
                int(
                    row.get(
                        "num_eval_images",
                        0,
                    )
                )
                for row in details
            )
        )

        total_fallbacks += int(
            sum(
                int(
                    row.get(
                        "fallback",
                        0,
                    )
                )
                for row in details
            )
        )

        score_checksum += float(
            sum(
                float(
                    row.get(
                        "vlm_anomaly_score",
                        0.0,
                    )
                )
                for row in details
            )
        )

    synchronize(device)

    elapsed = (
        time.perf_counter() - started
    )

    if device.startswith("cuda"):
        peak_allocated_mib = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        peak_reserved_mib = (
            torch.cuda.max_memory_reserved()
            / 1024**2
        )
    else:
        peak_allocated_mib = 0.0
        peak_reserved_mib = 0.0

    return {
        "mode": mode,
        "repeat": repeat,
        "wall_time_sec": elapsed,
        "actual_vlm_calls": total_calls,
        "actual_crop_encodings": (
            total_eval_images
        ),
        "fallback_calls": total_fallbacks,
        "sec_per_vlm_call": (
            elapsed / total_calls
            if total_calls
            else float("nan")
        ),
        "sec_per_dataset_image": (
            elapsed
            / int(
                pd.read_csv(
                    GATE_PREDICTIONS
                )[
                    lambda frame: (
                        frame["backbone"]
                        .astype(str)
                        .str.contains(
                            "patchcore",
                            case=False,
                            regex=False,
                        )
                    )
                ]
                .drop_duplicates(
                    subset=[
                        "category",
                        "image_key",
                    ]
                )
                .shape[0]
            )
        ),
        "peak_gpu_allocated_mib": (
            peak_allocated_mib
        ),
        "peak_gpu_reserved_mib": (
            peak_reserved_mib
        ),
        "score_checksum": score_checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
    )

    parser.add_argument(
        "--clip-pretrained",
        default="openai",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--crop-padding",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be >= 1"
        )

    module = load_reasoning_module()

    inventory = build_selective_view(module)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        OUT_INVENTORY,
        index=False,
        lineterminator="\n",
    )

    full_images = int(
        inventory["full_images"].sum()
    )

    selective_images = int(
        inventory[
            "selective_images"
        ].sum()
    )

    call_rate = (
        selective_images / full_images
    )

    print(
        "===== STAGE 22-C2b VALIDATION ====="
    )
    print("full images:", full_images)
    print(
        "selective images:",
        selective_images,
    )
    print(
        "gate call rate:",
        f"{call_rate:.6f}",
    )
    print(
        "calls saved:",
        f"{1.0 - call_rate:.6f}",
    )
    print(
        "categories:",
        len(inventory),
    )
    print(
        "unmatched:",
        int(
            inventory[
                "unmatched_images"
            ].sum()
        ),
    )
    print()

    if len(inventory) != 12:
        raise RuntimeError(
            "Expected 12 VisA categories."
        )

    if full_images < 2000:
        raise RuntimeError(
            "Unexpectedly few VisA images."
        )

    if not (
        0.35 <= call_rate <= 0.50
    ):
        raise RuntimeError(
            "Selective call rate is outside "
            "the expected range."
        )

    if args.validate_only:
        print(
            "[OK] Validation-only run passed."
        )
        return

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA device requested but "
            "CUDA is unavailable."
        )

    torch.set_float32_matmul_precision(
        "high"
    )

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    (
        model,
        preprocess,
        tokenizer,
        model_load_sec,
    ) = load_clip_model(
        module=module,
        device=args.device,
        clip_model=args.clip_model,
        clip_pretrained=(
            args.clip_pretrained
        ),
    )

    print(
        "model load time:",
        f"{model_load_sec:.3f} sec",
    )

    warm_up(
        module=module,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=args.device,
        category=module.VISA_CATEGORIES[0],
        top_k=args.top_k,
        crop_padding=args.crop_padding,
        min_crop_size=args.min_crop_size,
    )

    print("[OK] CUDA warm-up complete.")
    print()

    run_rows = []

    for repeat in range(
        1,
        args.repeats + 1,
    ):
        if repeat % 2 == 1:
            order = [
                (
                    "selective",
                    VIEW_ROOT,
                ),
                (
                    "full",
                    PATCHCORE_ROOT,
                ),
            ]
        else:
            order = [
                (
                    "full",
                    PATCHCORE_ROOT,
                ),
                (
                    "selective",
                    VIEW_ROOT,
                ),
            ]

        for mode, path in order:
            result = benchmark_mode(
                module=module,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=args.device,
                patchcore_root=path,
                mode=mode,
                repeat=repeat,
                top_k=args.top_k,
                crop_padding=(
                    args.crop_padding
                ),
                min_crop_size=(
                    args.min_crop_size
                ),
            )

            run_rows.append(result)

            print(
                f"[RESULT] {mode} "
                f"repeat={repeat} "
                f"time={result['wall_time_sec']:.3f}s "
                f"calls={result['actual_vlm_calls']} "
                f"crops={result['actual_crop_encodings']}"
            )
            print()

    runs = pd.DataFrame(run_rows)

    runs.to_csv(
        OUT_RUNS,
        index=False,
        lineterminator="\n",
    )

    full_times = (
        runs.loc[
            runs["mode"] == "full",
            "wall_time_sec",
        ]
        .astype(float)
        .tolist()
    )

    selective_times = (
        runs.loc[
            runs["mode"] == "selective",
            "wall_time_sec",
        ]
        .astype(float)
        .tolist()
    )

    full_median = float(
        statistics.median(full_times)
    )

    selective_median = float(
        statistics.median(
            selective_times
        )
    )

    speedup = (
        full_median / selective_median
    )

    wall_time_saved = (
        1.0
        - selective_median / full_median
    )

    summary = {
        "status": "success",
        "benchmark_scope": (
            "VisA PatchCore VLM reasoning "
            "stage after detector/candidate generation"
        ),
        "device": args.device,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if args.device.startswith("cuda")
            else "cpu"
        ),
        "clip_model": args.clip_model,
        "clip_pretrained": (
            args.clip_pretrained
        ),
        "strategy": "inspection_binary",
        "eval_mode": (
            "crop_topk_ensemble"
        ),
        "top_k": args.top_k,
        "repeats": args.repeats,
        "model_load_sec_excluded": (
            model_load_sec
        ),
        "full_images": full_images,
        "selective_images": (
            selective_images
        ),
        "actual_call_rate": call_rate,
        "actual_calls_saved_fraction": (
            1.0 - call_rate
        ),
        "full_median_wall_time_sec": (
            full_median
        ),
        "selective_median_wall_time_sec": (
            selective_median
        ),
        "wall_time_saved_fraction": (
            wall_time_saved
        ),
        "speedup": speedup,
        "full_times_sec": full_times,
        "selective_times_sec": (
            selective_times
        ),
        "peak_gpu_allocated_mib": float(
            runs[
                "peak_gpu_allocated_mib"
            ].max()
        ),
        "peak_gpu_reserved_mib": float(
            runs[
                "peak_gpu_reserved_mib"
            ].max()
        ),
        "performance_source": (
            "locked Stage 22-B1 "
            "cached-score evaluation"
        ),
        "caveat": (
            "This is a VLM-stage benchmark. "
            "Detector inference and candidate "
            "generation are excluded."
        ),
    }

    OUT_SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Stage 22-C2b: Actual Selective VLM Runtime",
        "",
        "## Scope",
        "",
        "- dataset: `VisA`",
        "- detector: `PatchCore`",
        "- reasoning strategy: `inspection_binary`",
        "- evaluation mode: `crop_topk_ensemble`",
        f"- GPU: `{summary['gpu_name']}`",
        f"- CLIP: `{args.clip_model} / {args.clip_pretrained}`",
        "- detector and candidate generation: `excluded`",
        "- model loading: `excluded from mode timing`",
        "",
        "## Actual invocation",
        "",
        f"- full VLM calls: `{full_images}`",
        f"- selective VLM calls: `{selective_images}`",
        f"- actual call rate: `{call_rate:.4f}`",
        f"- actual calls saved: `{1.0 - call_rate:.4f}`",
        "",
        "## Wall-clock result",
        "",
        f"- full median time: `{full_median:.3f} s`",
        f"- selective median time: `{selective_median:.3f} s`",
        f"- wall-clock reduction: `{wall_time_saved:.4f}`",
        f"- speedup: `{speedup:.3f}x`",
        "",
        "## GPU memory",
        "",
        f"- peak allocated: `{summary['peak_gpu_allocated_mib']:.1f} MiB`",
        f"- peak reserved: `{summary['peak_gpu_reserved_mib']:.1f} MiB`",
        "",
        "## Interpretation",
        "",
        "The benchmark executes the original Stage 7",
        "CLIP reasoning path. Samples rejected by the",
        "SRB pre-gate are removed before the VLM loop,",
        "so the reported call reduction is an actual",
        "execution-level reduction rather than an",
        "offline estimate.",
        "",
        "The benchmark covers only the VLM reasoning",
        "stage after detector inference and candidate",
        "generation.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(
        "===== STAGE 22-C2b SUCCESS ====="
    )
    print(
        "full VLM calls:",
        full_images,
    )
    print(
        "selective VLM calls:",
        selective_images,
    )
    print(
        "actual calls saved:",
        f"{1.0 - call_rate:.6f}",
    )
    print(
        "full median time:",
        f"{full_median:.3f}",
    )
    print(
        "selective median time:",
        f"{selective_median:.3f}",
    )
    print(
        "wall time saved:",
        f"{wall_time_saved:.6f}",
    )
    print(
        "speedup:",
        f"{speedup:.6f}x",
    )
    print(
        "peak GPU allocated MiB:",
        f"{summary['peak_gpu_allocated_mib']:.1f}",
    )
    print()
    print("[DONE]", OUT_INVENTORY)
    print("[DONE]", OUT_RUNS)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
