from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path("/root/private_data/iad-vlm-anomaly").resolve()
REPO = Path("/root/private_data/third_party/AnomalyCLIP").resolve()
DATA_ROOT = Path("/root/private_data/anomalyclip_data/ad2four").resolve()

OUT_DIR = PROJECT / "results/stage20_anomalyclip_baseline"
DOC_DIR = PROJECT / "docs/stage20_anomalyclip_baseline"

OUT_JSON = OUT_DIR / "stage20_c3_single_image_smoke.json"
OUT_REPORT = DOC_DIR / "stage20_c3_single_image_smoke_report.md"


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return scalar_int(value[0])
    return int(value)


def scalar_str(value) -> str:
    if isinstance(value, (list, tuple)):
        return scalar_str(value[0])
    return str(value)


def resolve_checkpoint(user_value: str | None) -> Path:
    if user_value:
        p = Path(user_value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    checkpoint_root = REPO / "checkpoints"

    candidates = []
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        candidates.extend(checkpoint_root.rglob(pattern))

    candidates = sorted(
        {p.resolve() for p in candidates if p.is_file()},
        key=lambda p: (-p.stat().st_size, str(p)),
    )

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found under "
            f"{checkpoint_root}. Supply --checkpoint explicitly."
        )

    print("[CHECKPOINT CANDIDATES]")
    for i, p in enumerate(candidates):
        print(
            f"  {i}: {p} "
            f"({p.stat().st_size / 1024**2:.2f} MiB)"
        )

    selected = candidates[0]
    print()
    print("[SMOKE CHECKPOINT]", selected)
    print(
        "[NOTE] Automatically selected only for smoke testing. "
        "Final evaluation checkpoint must be fixed explicitly."
    )

    return selected


def load_first_anomalous_sample(dataset):
    for index in range(len(dataset)):
        sample = dataset[index]
        anomaly = scalar_int(sample["anomaly"])

        if anomaly == 1:
            return index, sample

    raise RuntimeError("No anomalous sample found in AD2-four test set.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    if not REPO.exists():
        raise FileNotFoundError(REPO)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(DATA_ROOT)

    sys.path.insert(0, str(REPO))

    import AnomalyCLIP_lib
    from dataset import Dataset
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    setup_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda:0")
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    class TransformArgs:
        image_size = args.image_size

    transform, target_transform = get_transform(TransformArgs())

    dataset = Dataset(
        root=str(DATA_ROOT),
        transform=transform,
        target_transform=target_transform,
        dataset_name="ad2four",
        mode="test",
    )

    sample_index, sample = load_first_anomalous_sample(dataset)

    image = sample["img"]
    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    cls_name = scalar_str(sample["cls_name"])
    anomaly_label = scalar_int(sample["anomaly"])
    img_path = scalar_str(sample["img_path"])

    design_details = {
        "Prompt_length": 12,
        "learnabel_text_embedding_depth": 9,
        "learnabel_text_embedding_length": 4,
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()

    print("[LOAD] base AnomalyCLIP model")
    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=design_details,
    )
    model.eval()

    print("[LOAD] prompt checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Unexpected checkpoint type: {type(checkpoint).__name__}"
        )

    print("[CHECKPOINT KEYS]", list(checkpoint.keys()))

    prompt_state = checkpoint.get("prompt_learner")
    if prompt_state is None:
        raise KeyError(
            "Checkpoint does not contain key 'prompt_learner'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    prompt_learner = AnomalyCLIP_PromptLearner(
        model.to("cpu"),
        design_details,
    )
    prompt_learner.load_state_dict(prompt_state)
    prompt_learner.to(device)
    prompt_learner.eval()

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = (
            prompt_learner(cls_id=None)
        )

        text_features = model.encode_text_learn(
            prompts,
            tokenized_prompts,
            compound_prompts_text,
        ).float()

        text_features = torch.stack(
            torch.chunk(text_features, dim=0, chunks=2),
            dim=1,
        )

        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True,
        )

        image_features, patch_features = model.encode_image(
            image,
            [6, 12, 18, 24],
            DPAM_layer=20,
        )

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True,
        )

        text_logits = (
            image_features
            @ text_features.permute(0, 2, 1)
        )

        text_probs = (text_logits / 0.07).softmax(dim=-1)
        anomaly_score = float(
            text_probs[:, 0, 1].detach().cpu().item()
        )

        anomaly_maps = []

        for patch_feature in patch_features:
            patch_feature = patch_feature / patch_feature.norm(
                dim=-1,
                keepdim=True,
            )

            similarity, _ = AnomalyCLIP_lib.compute_similarity(
                patch_feature,
                text_features[0],
            )

            similarity_map = AnomalyCLIP_lib.get_similarity_map(
                similarity[:, 1:, :],
                args.image_size,
            )

            anomaly_map = (
                similarity_map[..., 1]
                + 1
                - similarity_map[..., 0]
            ) / 2.0

            anomaly_maps.append(anomaly_map)

        if not anomaly_maps:
            raise RuntimeError("No anomaly map was generated.")

        anomaly_map = torch.stack(anomaly_maps).sum(dim=0)

    if not torch.isfinite(torch.tensor(anomaly_score)):
        raise RuntimeError(f"Non-finite anomaly score: {anomaly_score}")

    if not torch.isfinite(anomaly_map).all():
        raise RuntimeError("Anomaly map contains NaN or Inf.")

    elapsed = time.perf_counter() - started

    result = {
        "status": "success",
        "sample_index": sample_index,
        "image_path": img_path,
        "category": cls_name,
        "ground_truth_anomaly": anomaly_label,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_mb": round(
            checkpoint_path.stat().st_size / 1024**2,
            4,
        ),
        "image_shape": list(image.shape),
        "image_anomaly_score": anomaly_score,
        "num_patch_feature_maps": len(patch_features),
        "anomaly_map_shape": list(anomaly_map.shape),
        "anomaly_map_min": float(
            anomaly_map.min().detach().cpu().item()
        ),
        "anomaly_map_max": float(
            anomaly_map.max().detach().cpu().item()
        ),
        "anomaly_map_mean": float(
            anomaly_map.mean().detach().cpu().item()
        ),
        "elapsed_sec": elapsed,
        "gpu_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }

    OUT_JSON.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Stage 20-C3: AnomalyCLIP Single-image Smoke Test",
        "",
        "## Status",
        "",
        "- status: `success`",
        f"- category: `{result['category']}`",
        f"- ground-truth anomaly: `{result['ground_truth_anomaly']}`",
        f"- image: `{result['image_path']}`",
        f"- checkpoint: `{result['checkpoint_path']}`",
        "",
        "## Outputs",
        "",
        f"- image anomaly score: `{result['image_anomaly_score']:.8f}`",
        f"- image tensor shape: `{result['image_shape']}`",
        f"- number of patch feature maps: `{result['num_patch_feature_maps']}`",
        f"- anomaly-map shape: `{result['anomaly_map_shape']}`",
        f"- anomaly-map minimum: `{result['anomaly_map_min']:.8f}`",
        f"- anomaly-map maximum: `{result['anomaly_map_max']:.8f}`",
        f"- anomaly-map mean: `{result['anomaly_map_mean']:.8f}`",
        "",
        "## Runtime",
        "",
        f"- elapsed seconds: `{result['elapsed_sec']:.3f}`",
        f"- peak allocated VRAM: `{result['gpu_peak_allocated_mb']:.2f} MiB`",
        f"- GPU: `{result['gpu']}`",
        "",
        "## Interpretation",
        "",
        "This smoke test confirms that the official model, prompt checkpoint,",
        "AD2-four Dataset adapter, image-level score, and pixel-level anomaly",
        "map can run together. It is not a final AnomalyCLIP benchmark result.",
        "",
        "## Next step",
        "",
        "Fix the official cross-dataset checkpoint explicitly and run the",
        "complete 484-image AD2-four evaluation.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("===== SMOKE TEST SUCCESS =====")
    print(json.dumps(result, indent=2))
    print()
    print("[DONE]", OUT_JSON)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
