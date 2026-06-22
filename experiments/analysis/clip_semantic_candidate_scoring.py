import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import open_clip


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_clip_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    model.eval()

    print(f"[INFO] Loaded CLIP model: {args.clip_model}")
    print(f"[INFO] Pretrained: {args.clip_pretrained}")
    print(f"[INFO] Device: {device}")

    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device):
    tokens = tokenizer(texts).to(device)
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


@torch.no_grad()
def encode_image_crop(model, preprocess, crop, device):
    image_tensor = preprocess(crop).unsqueeze(0).to(device)
    features = model.encode_image(image_tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


def load_prompt_groups(prompt_bank_csv, category):
    df = pd.read_csv(prompt_bank_csv)

    category_df = df[df["category"] == category]
    generic_df = df[df["category"] == "generic"]

    normal_prompts = []
    defect_prompts = []

    normal_prompts.extend(
        category_df[category_df["prompt_type"] == "normal"]["prompt"].dropna().tolist()
    )
    defect_prompts.extend(
        category_df[category_df["prompt_type"] == "defect"]["prompt"].dropna().tolist()
    )

    normal_prompts.extend(
        generic_df[generic_df["prompt_type"].str.startswith("negative_", na=False)]["prompt"].dropna().tolist()
    )
    defect_prompts.extend(
        generic_df[generic_df["prompt_type"].str.startswith("defect_", na=False)]["prompt"].dropna().tolist()
    )

    normal_prompts = list(dict.fromkeys(normal_prompts))
    defect_prompts = list(dict.fromkeys(defect_prompts))

    if len(normal_prompts) == 0:
        raise ValueError(f"No normal prompts found for category: {category}")

    if len(defect_prompts) == 0:
        raise ValueError(f"No defect prompts found for category: {category}")

    return normal_prompts, defect_prompts


def read_resized_image(image_path, image_size):
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return image


def crop_candidate(image, row, padding=4):
    width, height = image.size

    x1 = int(float(row["x1"])) - padding
    y1 = int(float(row["y1"])) - padding
    x2 = int(float(row["x2"])) + padding
    y2 = int(float(row["y2"])) + padding

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return image.crop((x1, y1, x2 + 1, y2 + 1))


def minmax_norm(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def score_category(args, model, preprocess, tokenizer, device, category):
    candidate_csv = (
        Path(args.candidate_root)
        / "MVTecAD"
        / category
        / "candidate_regions"
        / "candidate_regions.csv"
    )

    if not candidate_csv.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")

    normal_prompts, defect_prompts = load_prompt_groups(args.prompt_bank_csv, category)

    normal_text_features = encode_texts(model, tokenizer, normal_prompts, device)
    defect_text_features = encode_texts(model, tokenizer, defect_prompts, device)

    df = pd.read_csv(candidate_csv)

    for col in ["component_rank", "x1", "y1", "x2", "y2", "mean_score", "max_score", "area", "gt_iou", "gt_f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["component_rank"] > 0].copy()
    df = df.dropna(subset=["x1", "y1", "x2", "y2"])

    rows = []

    image_cache = {}

    for _, row in df.iterrows():
        image_path = str(row["image_path"])

        if image_path not in image_cache:
            image_cache[image_path] = read_resized_image(image_path, args.image_size)

        image = image_cache[image_path]
        crop = crop_candidate(image, row, padding=args.crop_padding)

        image_feature = encode_image_crop(model, preprocess, crop, device)

        normal_sims = (image_feature @ normal_text_features.T).squeeze(0)
        defect_sims = (image_feature @ defect_text_features.T).squeeze(0)

        max_normal = float(normal_sims.max().item())
        max_defect = float(defect_sims.max().item())
        semantic_score = max_defect - max_normal

        rows.append(
            {
                "category": category,
                "image_path": image_path,
                "component_rank": int(row["component_rank"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "area": float(row["area"]),
                "mean_score": float(row["mean_score"]),
                "max_score": float(row["max_score"]),
                "clip_max_normal": max_normal,
                "clip_max_defect": max_defect,
                "clip_semantic_score": semantic_score,
                "gt_iou": float(row["gt_iou"]) if not pd.isna(row["gt_iou"]) else np.nan,
                "gt_f1": float(row["gt_f1"]) if not pd.isna(row["gt_f1"]) else np.nan,
            }
        )

    out_df = pd.DataFrame(rows)

    if len(out_df) == 0:
        return {
            "category": category,
            "num_images": 0,
            "num_candidates": 0,
            "patchcore_top1_gt_iou": 0.0,
            "patchcore_top1_gt_f1": 0.0,
            "clip_rerank_gt_iou": 0.0,
            "clip_rerank_gt_f1": 0.0,
            "combined_rerank_gt_iou": 0.0,
            "combined_rerank_gt_f1": 0.0,
            "delta_combined_f1": 0.0,
            "result_csv": "",
        }

    # Normalize per category for combined ranking.
    out_df["clip_semantic_score_norm"] = minmax_norm(out_df["clip_semantic_score"].values)
    out_df["mean_score_norm"] = minmax_norm(out_df["mean_score"].values)
    out_df["max_score_norm"] = minmax_norm(out_df["max_score"].values)
    out_df["area_norm"] = minmax_norm(np.log1p(out_df["area"].values))

    out_df["combined_score"] = (
        args.semantic_weight * out_df["clip_semantic_score_norm"]
        + args.anomaly_weight * out_df["mean_score_norm"]
        + args.max_score_weight * out_df["max_score_norm"]
        + args.area_weight * out_df["area_norm"]
    )

    out_root = Path(args.output_root) / "MVTecAD" / category
    out_root.mkdir(parents=True, exist_ok=True)

    detail_csv = out_root / "clip_candidate_scores.csv"
    out_df.to_csv(detail_csv, index=False)

    def pick_mean(metric_col, selector_col=None, top1=False):
        selected = []
        for _, group in out_df.groupby("image_path"):
            if top1:
                row = group.sort_values("component_rank", ascending=True).iloc[0]
            else:
                row = group.sort_values(selector_col, ascending=False).iloc[0]
            selected.append(row[metric_col])
        return float(np.nanmean(selected)) if selected else 0.0

    patchcore_iou = pick_mean("gt_iou", top1=True)
    patchcore_f1 = pick_mean("gt_f1", top1=True)

    clip_iou = pick_mean("gt_iou", selector_col="clip_semantic_score")
    clip_f1 = pick_mean("gt_f1", selector_col="clip_semantic_score")

    combined_iou = pick_mean("gt_iou", selector_col="combined_score")
    combined_f1 = pick_mean("gt_f1", selector_col="combined_score")

    summary = {
        "category": category,
        "num_images": int(out_df["image_path"].nunique()),
        "num_candidates": int(len(out_df)),
        "patchcore_top1_gt_iou": patchcore_iou,
        "patchcore_top1_gt_f1": patchcore_f1,
        "clip_rerank_gt_iou": clip_iou,
        "clip_rerank_gt_f1": clip_f1,
        "combined_rerank_gt_iou": combined_iou,
        "combined_rerank_gt_f1": combined_f1,
        "delta_combined_f1": combined_f1 - patchcore_f1,
        "result_csv": str(detail_csv),
    }

    print(
        f"[DONE] {category}: "
        f"PatchCore F1={patchcore_f1:.4f}, "
        f"CLIP rerank F1={clip_f1:.4f}, "
        f"Combined rerank F1={combined_f1:.4f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_root", type=str, default="results/analysis/patchcore_candidate_regions")
    parser.add_argument("--prompt_bank_csv", type=str, default="results/analysis/semantic_prompt_bank/mvtec_prompt_bank.csv")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/clip_semantic_candidate_scoring")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_padding", type=int, default=4)
    parser.add_argument("--semantic_weight", type=float, default=0.45)
    parser.add_argument("--anomaly_weight", type=float, default=0.35)
    parser.add_argument("--max_score_weight", type=float, default=0.10)
    parser.add_argument("--area_weight", type=float, default=0.10)
    args = parser.parse_args()

    model, preprocess, tokenizer, device = load_clip_model(args)

    summaries = []
    for category in args.categories:
        summaries.append(score_category(args, model, preprocess, tokenizer, device, category))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = out_root / "clip_semantic_candidate_scoring_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)

    print(f"[DONE] Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
