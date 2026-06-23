import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip


OBJECT_INFO = {
    "grid": {
        "object_type": "grid texture surface",
        "surface": "regular repeated grid pattern",
    },
    "screw": {
        "object_type": "metal screw",
        "surface": "metal threaded surface",
    },
    "leather": {
        "object_type": "leather surface",
        "surface": "soft textured leather surface",
    },
    "wood": {
        "object_type": "wood surface",
        "surface": "natural wood grain surface",
    },
}


DEFECT_VISUAL_MAP = {
    "bent": "a bent or deformed local structure",
    "broken": "a broken or missing local structure",
    "glue": "a glue-like contamination or sticky residue",
    "metal_contamination": "a metallic foreign contamination spot",
    "thread": "an irregular thread-like defect or line pattern interruption",

    "color": "a local abnormal color change or stain",
    "cut": "a sharp cut mark on the surface",
    "fold": "a folded or wrinkled surface region",
    "poke": "a small puncture-like local damage",

    "manipulated_front": "a manipulated or deformed front surface",
    "scratch_head": "thin scratch marks on the screw head",
    "scratch_neck": "thin scratch marks on the screw neck",
    "thread_side": "abnormal damage along the side thread",
    "thread_top": "abnormal damage on the top thread",

    "combined": "multiple mixed defect patterns in one region",
    "hole": "a local hole or missing region",
    "liquid": "a liquid-like stain or wet contamination",
    "scratch": "a thin line-like scratch crossing the surface",
}


def canonical_path(path):
    s = str(path).replace("\\", "/")
    marker = "datasets/MVTecAD/"
    if marker in s:
        return s[s.index(marker):]
    return s


def defect_visual_description(defect_type):
    if defect_type in DEFECT_VISUAL_MAP:
        return DEFECT_VISUAL_MAP[defect_type]

    lower = defect_type.lower()

    if "scratch" in lower:
        return "thin line-like scratch marks"
    if "crack" in lower:
        return "a crack or split line on the surface"
    if "cut" in lower:
        return "a sharp local cut mark"
    if "color" in lower:
        return "a local abnormal color change"
    if "contamination" in lower:
        return "a foreign contamination spot"
    if "hole" in lower:
        return "a local hole or missing region"
    if "thread" in lower:
        return "an abnormal thread-like structure"

    return f"a visible {defect_type} defect"


def build_prompts(category, defect_type, strategy):
    info = OBJECT_INFO.get(
        category,
        {
            "object_type": category,
            "surface": "industrial object surface",
        },
    )

    object_type = info["object_type"]
    surface = info["surface"]
    visual_desc = defect_visual_description(defect_type)

    if strategy == "generic_label":
        return [
            f"a close-up photo of {defect_type} defect",
        ]

    if strategy == "short_visual":
        return [
            f"a close-up inspection image showing {visual_desc}",
        ]

    if strategy == "category_visual":
        return [
            f"a close-up crop of a defective {object_type} showing {visual_desc}",
        ]

    if strategy == "visual_ensemble":
        return [
            f"a close-up photo of {defect_type} defect",
            f"a close-up inspection image showing {visual_desc}",
            f"a local defect region with {visual_desc}",
            f"a defective {object_type} on {surface} showing {visual_desc}",
        ]

    raise ValueError(f"Unknown strategy: {strategy}")


def load_candidate_boxes(candidate_root, categories, top_k):
    boxes = {}
    coverage_rows = []

    for category in categories:
        csv_path = (
            Path(candidate_root)
            / "MVTecAD"
            / category
            / "candidate_regions"
            / "candidate_regions.csv"
        )

        if not csv_path.exists():
            print(f"[WARN] Missing candidate CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        for col in ["component_rank", "x1", "y1", "x2", "y2", "mean_score", "max_score", "area"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["component_rank"] > 0].copy()
        df = df.sort_values(["image_path", "component_rank"])

        for image_path, group in df.groupby("image_path"):
            key = canonical_path(image_path)
            group = group.sort_values("component_rank").head(top_k)

            box_list = []
            for _, row in group.iterrows():
                if pd.isna(row["x1"]) or pd.isna(row["y1"]) or pd.isna(row["x2"]) or pd.isna(row["y2"]):
                    continue

                box_list.append(
                    {
                        "x1": int(row["x1"]),
                        "y1": int(row["y1"]),
                        "x2": int(row["x2"]),
                        "y2": int(row["y2"]),
                        "rank": int(row["component_rank"]),
                        "mean_score": float(row["mean_score"]) if "mean_score" in row and pd.notna(row["mean_score"]) else 0.0,
                        "max_score": float(row["max_score"]) if "max_score" in row and pd.notna(row["max_score"]) else 0.0,
                        "area": float(row["area"]) if "area" in row and pd.notna(row["area"]) else 0.0,
                    }
                )

            if box_list:
                boxes[key] = box_list
                coverage_rows.append(
                    {
                        "category": category,
                        "image_path": key,
                        "num_candidate_boxes": len(box_list),
                    }
                )

    return boxes, pd.DataFrame(coverage_rows)


def crop_candidate(image, box, map_size=224, crop_padding=12, min_crop_size=48):
    resized = image.resize((map_size, map_size), Image.BILINEAR)

    x1 = int(box["x1"]) - crop_padding
    y1 = int(box["y1"]) - crop_padding
    x2 = int(box["x2"]) + crop_padding
    y2 = int(box["y2"]) + crop_padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(map_size - 1, x2)
    y2 = min(map_size - 1, y2)

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    if width < min_crop_size:
        pad = (min_crop_size - width) // 2 + 1
        x1 = max(0, x1 - pad)
        x2 = min(map_size - 1, x2 + pad)

    if height < min_crop_size:
        pad = (min_crop_size - height) // 2 + 1
        y1 = max(0, y1 - pad)
        y2 = min(map_size - 1, y2 + pad)

    return resized.crop((x1, y1, x2 + 1, y2 + 1))


def get_eval_images(row, eval_mode, candidate_boxes, args):
    image = Image.open(row["image_path"]).convert("RGB")
    key = canonical_path(row["image_path"])
    boxes = candidate_boxes.get(key, [])

    if eval_mode == "full_all":
        return [image], "full", False

    if eval_mode == "crop_or_full":
        if not boxes:
            return [image], "fallback_full", True
        return [
            crop_candidate(
                image=image,
                box=boxes[0],
                map_size=args.map_size,
                crop_padding=args.crop_padding,
                min_crop_size=args.min_crop_size,
            )
        ], "crop_top1", False

    if eval_mode == "crop_only":
        if not boxes:
            return [], "skipped_no_candidate", True
        return [
            crop_candidate(
                image=image,
                box=boxes[0],
                map_size=args.map_size,
                crop_padding=args.crop_padding,
                min_crop_size=args.min_crop_size,
            )
        ], "crop_top1", False

    if eval_mode == "crop_topk_ensemble":
        if not boxes:
            return [image], "fallback_full", True

        crops = []
        for box in boxes[: args.top_k]:
            crops.append(
                crop_candidate(
                    image=image,
                    box=box,
                    map_size=args.map_size,
                    crop_padding=args.crop_padding,
                    min_crop_size=args.min_crop_size,
                )
            )
        return crops, f"crop_top{len(crops)}_ensemble", False

    if eval_mode == "crop_topk_only":
        if not boxes:
            return [], "skipped_no_candidate", True

        crops = []
        for box in boxes[: args.top_k]:
            crops.append(
                crop_candidate(
                    image=image,
                    box=box,
                    map_size=args.map_size,
                    crop_padding=args.crop_padding,
                    min_crop_size=args.min_crop_size,
                )
            )
        return crops, f"crop_top{len(crops)}_ensemble", False

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def encode_prompt_set(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    feature = features.mean(dim=0, keepdim=True)
    feature = feature / feature.norm(dim=-1, keepdim=True)

    return feature


def build_text_features(model, tokenizer, category, defect_types, strategy, device):
    features = []
    prompt_rows = []

    for defect_type in defect_types:
        prompts = build_prompts(category, defect_type, strategy)
        feature = encode_prompt_set(model, tokenizer, prompts, device)
        features.append(feature)

        prompt_rows.append(
            {
                "strategy": strategy,
                "category": category,
                "defect_type": defect_type,
                "num_prompts": len(prompts),
                "prompts": " || ".join(prompts),
            }
        )

    return torch.cat(features, dim=0), prompt_rows


def encode_images(model, preprocess, images, device):
    tensors = [preprocess(image).unsqueeze(0) for image in images]
    batch = torch.cat(tensors, dim=0).to(device)

    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)

    return features


def macro_f1_score(y_true, y_pred, labels):
    f1s = []

    for label in labels:
        tp = sum((t == label and p == label) for t, p in zip(y_true, y_pred))
        fp = sum((t != label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum((t == label and p != label) for t, p in zip(y_true, y_pred))

        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)

    return float(np.mean(f1s)) if f1s else 0.0


def per_defect_metrics(y_true, y_pred, labels):
    rows = []

    for label in labels:
        tp = sum((t == label and p == label) for t, p in zip(y_true, y_pred))
        fp = sum((t != label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum((t == label and p != label) for t, p in zip(y_true, y_pred))
        support = sum(t == label for t in y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0

        rows.append(
            {
                "defect_type": label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return rows


def evaluate_one(df, model, preprocess, tokenizer, device, strategy, eval_mode, candidate_boxes, args):
    detail_rows = []
    summary_rows = []
    defect_rows = []
    prompt_rows = []

    for category in sorted(df["category"].unique()):
        category_df = df[df["category"] == category].copy().reset_index(drop=True)
        defect_types = sorted(category_df["defect_type"].unique().tolist())

        text_features, prompt_records = build_text_features(
            model=model,
            tokenizer=tokenizer,
            category=category,
            defect_types=defect_types,
            strategy=strategy,
            device=device,
        )
        prompt_rows.extend(prompt_records)

        y_true = []
        y_pred = []
        top2_correct = []
        fallback_count = 0
        skipped_count = 0
        covered_count = 0

        for _, row in tqdm(
            category_df.iterrows(),
            total=len(category_df),
            desc=f"{strategy}:{eval_mode}:{category}",
        ):
            eval_images, used_mode, is_fallback_or_skipped = get_eval_images(
                row=row,
                eval_mode=eval_mode,
                candidate_boxes=candidate_boxes,
                args=args,
            )

            if len(eval_images) == 0:
                skipped_count += 1
                continue

            if used_mode.startswith("fallback"):
                fallback_count += 1
            else:
                covered_count += 1

            true_defect = row["defect_type"]

            image_features = encode_images(model, preprocess, eval_images, device)
            sims_matrix = (image_features @ text_features.T).detach().cpu().numpy()

            # For top-k crops: use max similarity over crops for each candidate defect.
            sims = sims_matrix.max(axis=0)

            order = np.argsort(-sims)
            pred_defect = defect_types[int(order[0])]
            top2_defects = [defect_types[int(i)] for i in order[:2]]

            y_true.append(true_defect)
            y_pred.append(pred_defect)
            top2_correct.append(true_defect in top2_defects)

            score_dict = {
                f"score_{defect_types[i]}": float(sims[i])
                for i in range(len(defect_types))
            }

            detail_row = {
                "strategy": strategy,
                "eval_mode": eval_mode,
                "used_mode": used_mode,
                "category": category,
                "image_path": row["image_path"],
                "true_defect_type": true_defect,
                "pred_defect_type": pred_defect,
                "top2_defect_types": "|".join(top2_defects),
                "top1_correct": pred_defect == true_defect,
                "top2_correct": true_defect in top2_defects,
                "num_eval_crops": len(eval_images),
            }
            detail_row.update(score_dict)
            detail_rows.append(detail_row)

        if len(y_true) == 0:
            accuracy = 0.0
            top2_accuracy = 0.0
            macro_f1 = 0.0
        else:
            accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
            top2_accuracy = float(np.mean(top2_correct))
            macro_f1 = macro_f1_score(y_true, y_pred, defect_types)

        summary_rows.append(
            {
                "strategy": strategy,
                "eval_mode": eval_mode,
                "category": category,
                "num_images_total": len(category_df),
                "num_images_used": len(y_true),
                "num_defect_types": len(defect_types),
                "covered_count": covered_count,
                "fallback_count": fallback_count,
                "skipped_count": skipped_count,
                "coverage_ratio": covered_count / len(category_df) if len(category_df) else 0.0,
                "top1_accuracy": accuracy,
                "top2_accuracy": top2_accuracy,
                "macro_f1": macro_f1,
            }
        )

        for r in per_defect_metrics(y_true, y_pred, defect_types):
            r.update(
                {
                    "strategy": strategy,
                    "eval_mode": eval_mode,
                    "category": category,
                }
            )
            defect_rows.append(r)

    return detail_rows, summary_rows, defect_rows, prompt_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default="results/analysis/defect_type_reasoning/mvtec_defect_type_reasoning_manifest.csv",
    )
    parser.add_argument(
        "--candidate_root",
        type=str,
        default="results/analysis/patchcore_candidate_regions",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/analysis/real_anomaly_crop_visual_prompt_reasoning",
    )
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["generic_label", "short_visual", "category_visual", "visual_ensemble"],
    )
    parser.add_argument(
        "--eval_modes",
        nargs="+",
        default=["full_all", "crop_or_full", "crop_only", "crop_topk_ensemble", "crop_topk_only"],
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--map_size", type=int, default=224)
    parser.add_argument("--crop_padding", type=int, default=12)
    parser.add_argument("--min_crop_size", type=int, default=48)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    categories = sorted(df["category"].unique())

    candidate_boxes, candidate_coverage_df = load_candidate_boxes(
        candidate_root=args.candidate_root,
        categories=categories,
        top_k=args.top_k,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading CLIP: model={args.clip_model}, pretrained={args.clip_pretrained}, device={device}")
    print(f"[INFO] Candidate images with boxes: {len(candidate_boxes)} / {len(df)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.eval()

    all_detail_rows = []
    all_summary_rows = []
    all_defect_rows = []
    all_prompt_rows = []

    for strategy in args.strategies:
        for eval_mode in args.eval_modes:
            detail_rows, summary_rows, defect_rows, prompt_rows = evaluate_one(
                df=df,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                strategy=strategy,
                eval_mode=eval_mode,
                candidate_boxes=candidate_boxes,
                args=args,
            )

            all_detail_rows.extend(detail_rows)
            all_summary_rows.extend(summary_rows)
            all_defect_rows.extend(defect_rows)
            all_prompt_rows.extend(prompt_rows)

    detail_df = pd.DataFrame(all_detail_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    defect_df = pd.DataFrame(all_defect_rows)
    prompt_df = pd.DataFrame(all_prompt_rows).drop_duplicates()

    mean_rows = []

    for (strategy, eval_mode), group in summary_df.groupby(["strategy", "eval_mode"]):
        total_images = group["num_images_total"].sum()
        used_images = group["num_images_used"].sum()
        covered = group["covered_count"].sum()

        mean_rows.append(
            {
                "strategy": strategy,
                "eval_mode": eval_mode,
                "category": "MEAN",
                "num_images_total": total_images,
                "num_images_used": used_images,
                "num_defect_types": group["num_defect_types"].sum(),
                "covered_count": covered,
                "fallback_count": group["fallback_count"].sum(),
                "skipped_count": group["skipped_count"].sum(),
                "coverage_ratio": covered / total_images if total_images else 0.0,
                "top1_accuracy": group["top1_accuracy"].mean(),
                "top2_accuracy": group["top2_accuracy"].mean(),
                "macro_f1": group["macro_f1"].mean(),
            }
        )

    mean_df = pd.DataFrame(mean_rows).sort_values(
        ["top1_accuracy", "macro_f1"],
        ascending=[False, False],
    )

    combined_summary_df = pd.concat([summary_df, mean_df], ignore_index=True)

    detail_csv = output_root / "real_anomaly_crop_visual_prompt_predictions.csv"
    summary_csv = output_root / "real_anomaly_crop_visual_prompt_summary.csv"
    mean_csv = output_root / "real_anomaly_crop_visual_prompt_mean_summary.csv"
    defect_csv = output_root / "real_anomaly_crop_visual_prompt_per_defect_metrics.csv"
    prompt_csv = output_root / "real_anomaly_crop_visual_prompt_bank.csv"
    coverage_csv = output_root / "patchcore_candidate_coverage.csv"

    detail_df.to_csv(detail_csv, index=False)
    combined_summary_df.to_csv(summary_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    defect_df.to_csv(defect_csv, index=False)
    prompt_df.to_csv(prompt_csv, index=False)
    candidate_coverage_df.to_csv(coverage_csv, index=False)

    print("\n========== Mean Summary ==========")
    print(mean_df.to_string(index=False))

    print(f"\n[DONE] Predictions saved to: {detail_csv}")
    print(f"[DONE] Summary saved to: {summary_csv}")
    print(f"[DONE] Mean summary saved to: {mean_csv}")
    print(f"[DONE] Per-defect metrics saved to: {defect_csv}")
    print(f"[DONE] Prompt bank saved to: {prompt_csv}")
    print(f"[DONE] Candidate coverage saved to: {coverage_csv}")


if __name__ == "__main__":
    main()
