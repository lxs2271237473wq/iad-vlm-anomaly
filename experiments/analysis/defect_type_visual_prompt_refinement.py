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


def infer_gt_mask_path(image_path):
    image_path = Path(image_path)
    parts = list(image_path.parts)

    if "test" not in parts:
        return None

    test_idx = parts.index("test")
    parts[test_idx] = "ground_truth"

    parts[-1] = f"{image_path.stem}_mask{image_path.suffix}"

    return Path(*parts)


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_by_gt_mask(image, mask_path, padding=12, min_crop_size=48):
    mask_img = Image.open(mask_path).convert("L")

    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.NEAREST)

    mask = np.array(mask_img) > 0
    bbox = mask_bbox(mask)

    if bbox is None:
        return image, "fallback_full_empty_mask"

    x1, y1, x2, y2 = bbox

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    w, h = image.size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1

    if crop_w < min_crop_size:
        pad = (min_crop_size - crop_w) // 2 + 1
        x1 = max(0, x1 - pad)
        x2 = min(w - 1, x2 + pad)

    if crop_h < min_crop_size:
        pad = (min_crop_size - crop_h) // 2 + 1
        y1 = max(0, y1 - pad)
        y2 = min(h - 1, y2 + pad)

    return image.crop((x1, y1, x2 + 1, y2 + 1)), "gt_crop"


def load_eval_image(row, image_mode, args):
    image = Image.open(row["image_path"]).convert("RGB")

    if image_mode == "full":
        return image, "full"

    if image_mode == "gt_crop":
        mask_path = infer_gt_mask_path(row["image_path"])

        if mask_path is None or not mask_path.exists():
            return image, "fallback_full_missing_mask"

        return crop_by_gt_mask(
            image=image,
            mask_path=mask_path,
            padding=args.crop_padding,
            min_crop_size=args.min_crop_size,
        )

    raise ValueError(f"Unknown image mode: {image_mode}")


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


def encode_prompt_set(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    feature = features.mean(dim=0, keepdim=True)
    feature = feature / feature.norm(dim=-1, keepdim=True)

    return feature


def encode_image(model, preprocess, image, device):
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model.encode_image(image_tensor)
        feature = feature / feature.norm(dim=-1, keepdim=True)

    return feature


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


def build_text_features(model, tokenizer, category, defect_types, strategy, device):
    features = []
    prompt_records = []

    for defect_type in defect_types:
        prompts = build_prompts(category, defect_type, strategy)
        feature = encode_prompt_set(model, tokenizer, prompts, device)
        features.append(feature)

        prompt_records.append(
            {
                "strategy": strategy,
                "category": category,
                "defect_type": defect_type,
                "num_prompts": len(prompts),
                "prompts": " || ".join(prompts),
            }
        )

    text_features = torch.cat(features, dim=0)

    return text_features, prompt_records


def evaluate_one(df, model, preprocess, tokenizer, device, strategy, image_mode, args):
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
        used_modes = []

        for _, row in tqdm(
            category_df.iterrows(),
            total=len(category_df),
            desc=f"{strategy}:{image_mode}:{category}",
        ):
            true_defect = row["defect_type"]
            image, used_mode = load_eval_image(row, image_mode, args)

            image_feature = encode_image(model, preprocess, image, device)
            sims = (image_feature @ text_features.T).squeeze(0).detach().cpu().numpy()

            order = np.argsort(-sims)
            pred_defect = defect_types[int(order[0])]
            top2_defects = [defect_types[int(i)] for i in order[:2]]

            y_true.append(true_defect)
            y_pred.append(pred_defect)
            top2_correct.append(true_defect in top2_defects)
            used_modes.append(used_mode)

            score_dict = {
                f"score_{defect_types[i]}": float(sims[i])
                for i in range(len(defect_types))
            }

            detail_row = {
                "strategy": strategy,
                "image_mode": image_mode,
                "used_mode": used_mode,
                "category": category,
                "image_path": row["image_path"],
                "true_defect_type": true_defect,
                "pred_defect_type": pred_defect,
                "top2_defect_types": "|".join(top2_defects),
                "top1_correct": pred_defect == true_defect,
                "top2_correct": true_defect in top2_defects,
            }
            detail_row.update(score_dict)
            detail_rows.append(detail_row)

        accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
        top2_accuracy = float(np.mean(top2_correct))
        macro_f1 = macro_f1_score(y_true, y_pred, defect_types)
        fallback_count = sum(m.startswith("fallback_full") for m in used_modes)

        summary_rows.append(
            {
                "strategy": strategy,
                "image_mode": image_mode,
                "category": category,
                "num_images": len(category_df),
                "num_defect_types": len(defect_types),
                "fallback_count": fallback_count,
                "top1_accuracy": accuracy,
                "top2_accuracy": top2_accuracy,
                "macro_f1": macro_f1,
            }
        )

        for r in per_defect_metrics(y_true, y_pred, defect_types):
            r.update(
                {
                    "strategy": strategy,
                    "image_mode": image_mode,
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
        "--output_root",
        type=str,
        default="results/analysis/defect_type_visual_prompt_refinement",
    )
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["generic_label", "short_visual", "category_visual", "visual_ensemble"],
    )
    parser.add_argument(
        "--image_modes",
        nargs="+",
        default=["full", "gt_crop"],
    )
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading CLIP: model={args.clip_model}, pretrained={args.clip_pretrained}, device={device}")

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
        for image_mode in args.image_modes:
            detail_rows, summary_rows, defect_rows, prompt_rows = evaluate_one(
                df=df,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                strategy=strategy,
                image_mode=image_mode,
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

    for (strategy, image_mode), group in summary_df.groupby(["strategy", "image_mode"]):
        mean_rows.append(
            {
                "strategy": strategy,
                "image_mode": image_mode,
                "category": "MEAN",
                "num_images": group["num_images"].sum(),
                "num_defect_types": group["num_defect_types"].sum(),
                "fallback_count": group["fallback_count"].sum(),
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

    detail_csv = output_root / "defect_type_visual_prompt_refinement_predictions.csv"
    summary_csv = output_root / "defect_type_visual_prompt_refinement_summary.csv"
    mean_csv = output_root / "defect_type_visual_prompt_refinement_mean_summary.csv"
    defect_csv = output_root / "defect_type_visual_prompt_refinement_per_defect_metrics.csv"
    prompt_csv = output_root / "visual_prompt_bank.csv"

    detail_df.to_csv(detail_csv, index=False)
    combined_summary_df.to_csv(summary_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    defect_df.to_csv(defect_csv, index=False)
    prompt_df.to_csv(prompt_csv, index=False)

    print("\n========== Mean Summary ==========")
    print(mean_df.to_string(index=False))

    print(f"\n[DONE] Predictions saved to: {detail_csv}")
    print(f"[DONE] Summary saved to: {summary_csv}")
    print(f"[DONE] Mean summary saved to: {mean_csv}")
    print(f"[DONE] Per-defect metrics saved to: {defect_csv}")
    print(f"[DONE] Visual prompt bank saved to: {prompt_csv}")


if __name__ == "__main__":
    main()
