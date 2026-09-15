import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip


PROMPT_COLUMNS = {
    "generic": "generic_prompt",
    "category_aware": "category_aware_prompt",
    "manufacturing_aware": "manufacturing_aware_prompt",
}


def canonical_path(path):
    s = str(path).replace("\\", "/")
    marker = "datasets/MVTecAD/"
    if marker in s:
        return s[s.index(marker):]
    return s


def load_image(path):
    return Image.open(path).convert("RGB")


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


def encode_texts(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def encode_pil_image(model, preprocess, image, device):
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features


def build_candidate_prompts(df, category, prompt_col):
    sub = df[df["category"] == category].copy()

    candidate_rows = (
        sub[["defect_type", prompt_col]]
        .drop_duplicates()
        .sort_values("defect_type")
        .reset_index(drop=True)
    )

    defect_types = candidate_rows["defect_type"].tolist()
    prompts = candidate_rows[prompt_col].tolist()

    return defect_types, prompts


def load_candidate_boxes(candidate_root, categories):
    all_boxes = {}

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

        for col in ["component_rank", "x1", "y1", "x2", "y2", "mean_score", "area"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Use PatchCore top-1 component by default.
        df = df[df["component_rank"] == 1].copy()

        for _, row in df.iterrows():
            key = canonical_path(row["image_path"])
            all_boxes[key] = {
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "mean_score": float(row["mean_score"]),
                "area": float(row["area"]),
            }

    return all_boxes


def crop_top1_candidate(image, box, map_size=224, crop_padding=8, min_crop_size=32):
    # Candidate boxes come from PatchCore anomaly map coordinates.
    # Resize original image to the same map coordinate system before cropping.
    resized = image.resize((map_size, map_size), Image.BILINEAR)

    x1 = int(box["x1"]) - crop_padding
    y1 = int(box["y1"]) - crop_padding
    x2 = int(box["x2"]) + crop_padding
    y2 = int(box["y2"]) + crop_padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(map_size - 1, x2)
    y2 = min(map_size - 1, y2)

    w = x2 - x1 + 1
    h = y2 - y1 + 1

    if w < min_crop_size:
        pad = (min_crop_size - w) // 2 + 1
        x1 = max(0, x1 - pad)
        x2 = min(map_size - 1, x2 + pad)

    if h < min_crop_size:
        pad = (min_crop_size - h) // 2 + 1
        y1 = max(0, y1 - pad)
        y2 = min(map_size - 1, y2 + pad)

    crop = resized.crop((x1, y1, x2 + 1, y2 + 1))
    return crop


def get_eval_image(row, image_mode, candidate_boxes, args):
    image = load_image(row["image_path"])

    if image_mode == "full":
        return image, "full"

    if image_mode == "crop_top1":
        key = canonical_path(row["image_path"])
        box = candidate_boxes.get(key)

        if box is None:
            return image, "fallback_full"

        crop = crop_top1_candidate(
            image=image,
            box=box,
            map_size=args.map_size,
            crop_padding=args.crop_padding,
            min_crop_size=args.min_crop_size,
        )
        return crop, "crop_top1"

    raise ValueError(f"Unknown image mode: {image_mode}")


def evaluate_strategy_and_mode(
    df,
    model,
    preprocess,
    tokenizer,
    device,
    strategy,
    image_mode,
    candidate_boxes,
    args,
):
    prompt_col = PROMPT_COLUMNS[strategy]

    detail_rows = []
    summary_rows = []
    defect_rows = []

    for category in sorted(df["category"].unique()):
        category_df = df[df["category"] == category].copy().reset_index(drop=True)

        defect_types, prompts = build_candidate_prompts(df, category, prompt_col)
        text_features = encode_texts(model, tokenizer, prompts, device)

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
            eval_image, used_mode = get_eval_image(row, image_mode, candidate_boxes, args)

            image_features = encode_pil_image(model, preprocess, eval_image, device)
            sims = (image_features @ text_features.T).squeeze(0).detach().cpu().numpy()

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
        fallback_count = sum(m == "fallback_full" for m in used_modes)

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

    return detail_rows, summary_rows, defect_rows


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
        default="results/analysis/defect_type_prompt_reasoning_crop",
    )
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["generic", "category_aware", "manufacturing_aware"],
    )
    parser.add_argument(
        "--image_modes",
        nargs="+",
        default=["full", "crop_top1"],
    )
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

    candidate_boxes = load_candidate_boxes(args.candidate_root, categories)

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

    for strategy in args.strategies:
        if strategy not in PROMPT_COLUMNS:
            raise ValueError(f"Unknown strategy: {strategy}")

        for image_mode in args.image_modes:
            detail_rows, summary_rows, defect_rows = evaluate_strategy_and_mode(
                df=df,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                strategy=strategy,
                image_mode=image_mode,
                candidate_boxes=candidate_boxes,
                args=args,
            )

            all_detail_rows.extend(detail_rows)
            all_summary_rows.extend(summary_rows)
            all_defect_rows.extend(defect_rows)

    detail_df = pd.DataFrame(all_detail_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    defect_df = pd.DataFrame(all_defect_rows)

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

    detail_csv = output_root / "defect_type_prompt_reasoning_crop_predictions.csv"
    summary_csv = output_root / "defect_type_prompt_reasoning_crop_summary.csv"
    mean_csv = output_root / "defect_type_prompt_reasoning_crop_mean_summary.csv"
    defect_csv = output_root / "defect_type_prompt_reasoning_crop_per_defect_metrics.csv"

    detail_df.to_csv(detail_csv, index=False)
    combined_summary_df.to_csv(summary_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    defect_df.to_csv(defect_csv, index=False)

    print("\n========== Mean Summary ==========")
    print(mean_df.to_string(index=False))

    print(f"\n[DONE] Predictions saved to: {detail_csv}")
    print(f"[DONE] Summary saved to: {summary_csv}")
    print(f"[DONE] Mean summary saved to: {mean_csv}")
    print(f"[DONE] Per-defect metrics saved to: {defect_csv}")


if __name__ == "__main__":
    main()
