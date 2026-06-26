import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import open_clip


VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


OBJECT_NAMES = {
    "candle": "candle",
    "capsules": "capsules",
    "cashew": "cashew nut",
    "chewinggum": "chewing gum",
    "fryum": "fryum snack",
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe-shaped fryum snack",
}


def canonical_path(path):
    s = str(path).replace("\\", "/")
    marker = "datasets/VisA_anomalib_1cls/"
    if marker in s:
        return s[s.index(marker):]
    marker = "datasets/VisA/"
    if marker in s:
        return s[s.index(marker):]
    return s


def build_prompts(category, strategy):
    obj = OBJECT_NAMES.get(category, category)

    if strategy == "generic_binary":
        normal = [
            "a normal industrial product",
            "a defect-free industrial object",
            "a clean normal object",
        ]
        anomaly = [
            "a defective industrial product",
            "an anomalous industrial object",
            "a damaged or contaminated object",
        ]
        return normal, anomaly

    if strategy == "category_binary":
        normal = [
            f"a normal {obj}",
            f"a defect-free {obj}",
            f"a clean undamaged {obj}",
        ]
        anomaly = [
            f"a defective {obj}",
            f"an anomalous {obj}",
            f"a damaged or contaminated {obj}",
        ]
        return normal, anomaly

    if strategy == "inspection_binary":
        normal = [
            f"a quality inspection image of a normal {obj}",
            f"an industrial inspection photo of a defect-free {obj}",
            f"a close-up inspection image showing a normal {obj}",
        ]
        anomaly = [
            f"a quality inspection image of a defective {obj}",
            f"an industrial inspection photo showing an anomaly on a {obj}",
            f"a close-up inspection image showing damage, contamination, or abnormal texture on a {obj}",
        ]
        return normal, anomaly

    raise ValueError(f"Unknown prompt strategy: {strategy}")


def encode_prompt_set(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    feature = features.mean(dim=0, keepdim=True)
    feature = feature / feature.norm(dim=-1, keepdim=True)
    return feature


def build_text_features(model, tokenizer, category, strategy, device):
    normal_prompts, anomaly_prompts = build_prompts(category, strategy)

    normal_feature = encode_prompt_set(model, tokenizer, normal_prompts, device)
    anomaly_feature = encode_prompt_set(model, tokenizer, anomaly_prompts, device)

    text_features = torch.cat([normal_feature, anomaly_feature], dim=0)

    prompt_row = {
        "category": category,
        "strategy": strategy,
        "normal_prompts": " || ".join(normal_prompts),
        "anomaly_prompts": " || ".join(anomaly_prompts),
    }

    return text_features, prompt_row


def load_candidate_boxes(candidate_root, category, top_k):
    csv_path = Path(candidate_root) / "VisA" / category / "candidate_regions.csv"

    if not csv_path.exists():
        print(f"[WARN] Missing candidate CSV: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    for col in ["component_rank", "x1", "y1", "x2", "y2", "area", "mean_score", "max_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[pd.to_numeric(df["component_rank"], errors="coerce") > 0].copy()
    boxes = {}

    for image_path, group in df.groupby("image_path"):
        key = canonical_path(image_path)
        group = group.sort_values("component_rank").head(top_k)

        box_list = []
        for _, row in group.iterrows():
            box_list.append(
                {
                    "x1": int(row["x1"]),
                    "y1": int(row["y1"]),
                    "x2": int(row["x2"]),
                    "y2": int(row["y2"]),
                    "rank": int(row["component_rank"]),
                    "area": float(row["area"]),
                    "mean_score": float(row["mean_score"]) if pd.notna(row["mean_score"]) else 0.0,
                    "max_score": float(row["max_score"]) if pd.notna(row["max_score"]) else 0.0,
                }
            )

        if box_list:
            boxes[key] = box_list

    return boxes


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


def get_eval_images(row, boxes, eval_mode, args):
    image = Image.open(row["image_path"]).convert("RGB")
    key = canonical_path(row["image_path"])
    box_list = boxes.get(key, [])

    if eval_mode == "full_all":
        return [image], "full", False

    if eval_mode == "crop_or_full":
        if not box_list:
            return [image], "fallback_full", True
        return [
            crop_candidate(
                image=image,
                box=box_list[0],
                map_size=args.map_size,
                crop_padding=args.crop_padding,
                min_crop_size=args.min_crop_size,
            )
        ], "crop_top1", False

    if eval_mode == "crop_topk_ensemble":
        if not box_list:
            return [image], "fallback_full", True

        crops = [
            crop_candidate(
                image=image,
                box=box,
                map_size=args.map_size,
                crop_padding=args.crop_padding,
                min_crop_size=args.min_crop_size,
            )
            for box in box_list[: args.top_k]
        ]
        return crops, f"crop_top{len(crops)}_ensemble", False

    raise ValueError(f"Unknown eval mode: {eval_mode}")


def encode_images(model, preprocess, images, device):
    batch = torch.cat([preprocess(img).unsqueeze(0) for img in images], dim=0).to(device)

    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)

    return features


def safe_auroc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, scores))


def best_f1_threshold(y_true, scores, max_thresholds=256):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0

    unique_scores = np.unique(scores)
    if len(unique_scores) > max_thresholds:
        thresholds = np.quantile(scores, np.linspace(0, 1, max_thresholds))
        thresholds = np.unique(thresholds)
    else:
        thresholds = unique_scores

    best_thr = float(thresholds[0])
    best_f1 = -1.0
    best_acc = 0.0

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = float((pred == y_true).mean())
        if f1 > best_f1:
            best_f1 = float(f1)
            best_acc = acc
            best_thr = float(thr)

    return best_thr, best_f1, best_acc


def evaluate_category(args, model, preprocess, tokenizer, device, category, strategy, eval_mode):
    pred_csv = Path(args.candidate_root) / "VisA" / category / f"{args.backbone_model}_image_predictions.csv"

    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing PatchCore image prediction CSV: {pred_csv}")

    df = pd.read_csv(pred_csv)
    df = df[df["label"].isin(["normal", "anomaly"])].copy().reset_index(drop=True)

    boxes = load_candidate_boxes(args.candidate_root, category, args.top_k)
    text_features, prompt_row = build_text_features(
        model=model,
        tokenizer=tokenizer,
        category=category,
        strategy=strategy,
        device=device,
    )

    y_true = []
    anomaly_scores = []
    detail_rows = []
    fallback_count = 0
    covered_count = 0

    for _, row in df.iterrows():
        eval_images, used_mode, fallback = get_eval_images(row, boxes, eval_mode, args)

        if fallback:
            fallback_count += 1
        else:
            covered_count += 1

        image_features = encode_images(model, preprocess, eval_images, device)
        sims_matrix = (image_features @ text_features.T).detach().cpu().numpy()

        # text index 0 = normal, 1 = anomaly.
        # for top-k crops, use max anomaly margin over crops.
        margins = sims_matrix[:, 1] - sims_matrix[:, 0]
        anomaly_score = float(np.max(margins))

        true = int(row["is_anomaly"])
        y_true.append(true)
        anomaly_scores.append(anomaly_score)

        detail_rows.append(
            {
                "dataset": "VisA",
                "category": category,
                "strategy": strategy,
                "eval_mode": eval_mode,
                "used_mode": used_mode,
                "image_path": row["image_path"],
                "canonical_image_path": canonical_path(row["image_path"]),
                "label": row["label"],
                "is_anomaly": true,
                "vlm_anomaly_score": anomaly_score,
                "fallback": int(fallback),
                "num_eval_images": len(eval_images),
            }
        )

    y_true_np = np.asarray(y_true).astype(int)
    scores_np = np.asarray(anomaly_scores).astype(float)

    thr, best_f1, best_acc = best_f1_threshold(y_true_np, scores_np)

    summary_row = {
        "dataset": "VisA",
        "category": category,
        "strategy": strategy,
        "eval_mode": eval_mode,
        "num_images_total": len(df),
        "num_images_used": len(df),
        "num_normal": int((y_true_np == 0).sum()),
        "num_anomaly": int((y_true_np == 1).sum()),
        "covered_count": int(covered_count),
        "fallback_count": int(fallback_count),
        "coverage_ratio": float(covered_count / len(df)) if len(df) else 0.0,
        "auroc": safe_auroc(y_true_np, scores_np),
        "ap": safe_ap(y_true_np, scores_np),
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "best_threshold": thr,
    }

    return summary_row, detail_rows, prompt_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--candidate_root",
    type=str,
    default="results/stage7_generalization/visa_multibackbone/fastflow_12cls",
    )
    parser.add_argument(
    "--backbone_model",
    type=str,
    default="fastflow",
    )
    parser.add_argument("--output_root", type=str, default="results/stage7_generalization/visa_binary_prompt_reasoning")
    parser.add_argument("--categories", nargs="+", default=VISA_CATEGORIES)
    parser.add_argument("--strategies", nargs="+", default=["generic_binary", "category_binary", "inspection_binary"])
    parser.add_argument("--eval_modes", nargs="+", default=["full_all", "crop_or_full", "crop_topk_ensemble"])
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--map_size", type=int, default=224)
    parser.add_argument("--crop_padding", type=int, default=12)
    parser.add_argument("--min_crop_size", type=int, default=48)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading CLIP: {args.clip_model}, pretrained={args.clip_pretrained}, device={device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.eval()

    summary_rows = []
    detail_rows = []
    prompt_rows = []

    for strategy in args.strategies:
        for eval_mode in args.eval_modes:
            for category in args.categories:
                print(f"[INFO] Evaluating category={category}, strategy={strategy}, eval_mode={eval_mode}")

                summary, details, prompt = evaluate_category(
                    args=args,
                    model=model,
                    preprocess=preprocess,
                    tokenizer=tokenizer,
                    device=device,
                    category=category,
                    strategy=strategy,
                    eval_mode=eval_mode,
                )

                summary_rows.append(summary)
                detail_rows.extend(details)
                prompt_rows.append(prompt)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    prompt_df = pd.DataFrame(prompt_rows).drop_duplicates()

    mean_rows = []

    for (strategy, eval_mode), group in summary_df.groupby(["strategy", "eval_mode"]):
        total = int(group["num_images_total"].sum())
        covered = int(group["covered_count"].sum())
        fallback = int(group["fallback_count"].sum())

        mean_rows.append(
            {
                "dataset": "VisA",
                "category": "MEAN",
                "strategy": strategy,
                "eval_mode": eval_mode,
                "num_images_total": total,
                "num_images_used": int(group["num_images_used"].sum()),
                "num_normal": int(group["num_normal"].sum()),
                "num_anomaly": int(group["num_anomaly"].sum()),
                "covered_count": covered,
                "fallback_count": fallback,
                "coverage_ratio": covered / total if total else 0.0,
                "auroc": float(group["auroc"].mean()),
                "ap": float(group["ap"].mean()),
                "best_f1": float(group["best_f1"].mean()),
                "best_accuracy": float(group["best_accuracy"].mean()),
                "best_threshold": "",
            }
        )

    mean_df = pd.DataFrame(mean_rows).sort_values(["best_f1", "auroc"], ascending=[False, False])
    full_summary_df = pd.concat([summary_df, mean_df], ignore_index=True)

    predictions_csv = output_root / "visa_binary_prompt_predictions.csv"
    summary_csv = output_root / "visa_binary_prompt_summary.csv"
    mean_csv = output_root / "visa_binary_prompt_mean_summary.csv"
    prompts_csv = output_root / "visa_binary_prompt_bank.csv"

    detail_df.to_csv(predictions_csv, index=False)
    full_summary_df.to_csv(summary_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    prompt_df.to_csv(prompts_csv, index=False)

    print("\n========== VisA Binary Prompt Mean Summary ==========")
    print(mean_df.to_string(index=False))

    print(f"\n[DONE] Predictions saved to: {predictions_csv}")
    print(f"[DONE] Summary saved to: {summary_csv}")
    print(f"[DONE] Mean summary saved to: {mean_csv}")
    print(f"[DONE] Prompt bank saved to: {prompts_csv}")


if __name__ == "__main__":
    main()
