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


def load_image(path):
    image = Image.open(path).convert("RGB")
    return image


def encode_texts(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def encode_image(model, preprocess, image_path, device):
    image = load_image(image_path)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features


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


def evaluate_strategy(df, model, preprocess, tokenizer, device, strategy, output_root):
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

        for _, row in tqdm(
            category_df.iterrows(),
            total=len(category_df),
            desc=f"{strategy}:{category}",
        ):
            image_path = row["image_path"]
            true_defect = row["defect_type"]

            image_features = encode_image(model, preprocess, image_path, device)
            sims = (image_features @ text_features.T).squeeze(0).detach().cpu().numpy()

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
                "category": category,
                "image_path": image_path,
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

        summary_rows.append(
            {
                "strategy": strategy,
                "category": category,
                "num_images": len(category_df),
                "num_defect_types": len(defect_types),
                "top1_accuracy": accuracy,
                "top2_accuracy": top2_accuracy,
                "macro_f1": macro_f1,
            }
        )

        for r in per_defect_metrics(y_true, y_pred, defect_types):
            r.update({"strategy": strategy, "category": category})
            defect_rows.append(r)

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    defect_df = pd.DataFrame(defect_rows)

    mean_row = {
        "strategy": strategy,
        "category": "MEAN",
        "num_images": summary_df["num_images"].sum(),
        "num_defect_types": summary_df["num_defect_types"].sum(),
        "top1_accuracy": summary_df["top1_accuracy"].mean(),
        "top2_accuracy": summary_df["top2_accuracy"].mean(),
        "macro_f1": summary_df["macro_f1"].mean(),
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    output_root.mkdir(parents=True, exist_ok=True)

    detail_csv = output_root / f"{strategy}_defect_type_predictions.csv"
    summary_csv = output_root / f"{strategy}_defect_type_summary.csv"
    defect_csv = output_root / f"{strategy}_per_defect_metrics.csv"

    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    defect_df.to_csv(defect_csv, index=False)

    return summary_df, detail_csv, summary_csv, defect_csv


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
        default="results/analysis/defect_type_prompt_reasoning",
    )
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["generic", "category_aware", "manufacturing_aware"],
    )
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv)
    output_root = Path(args.output_root)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_csv}")

    df = pd.read_csv(manifest_csv)

    required_cols = [
        "category",
        "defect_type",
        "image_path",
        "generic_prompt",
        "category_aware_prompt",
        "manufacturing_aware_prompt",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading CLIP: model={args.clip_model}, pretrained={args.clip_pretrained}, device={device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.eval()

    all_summary = []

    for strategy in args.strategies:
        if strategy not in PROMPT_COLUMNS:
            raise ValueError(f"Unknown strategy: {strategy}")

        summary_df, detail_csv, summary_csv, defect_csv = evaluate_strategy(
            df=df,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
            strategy=strategy,
            output_root=output_root,
        )

        all_summary.append(summary_df)

        print(f"[DONE] {strategy}")
        print(f"  detail:  {detail_csv}")
        print(f"  summary: {summary_csv}")
        print(f"  defect:  {defect_csv}")

    all_summary_df = pd.concat(all_summary, ignore_index=True)

    combined_summary_csv = output_root / "defect_type_prompt_reasoning_summary.csv"
    all_summary_df.to_csv(combined_summary_csv, index=False)

    mean_df = all_summary_df[all_summary_df["category"] == "MEAN"].copy()
    mean_df = mean_df.sort_values("top1_accuracy", ascending=False)

    mean_csv = output_root / "defect_type_prompt_reasoning_mean_summary.csv"
    mean_df.to_csv(mean_csv, index=False)

    print("\n========== Mean Summary ==========")
    print(mean_df.to_string(index=False))

    print(f"\n[DONE] Combined summary saved to: {combined_summary_csv}")
    print(f"[DONE] Mean summary saved to: {mean_csv}")


if __name__ == "__main__":
    main()
