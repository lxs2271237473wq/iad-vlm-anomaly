from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(".").resolve()

IN_REGIONS = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_regions.csv"
IN_STAGE11C_SUMMARY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_c_candidate_summary.csv"
IN_STAGE11B1_QUALITY = ROOT / "results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_CANDIDATES = OUT_DIR / "stage11_d_vlm_candidate_scores.csv"
OUT_IMAGES = OUT_DIR / "stage11_d_vlm_image_predictions.csv"
OUT_SUMMARY = OUT_DIR / "stage11_d_vlm_summary.csv"
OUT_REPORT = DOC_DIR / "stage11_d_vlm_full_vs_context_report.md"


CATEGORY_DISPLAY = {
    "fruit_jelly": "fruit jelly",
    "sheet_metal": "sheet metal",
    "vial": "vial",
    "walnuts": "walnuts",
}


def category_name(category: str) -> str:
    return CATEGORY_DISPLAY.get(category, category.replace("_", " "))


def normal_prompts(category: str) -> List[str]:
    name = category_name(category)
    return [
        f"a quality inspection image of a normal {name}",
        f"an industrial inspection photo of an intact {name}",
        f"a clean and defect-free {name}",
        f"a normal product image of a {name} with no defect",
    ]


def anomaly_prompts(category: str) -> List[str]:
    name = category_name(category)
    return [
        f"a quality inspection image of a defective {name}",
        f"an industrial inspection photo of a damaged {name}",
        f"a {name} with visible defects or anomalies",
        f"an abnormal product image of a {name} with defect",
    ]


def resolve_path(path_text: object) -> Path:
    text = "" if pd.isna(path_text) else str(path_text)

    if not text:
        return Path("")

    p = Path(text)

    if p.is_absolute():
        return p

    return ROOT / p


def to_binary_label(x: object) -> int:
    if pd.isna(x):
        return 0

    if isinstance(x, bool):
        return int(x)

    text = str(x).strip().lower()

    if text in {"1", "true", "yes", "anomaly", "abnormal", "defect", "bad", "tensor(1)"}:
        return 1

    if text in {"0", "false", "no", "normal", "good", "ok", "tensor(0)"}:
        return 0

    try:
        return int(float(text) > 0)
    except Exception:
        return 0


def to_float(x: object, default: float = 0.0) -> float:
    if pd.isna(x):
        return default

    text = str(x).strip()

    if text.startswith("tensor(") and text.endswith(")"):
        text = text[len("tensor("):-1]

    try:
        return float(text)
    except Exception:
        return default


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(s).rank(method="average").to_numpy()
    pos_rank_sum = ranks[y == 1].sum()

    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    n_pos = int((y == 1).sum())

    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]

    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)

    return float(precision[y_sorted == 1].sum() / n_pos)


def best_f1_accuracy(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")

    thresholds = np.unique(s)

    best_f1 = -1.0
    best_acc = -1.0
    best_thr = float(thresholds[0]) if len(thresholds) else 0.0

    for thr in thresholds:
        pred = (s >= thr).astype(int)

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else 2 * tp / denom
        acc = float((pred == y).mean())

        if f1 > best_f1 or (math.isclose(f1, best_f1) and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_thr = float(thr)

    return float(best_f1), float(best_acc), float(best_thr)


class ClipScorer:
    def __init__(self) -> None:
        self.backend = ""
        self.device = "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.processor = None
        self.text_feature_cache: Dict[str, Dict[str, object]] = {}

        self._init_backend()

    def _init_backend(self) -> None:
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")

            self.backend = "open_clip:ViT-B-32/openai"
            self.model = model.to(self.device).eval()
            self.preprocess = preprocess
            self.tokenizer = tokenizer

            return

        except Exception as open_clip_error:
            print("[WARN] open_clip backend unavailable:", repr(open_clip_error))

        try:
            from transformers import CLIPModel, CLIPProcessor

            model_name = "openai/clip-vit-base-patch32"

            self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.backend = "transformers:openai/clip-vit-base-patch32"

            return

        except Exception as transformers_error:
            raise RuntimeError(
                "No usable CLIP backend found. "
                "Install/cache open_clip_torch or transformers CLIP first. "
                f"transformers error={repr(transformers_error)}"
            )

    def _prepare_text_features(self, category: str) -> Dict[str, object]:
        import torch

        if category in self.text_feature_cache:
            return self.text_feature_cache[category]

        prompts = normal_prompts(category) + anomaly_prompts(category)
        n_normal = len(normal_prompts(category))

        with torch.no_grad():
            if self.backend.startswith("open_clip"):
                tokens = self.tokenizer(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
            else:
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            normal_feature = text_features[:n_normal].mean(dim=0, keepdim=True)
            anomaly_feature = text_features[n_normal:].mean(dim=0, keepdim=True)

            normal_feature = normal_feature / normal_feature.norm(dim=-1, keepdim=True)
            anomaly_feature = anomaly_feature / anomaly_feature.norm(dim=-1, keepdim=True)

        self.text_feature_cache[category] = {
            "normal": normal_feature,
            "anomaly": anomaly_feature,
        }

        return self.text_feature_cache[category]

    def score_image(self, image_path: Path, category: str) -> Dict[str, float]:
        import torch

        features = self._prepare_text_features(category)
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            if self.backend.startswith("open_clip"):
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                image_feature = self.model.encode_image(image_tensor)
            else:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_feature = self.model.get_image_features(**inputs)

            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            normal_sim = float((image_feature @ features["normal"].T).item())
            anomaly_sim = float((image_feature @ features["anomaly"].T).item())

        return {
            "normal_score": normal_sim,
            "anomaly_score": anomaly_sim,
            "vlm_margin": anomaly_sim - normal_sim,
        }


def validate_inputs(regions: pd.DataFrame) -> None:
    required = [
        "category",
        "image_path",
        "gt_label",
        "pred_score",
        "candidate_rank",
        "tight_crop_path",
        "context_1p50_crop_path",
    ]

    missing = [c for c in required if c not in regions.columns]

    if missing:
        raise ValueError(f"Stage 11-C candidate regions missing columns: {missing}")

    missing_paths = []

    check_cols = ["image_path", "tight_crop_path", "context_1p50_crop_path"]

    for col in check_cols:
        for p_text in regions[col].dropna().astype(str).unique():
            p = resolve_path(p_text)
            if not p.exists():
                missing_paths.append((col, p_text))

            if len(missing_paths) >= 20:
                break

        if len(missing_paths) >= 20:
            break

    if missing_paths:
        msg = "\n".join([f"{col}: {p}" for col, p in missing_paths[:20]])
        raise FileNotFoundError(f"Missing input image/crop files:\n{msg}")


def load_regions(categories: List[str]) -> pd.DataFrame:
    if not IN_REGIONS.exists():
        raise FileNotFoundError(f"Missing Stage 11-C regions CSV: {IN_REGIONS}")

    regions = pd.read_csv(IN_REGIONS)

    if categories:
        regions = regions[regions["category"].isin(categories)].copy()

    if regions.empty:
        raise RuntimeError("No Stage 11-C candidate region rows after category filtering.")

    regions["gt_binary"] = regions["gt_label"].map(to_binary_label)
    regions["candidate_rank"] = pd.to_numeric(regions["candidate_rank"], errors="coerce").fillna(999999).astype(int)
    regions["patchcore_score"] = regions["pred_score"].map(to_float)

    validate_inputs(regions)

    return regions


def score_regions(regions: pd.DataFrame, scorer: ClipScorer) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    cache: Dict[Tuple[str, str], Dict[str, float]] = {}

    total = len(regions)

    for idx, row in regions.reset_index(drop=True).iterrows():
        if idx % 50 == 0:
            print(f"[INFO] scoring candidate row {idx + 1}/{total}")

        category = str(row["category"])

        tight_path = resolve_path(row["tight_crop_path"])
        context_path = resolve_path(row["context_1p50_crop_path"])

        tight_key = (category, str(tight_path))
        context_key = (category, str(context_path))

        if tight_key not in cache:
            cache[tight_key] = scorer.score_image(tight_path, category)

        if context_key not in cache:
            cache[context_key] = scorer.score_image(context_path, category)

        tight_scores = cache[tight_key]
        context_scores = cache[context_key]

        out = row.to_dict()

        out.update({
            "clip_backend": scorer.backend,

            "tight_normal_score": tight_scores["normal_score"],
            "tight_anomaly_score": tight_scores["anomaly_score"],
            "tight_vlm_margin": tight_scores["vlm_margin"],

            "context_normal_score": context_scores["normal_score"],
            "context_anomaly_score": context_scores["anomaly_score"],
            "context_vlm_margin": context_scores["vlm_margin"],
        })

        rows.append(out)

    return pd.DataFrame(rows)


def build_image_predictions(candidate_scores: pd.DataFrame, scorer: ClipScorer) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    full_cache: Dict[Tuple[str, str], Dict[str, float]] = {}

    grouped = candidate_scores.groupby(["category", "image_path"], dropna=False, sort=False)

    total = len(grouped)

    for idx, ((category, image_path), part) in enumerate(grouped):
        if idx % 25 == 0:
            print(f"[INFO] building image prediction {idx + 1}/{total}")

        part = part.copy().sort_values("candidate_rank")

        image_key = (str(category), str(resolve_path(image_path)))

        if image_key not in full_cache:
            full_cache[image_key] = scorer.score_image(resolve_path(image_path), str(category))

        full_scores = full_cache[image_key]

        top1 = part.iloc[0]

        tight_scores = pd.to_numeric(part["tight_vlm_margin"], errors="coerce")
        context_scores = pd.to_numeric(part["context_vlm_margin"], errors="coerce")

        tight_max_idx = tight_scores.idxmax()
        context_max_idx = context_scores.idxmax()

        tight_max = part.loc[tight_max_idx]
        context_max = part.loc[context_max_idx]

        rows.append({
            "dataset": "MVTec AD 2",
            "category": category,
            "image_path": image_path,
            "gt_binary": int(top1["gt_binary"]),
            "num_candidates": int(len(part)),
            "clip_backend": scorer.backend,

            "full_image_score": full_scores["vlm_margin"],
            "full_image_normal_score": full_scores["normal_score"],
            "full_image_anomaly_score": full_scores["anomaly_score"],

            "tight_top1_score": float(top1["tight_vlm_margin"]),
            "tight_topk_max_score": float(tight_scores.max()),
            "tight_topk_mean_score": float(tight_scores.mean()),
            "tight_best_crop_path": tight_max["tight_crop_path"],

            "context_top1_score": float(top1["context_vlm_margin"]),
            "context_topk_max_score": float(context_scores.max()),
            "context_topk_mean_score": float(context_scores.mean()),
            "context_best_crop_path": context_max["context_1p50_crop_path"],

            "patchcore_score": float(top1["patchcore_score"]),
        })

    return pd.DataFrame(rows)


def metric_row(df: pd.DataFrame, category: str, method: str, score_col: str) -> Dict[str, object]:
    y = df["gt_binary"].astype(int).to_numpy()
    s = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).to_numpy()

    best_f1, best_acc, best_thr = best_f1_accuracy(y, s)

    return {
        "dataset": "MVTec AD 2",
        "category": category,
        "method": method,
        "score_column": score_col,
        "num_images": int(len(df)),
        "num_normal": int((y == 0).sum()),
        "num_anomaly": int((y == 1).sum()),
        "auroc": binary_auroc(y, s),
        "ap": average_precision(y, s),
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "best_threshold": best_thr,
    }


def summarize(image_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    score_specs = [
        ("full_image", "full_image_score"),
        ("tight_crop_top1", "tight_top1_score"),
        ("tight_crop_topk_max", "tight_topk_max_score"),
        ("tight_crop_topk_mean", "tight_topk_mean_score"),
        ("context_1.50_top1", "context_top1_score"),
        ("context_1.50_topk_max", "context_topk_max_score"),
        ("context_1.50_topk_mean", "context_topk_mean_score"),
        ("patchcore_score", "patchcore_score"),
    ]

    categories = sorted(image_predictions["category"].unique().tolist())

    for category in categories:
        cdf = image_predictions[image_predictions["category"] == category].copy()

        for method, score_col in score_specs:
            rows.append(metric_row(cdf, category, method, score_col))

    all_df = image_predictions.copy()

    for method, score_col in score_specs:
        rows.append(metric_row(all_df, "ALL_PRIMARY", method, score_col))

    summary = pd.DataFrame(rows)

    base_rows = summary[summary["method"] == "full_image"][
        ["category", "auroc", "ap", "best_f1", "best_accuracy"]
    ].rename(
        columns={
            "auroc": "full_auroc",
            "ap": "full_ap",
            "best_f1": "full_best_f1",
            "best_accuracy": "full_best_accuracy",
        }
    )

    summary = summary.merge(base_rows, on="category", how="left")

    summary["delta_auroc_vs_full"] = summary["auroc"] - summary["full_auroc"]
    summary["delta_ap_vs_full"] = summary["ap"] - summary["full_ap"]
    summary["delta_best_f1_vs_full"] = summary["best_f1"] - summary["full_best_f1"]
    summary["delta_accuracy_vs_full"] = summary["best_accuracy"] - summary["full_best_accuracy"]

    return summary.sort_values(["category", "auroc"], ascending=[True, False])


def load_stage11c_summary() -> pd.DataFrame:
    if IN_STAGE11C_SUMMARY.exists():
        return pd.read_csv(IN_STAGE11C_SUMMARY)
    return pd.DataFrame()


def load_quality() -> pd.DataFrame:
    if IN_STAGE11B1_QUALITY.exists():
        return pd.read_csv(IN_STAGE11B1_QUALITY)
    return pd.DataFrame()


def write_report(
    candidate_scores: pd.DataFrame,
    image_predictions: pd.DataFrame,
    summary: pd.DataFrame,
    categories: List[str],
) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    backend = image_predictions["clip_backend"].iloc[0] if len(image_predictions) else "unknown"

    csummary = load_stage11c_summary()
    quality = load_quality()

    lines: List[str] = []

    lines.append("# Stage 11-D MVTec AD 2 Full-image vs Context-aware Crop VLM Reasoning")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage evaluates whether localization-guided context-aware crops improve VLM anomaly reasoning over full-image prompting on the selected MVTec AD 2 primary categories.")
    lines.append("")
    lines.append("This step does not rerun PatchCore and does not regenerate candidate crops. It reads Stage 11-C candidate regions and scores full images, tight crops, and context-aware crops using a CLIP-style VLM backend.")
    lines.append("")
    lines.append("## 2. Selected Categories")
    lines.append("")
    for c in categories:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## 3. Inputs")
    lines.append("")
    lines.append(f"- Candidate regions: `{IN_REGIONS.relative_to(ROOT)}`")
    lines.append(f"- Candidate summary: `{IN_STAGE11C_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Detector quality analysis: `{IN_STAGE11B1_QUALITY.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 4. VLM Backend")
    lines.append("")
    lines.append(f"- Backend: `{backend}`")
    lines.append("- Scoring: anomaly prompt similarity minus normal prompt similarity")
    lines.append("- Prompt type: category-aware normal/anomaly prompts")
    lines.append("")
    lines.append("## 5. Evaluated Methods")
    lines.append("")
    lines.append("| Method | Meaning |")
    lines.append("|---|---|")
    lines.append("| full_image | VLM score on the original full inspection image |")
    lines.append("| tight_crop_top1 | VLM score on rank-1 tight PatchCore candidate crop |")
    lines.append("| tight_crop_topk_max | maximum VLM score over tight candidate crops |")
    lines.append("| tight_crop_topk_mean | mean VLM score over tight candidate crops |")
    lines.append("| context_1.50_top1 | VLM score on rank-1 context crop expanded from the candidate box |")
    lines.append("| context_1.50_topk_max | maximum VLM score over context crops |")
    lines.append("| context_1.50_topk_mean | mean VLM score over context crops |")
    lines.append("| patchcore_score | detector score reference, not a VLM reasoning method |")
    lines.append("")
    lines.append("## 6. Output Files")
    lines.append("")
    lines.append(f"- Candidate scores: `{OUT_CANDIDATES.relative_to(ROOT)}`")
    lines.append(f"- Image predictions: `{OUT_IMAGES.relative_to(ROOT)}`")
    lines.append(f"- Summary: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 7. Stage 11-C Candidate Quality Reference")
    lines.append("")

    if csummary.empty:
        lines.append("Stage 11-C candidate summary is unavailable.")
    else:
        lines.append("| Category | Images | Candidate rows | Coverage | Top1 tight GT coverage | Top1 context GT coverage |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in csummary.sort_values("category").iterrows():
            lines.append(
                f"| {r['category']} | {int(r['num_images'])} | {int(r['num_candidate_rows'])} | "
                f"{float(r['candidate_coverage']):.4f} | "
                f"{float(r['top1_tight_mean_gt_coverage_anomaly']):.4f} | "
                f"{float(r['top1_context_mean_gt_coverage_anomaly']):.4f} |"
            )

    lines.append("")
    lines.append("## 8. VLM Reasoning Summary")
    lines.append("")
    lines.append("| Category | Method | Images | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for _, r in summary.sort_values(["category", "auroc"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {r['category']} | {r['method']} | {int(r['num_images'])} | "
            f"{float(r['auroc']):.4f} | {float(r['ap']):.4f} | "
            f"{float(r['best_f1']):.4f} | {float(r['best_accuracy']):.4f} | "
            f"{float(r['delta_auroc_vs_full']):.4f} |"
        )

    lines.append("")
    lines.append("## 9. Category-level Decision")
    lines.append("")
    lines.append("| Category | Best VLM method | Best VLM AUROC | Full-image AUROC | ΔAUROC | Decision |")
    lines.append("|---|---|---:|---:|---:|---|")

    vlm_methods = summary[summary["method"] != "patchcore_score"].copy()

    for category, part in vlm_methods.groupby("category"):
        part = part.sort_values("auroc", ascending=False)
        best = part.iloc[0]
        full = summary[(summary["category"] == category) & (summary["method"] == "full_image")].iloc[0]

        delta = float(best["auroc"]) - float(full["auroc"])

        if str(best["method"]).startswith("context") and delta > 0:
            decision = "positive_context_evidence"
        elif str(best["method"]).startswith("tight") and delta > 0:
            decision = "positive_crop_evidence_but_context_not_best"
        elif delta > 0:
            decision = "positive_non_full_evidence"
        else:
            decision = "full_image_stronger_or_tie"

        lines.append(
            f"| {category} | {best['method']} | {float(best['auroc']):.4f} | "
            f"{float(full['auroc']):.4f} | {delta:.4f} | {decision} |"
        )

    lines.append("")
    lines.append("## 10. Interpretation")
    lines.append("")
    lines.append("The main question is not whether PatchCore alone is strong, but whether PatchCore localization can serve as a useful visual bridge for VLM reasoning.")
    lines.append("A positive result is strongest when context-aware crops outperform full-image prompting, because this supports the Stage 10-G conclusion that object context is necessary for crop-based VLM anomaly reasoning.")
    lines.append("")
    lines.append("Detector-risk categories are intentionally excluded from this stage because weak localization would make crop-based VLM results difficult to interpret.")
    lines.append("")
    lines.append("## 11. Next Step")
    lines.append("")
    lines.append("Stage 11-E should consolidate Stage 11-B, 11-C, and 11-D into a paper-ready multi-category evidence table and decide whether to include secondary category fabric.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def parse_categories(text: str) -> List[str]:
    if not text:
        return []

    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        default="fruit_jelly,sheet_metal,vial,walnuts",
        help="Comma-separated category list.",
    )
    args = parser.parse_args()

    categories = parse_categories(args.categories)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Load candidate regions:", IN_REGIONS)
    regions = load_regions(categories)

    print("[INFO] categories:", sorted(regions["category"].unique().tolist()))
    print("[INFO] candidate rows:", len(regions))
    print("[INFO] images:", regions[["category", "image_path"]].drop_duplicates().shape[0])

    print("[INFO] Initialize VLM scorer")
    scorer = ClipScorer()
    print("[INFO] backend:", scorer.backend)

    print("[INFO] Score tight/context candidate crops")
    candidate_scores = score_regions(regions, scorer)

    print("[INFO] Build image-level full/crop predictions")
    image_predictions = build_image_predictions(candidate_scores, scorer)

    print("[INFO] Summarize")
    summary = summarize(image_predictions)

    candidate_scores.to_csv(OUT_CANDIDATES, index=False)
    image_predictions.to_csv(OUT_IMAGES, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    write_report(candidate_scores, image_predictions, summary, categories)

    print("[DONE]", OUT_CANDIDATES)
    print("[DONE]", OUT_IMAGES)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)

    print("")
    print("===== summary =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
