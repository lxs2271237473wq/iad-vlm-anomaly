from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(".").resolve()

IN_REGIONS = ROOT / "results" / "stage10_dataset_expansion" / "stage10_d_patchcore_candidate_regions.csv"

OUT_DIR = ROOT / "results" / "stage10_dataset_expansion"
DOC_DIR = ROOT / "docs" / "stage10_dataset_expansion"

OUT_CANDIDATES = OUT_DIR / "stage10_e_vlm_candidate_scores.csv"
OUT_IMAGES = OUT_DIR / "stage10_e_vlm_image_predictions.csv"
OUT_SUMMARY = OUT_DIR / "stage10_e_vlm_summary.csv"
OUT_REPORT = DOC_DIR / "stage10_e_vlm_full_vs_crop_report.md"


NORMAL_PROMPTS = [
    "a quality inspection image of a normal vial",
    "an industrial inspection photo of an intact vial",
    "a clean and defect-free vial surface",
    "a normal product image of a vial with no defect",
]

ANOMALY_PROMPTS = [
    "a quality inspection image of a defective vial",
    "an industrial inspection photo of a damaged vial",
    "a vial with visible defects or anomalies",
    "an abnormal product image of a vial with defect",
]


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


def resolve_path(path_text: object) -> Path:
    text = "" if pd.isna(path_text) else str(path_text)

    if not text:
        return Path("")

    p = Path(text)

    if p.is_absolute():
        return p

    return ROOT / p


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
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


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
    ap = precision[y_sorted == 1].sum() / n_pos
    return float(ap)


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
        self.text_features = None

        self._init_backend()
        self._prepare_text_features()

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
                "No usable CLIP backend found. Install/cache open_clip_torch or transformers CLIP first. "
                f"transformers error={repr(transformers_error)}"
            )

    def _prepare_text_features(self) -> None:
        import torch

        prompts = NORMAL_PROMPTS + ANOMALY_PROMPTS

        with torch.no_grad():
            if self.backend.startswith("open_clip"):
                tokens = self.tokenizer(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
            else:
                inputs = self.processor(
                    text=prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                text_features = self.model.get_text_features(**inputs)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            n_normal = len(NORMAL_PROMPTS)
            normal_feature = text_features[:n_normal].mean(dim=0, keepdim=True)
            anomaly_feature = text_features[n_normal:].mean(dim=0, keepdim=True)

            normal_feature = normal_feature / normal_feature.norm(dim=-1, keepdim=True)
            anomaly_feature = anomaly_feature / anomaly_feature.norm(dim=-1, keepdim=True)

            self.text_features = {
                "normal": normal_feature,
                "anomaly": anomaly_feature,
            }

    def score_image(self, image_path: Path) -> Dict[str, float]:
        import torch

        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            if self.backend.startswith("open_clip"):
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                image_feature = self.model.encode_image(image_tensor)
            else:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_feature = self.model.get_image_features(**inputs)

            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            normal_sim = float((image_feature @ self.text_features["normal"].T).item())
            anomaly_sim = float((image_feature @ self.text_features["anomaly"].T).item())

        return {
            "normal_score": normal_sim,
            "anomaly_score": anomaly_sim,
            "vlm_anomaly_margin": anomaly_sim - normal_sim,
        }


def load_regions() -> pd.DataFrame:
    if not IN_REGIONS.exists():
        raise FileNotFoundError(f"Missing Stage 10-D regions CSV: {IN_REGIONS}")

    df = pd.read_csv(IN_REGIONS)

    required = [
        "image_path",
        "gt_label",
        "candidate_rank",
        "candidate_available",
        "crop_path",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Candidate CSV missing columns: {missing}")

    df = df[df["candidate_available"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    if df.empty:
        raise RuntimeError("No candidate rows available for VLM scoring.")

    df["gt_binary"] = df["gt_label"].map(to_binary_label)
    df["image_path_resolved"] = df["image_path"].map(lambda x: str(resolve_path(x)))
    df["crop_path_resolved"] = df["crop_path"].map(lambda x: str(resolve_path(x)))

    return df


def score_candidates(df: pd.DataFrame, scorer: ClipScorer) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    full_cache: Dict[str, Dict[str, float]] = {}
    crop_cache: Dict[str, Dict[str, float]] = {}

    for _, row in df.iterrows():
        image_path = resolve_path(row["image_path"])
        crop_path = resolve_path(row["crop_path"])

        if str(image_path) not in full_cache:
            full_cache[str(image_path)] = scorer.score_image(image_path)

        if str(crop_path) not in crop_cache:
            crop_cache[str(crop_path)] = scorer.score_image(crop_path)

        full_scores = full_cache[str(image_path)]
        crop_scores = crop_cache[str(crop_path)]

        out = row.to_dict()
        out.update({
            "clip_backend": scorer.backend,
            "full_normal_score": full_scores["normal_score"],
            "full_anomaly_score": full_scores["anomaly_score"],
            "full_vlm_margin": full_scores["vlm_anomaly_margin"],
            "crop_normal_score": crop_scores["normal_score"],
            "crop_anomaly_score": crop_scores["anomaly_score"],
            "crop_vlm_margin": crop_scores["vlm_anomaly_margin"],
        })
        rows.append(out)

    return pd.DataFrame(rows)


def build_image_predictions(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for image_path, part in candidate_scores.groupby("image_path", dropna=False, sort=False):
        part = part.copy()
        part["candidate_rank"] = pd.to_numeric(part["candidate_rank"], errors="coerce")

        part_ranked = part.sort_values("candidate_rank")
        top1 = part_ranked.iloc[0]

        crop_topk_max_idx = pd.to_numeric(part["crop_vlm_margin"], errors="coerce").idxmax()
        crop_topk_max_row = part.loc[crop_topk_max_idx]

        rows.append({
            "dataset": "MVTec AD 2",
            "category": "vial",
            "image_path": image_path,
            "gt_binary": int(top1["gt_binary"]),
            "num_candidates": int(len(part)),
            "clip_backend": top1["clip_backend"],

            "full_image_score": float(top1["full_vlm_margin"]),

            "crop_top1_score": float(top1["crop_vlm_margin"]),
            "crop_top1_path": top1["crop_path"],

            "crop_topk_max_score": float(crop_topk_max_row["crop_vlm_margin"]),
            "crop_topk_max_path": crop_topk_max_row["crop_path"],

            "crop_topk_mean_score": float(pd.to_numeric(part["crop_vlm_margin"], errors="coerce").mean()),

            "patchcore_pred_score": float(pd.to_numeric(part["pred_score"], errors="coerce").iloc[0])
            if "pred_score" in part.columns
            else float("nan"),
        })

    return pd.DataFrame(rows)


def summarize_predictions(image_pred: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "full_image": "full_image_score",
        "crop_top1": "crop_top1_score",
        "crop_topk_max": "crop_topk_max_score",
        "crop_topk_mean": "crop_topk_mean_score",
        "patchcore_score": "patchcore_pred_score",
    }

    rows = []

    y = image_pred["gt_binary"].astype(int).to_numpy()

    for method, score_col in methods.items():
        if score_col not in image_pred.columns:
            continue

        s = pd.to_numeric(image_pred[score_col], errors="coerce").fillna(0.0).to_numpy()
        f1, acc, thr = best_f1_accuracy(y, s)

        rows.append({
            "dataset": "MVTec AD 2",
            "category": "vial",
            "method": method,
            "num_images": int(len(image_pred)),
            "num_normal": int((y == 0).sum()),
            "num_anomaly": int((y == 1).sum()),
            "auroc": binary_auroc(y, s),
            "ap": average_precision(y, s),
            "best_f1": f1,
            "best_accuracy": acc,
            "best_threshold": thr,
        })

    summary = pd.DataFrame(rows)

    base = summary[summary["method"] == "full_image"][["auroc", "ap", "best_f1", "best_accuracy"]].iloc[0]

    summary["delta_auroc_vs_full"] = summary["auroc"] - float(base["auroc"])
    summary["delta_ap_vs_full"] = summary["ap"] - float(base["ap"])
    summary["delta_best_f1_vs_full"] = summary["best_f1"] - float(base["best_f1"])
    summary["delta_accuracy_vs_full"] = summary["best_accuracy"] - float(base["best_accuracy"])

    return summary.sort_values("auroc", ascending=False)


def write_report(candidate_scores: pd.DataFrame, image_pred: pd.DataFrame, summary: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    backend = candidate_scores["clip_backend"].iloc[0] if len(candidate_scores) else "unknown"

    lines: List[str] = []
    lines.append("# Stage 10-E MVTec AD 2 VLM Full-image vs Candidate-crop Reasoning")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage evaluates whether PatchCore candidate crops improve CLIP/VLM binary anomaly reasoning on MVTec AD 2 vial.")
    lines.append("It compares full-image prompting with crop-based prompting using the candidate crops generated in Stage 10-D.")
    lines.append("")
    lines.append("## 2. Input")
    lines.append("")
    lines.append(f"- Candidate CSV: `{IN_REGIONS.relative_to(ROOT)}`")
    lines.append("- Candidate crop directory: `results/stage10_dataset_expansion/stage10_d_patchcore_candidate_crops/`")
    lines.append("")
    lines.append("## 3. VLM Backend")
    lines.append("")
    lines.append(f"- Backend: `{backend}`")
    lines.append("")
    lines.append("## 4. Prompts")
    lines.append("")
    lines.append("Normal prompts:")
    lines.append("")
    for p in NORMAL_PROMPTS:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("Anomaly prompts:")
    lines.append("")
    for p in ANOMALY_PROMPTS:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## 5. Output Files")
    lines.append("")
    lines.append(f"- Candidate scores: `{OUT_CANDIDATES.relative_to(ROOT)}`")
    lines.append(f"- Image predictions: `{OUT_IMAGES.relative_to(ROOT)}`")
    lines.append(f"- Summary: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 6. Summary")
    lines.append("")
    lines.append("| Method | Images | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {int(r['num_images'])} | {float(r['auroc']):.4f} | "
            f"{float(r['ap']):.4f} | {float(r['best_f1']):.4f} | {float(r['best_accuracy']):.4f} | "
            f"{float(r['delta_auroc_vs_full']):.4f} |"
        )

    lines.append("")
    lines.append("## 7. Interpretation Rule")
    lines.append("")
    lines.append("- If crop_topk_max > full_image, MVTec AD 2 supports the localization-guided VLM reasoning claim.")
    lines.append("- If crop_topk_max is close to or worse than full_image, inspect whether candidate crops are too small, visually ambiguous, or dominated by background.")
    lines.append("- PatchCore_score is included only as a detector reference, not as a VLM reasoning result.")
    lines.append("")
    lines.append("## 8. Next Step")
    lines.append("")
    lines.append("Stage 10-F should generate a comparison table that integrates Stage 10-C detector metrics, Stage 10-D candidate coverage, and Stage 10-E VLM reasoning.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Load candidate regions:", IN_REGIONS)
    regions = load_regions()
    print("[INFO] candidate rows:", len(regions))
    print("[INFO] unique images:", regions["image_path"].nunique())

    print("[INFO] Initialize CLIP scorer ...")
    scorer = ClipScorer()
    print("[INFO] backend:", scorer.backend)

    print("[INFO] Score full images and crops ...")
    candidate_scores = score_candidates(regions, scorer)
    image_pred = build_image_predictions(candidate_scores)
    summary = summarize_predictions(image_pred)

    candidate_scores.to_csv(OUT_CANDIDATES, index=False)
    image_pred.to_csv(OUT_IMAGES, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(candidate_scores, image_pred, summary)

    print("[DONE]", OUT_CANDIDATES)
    print("[DONE]", OUT_IMAGES)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
