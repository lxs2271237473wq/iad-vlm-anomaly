from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(".").resolve()

IN_REGIONS = ROOT / "results" / "stage10_dataset_expansion" / "stage10_d_patchcore_candidate_regions.csv"
IN_STAGE10E_IMAGES = ROOT / "results" / "stage10_dataset_expansion" / "stage10_e_vlm_image_predictions.csv"

OUT_DIR = ROOT / "results" / "stage10_dataset_expansion"
DOC_DIR = ROOT / "docs" / "stage10_dataset_expansion"

CROP_DIR = OUT_DIR / "stage10_f_multiscale_context_crops"

OUT_CROP_SCORES = OUT_DIR / "stage10_f_multiscale_context_crop_scores.csv"
OUT_IMAGE_PREDS = OUT_DIR / "stage10_f_multiscale_context_image_predictions.csv"
OUT_SUMMARY = OUT_DIR / "stage10_f_multiscale_context_summary.csv"
OUT_REPORT = DOC_DIR / "stage10_f_multiscale_context_crop_report.md"


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

CONTEXT_CONFIGS = [
    {"context_name": "context_0.20", "scale": 0.20, "square": False},
    {"context_name": "context_0.50", "scale": 0.50, "square": False},
    {"context_name": "context_1.00", "scale": 1.00, "square": False},
    {"context_name": "context_1.50", "scale": 1.50, "square": False},
    {"context_name": "square_context_1.00", "scale": 1.00, "square": True},
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
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
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
            "vlm_margin": anomaly_sim - normal_sim,
        }


def expanded_box(row: pd.Series, image_size: Tuple[int, int], scale: float, square: bool) -> Tuple[int, int, int, int]:
    width, height = image_size

    x1 = int(row["x1"])
    y1 = int(row["y1"])
    x2 = int(row["x2"])
    y2 = int(row["y2"])

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))

    if square:
        side = max(bw, bh) * (1.0 + 2.0 * scale)
        new_w = side
        new_h = side
    else:
        new_w = bw * (1.0 + 2.0 * scale)
        new_h = bh * (1.0 + 2.0 * scale)

    nx1 = max(0, int(round(cx - new_w / 2.0)))
    ny1 = max(0, int(round(cy - new_h / 2.0)))
    nx2 = min(width, int(round(cx + new_w / 2.0)))
    ny2 = min(height, int(round(cy + new_h / 2.0)))

    if nx2 <= nx1:
        nx2 = min(width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(height, ny1 + 1)

    return nx1, ny1, nx2, ny2


def reset_crop_dir() -> None:
    if CROP_DIR.exists():
        shutil.rmtree(CROP_DIR)
    CROP_DIR.mkdir(parents=True, exist_ok=True)


def load_regions() -> pd.DataFrame:
    if not IN_REGIONS.exists():
        raise FileNotFoundError(f"Missing regions CSV: {IN_REGIONS}")

    df = pd.read_csv(IN_REGIONS)

    required = [
        "image_path",
        "gt_label",
        "candidate_rank",
        "candidate_available",
        "x1",
        "y1",
        "x2",
        "y2",
        "pred_score",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Stage 10-D regions CSV missing columns: {missing}")

    df = df[df["candidate_available"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    df["gt_binary"] = df["gt_label"].map(to_binary_label)

    if df.empty:
        raise RuntimeError("No available candidate rows.")

    return df


def build_multiscale_crops(regions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for idx, row in regions.reset_index(drop=True).iterrows():
        image_path = resolve_path(row["image_path"])

        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        label_dir = "bad" if int(row["gt_binary"]) == 1 else "good"

        for cfg in CONTEXT_CONFIGS:
            box = expanded_box(row, image.size, float(cfg["scale"]), bool(cfg["square"]))
            x1, y1, x2, y2 = box

            image_stem = image_path.stem
            crop_name = f"{image_stem}_cand{int(row['candidate_rank'])}_{cfg['context_name']}.png"
            crop_path = CROP_DIR / label_dir / cfg["context_name"] / crop_name
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            image.crop(box).save(crop_path)

            out = row.to_dict()
            out.update({
                "context_name": cfg["context_name"],
                "context_scale": cfg["scale"],
                "square_context": cfg["square"],
                "context_crop_path": str(crop_path.relative_to(ROOT)),
                "context_x1": x1,
                "context_y1": y1,
                "context_x2": x2,
                "context_y2": y2,
                "context_width": x2 - x1,
                "context_height": y2 - y1,
                "image_width": image.size[0],
                "image_height": image.size[1],
            })
            rows.append(out)

    return pd.DataFrame(rows)


def score_crops(crops: pd.DataFrame, scorer: ClipScorer) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    cache: Dict[str, Dict[str, float]] = {}

    for _, row in crops.iterrows():
        crop_path = resolve_path(row["context_crop_path"])

        if str(crop_path) not in cache:
            cache[str(crop_path)] = scorer.score_image(crop_path)

        scores = cache[str(crop_path)]

        out = row.to_dict()
        out.update({
            "clip_backend": scorer.backend,
            "context_normal_score": scores["normal_score"],
            "context_anomaly_score": scores["anomaly_score"],
            "context_vlm_margin": scores["vlm_margin"],
        })
        rows.append(out)

    return pd.DataFrame(rows)


def build_image_predictions(score_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (image_path, context_name), part in score_df.groupby(["image_path", "context_name"], dropna=False, sort=False):
        part = part.copy()
        part["candidate_rank"] = pd.to_numeric(part["candidate_rank"], errors="coerce")
        part_ranked = part.sort_values("candidate_rank")

        top1 = part_ranked.iloc[0]
        max_idx = pd.to_numeric(part["context_vlm_margin"], errors="coerce").idxmax()
        max_row = part.loc[max_idx]

        rows.append({
            "dataset": "MVTec AD 2",
            "category": "vial",
            "image_path": image_path,
            "context_name": context_name,
            "gt_binary": int(top1["gt_binary"]),
            "num_candidates": int(len(part)),
            "clip_backend": top1["clip_backend"],
            "context_top1_score": float(top1["context_vlm_margin"]),
            "context_topk_max_score": float(max_row["context_vlm_margin"]),
            "context_topk_mean_score": float(pd.to_numeric(part["context_vlm_margin"], errors="coerce").mean()),
            "best_context_crop_path": max_row["context_crop_path"],
            "patchcore_pred_score": float(pd.to_numeric(part["pred_score"], errors="coerce").iloc[0]),
        })

    out = pd.DataFrame(rows)

    if IN_STAGE10E_IMAGES.exists():
        e = pd.read_csv(IN_STAGE10E_IMAGES)
        keep_cols = ["image_path", "full_image_score", "crop_top1_score", "crop_topk_max_score", "crop_topk_mean_score"]
        keep_cols = [c for c in keep_cols if c in e.columns]
        if "image_path" in keep_cols:
            out = out.merge(e[keep_cols].drop_duplicates("image_path"), on="image_path", how="left")

    return out


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    score_specs = [
        ("full_image", "full_image_score"),
        ("stage10e_crop_top1", "crop_top1_score"),
        ("stage10e_crop_topk_max", "crop_topk_max_score"),
        ("stage10e_crop_topk_mean", "crop_topk_mean_score"),
        ("patchcore_score", "patchcore_pred_score"),
    ]

    for context_name in pred["context_name"].unique():
        score_specs.extend([
            (f"{context_name}_top1", "context_top1_score"),
            (f"{context_name}_topk_max", "context_topk_max_score"),
            (f"{context_name}_topk_mean", "context_topk_mean_score"),
        ])

    y_base = pred.drop_duplicates("image_path")[["image_path", "gt_binary"]].copy()
    y = y_base["gt_binary"].astype(int).to_numpy()

    for method, score_col in score_specs:
        if score_col not in pred.columns:
            continue

        if method in {"full_image", "stage10e_crop_top1", "stage10e_crop_topk_max", "stage10e_crop_topk_mean", "patchcore_score"}:
            part = pred.drop_duplicates("image_path")
        else:
            if method.endswith("_topk_max"):
                context_name = method[: -len("_topk_max")]
            elif method.endswith("_topk_mean"):
                context_name = method[: -len("_topk_mean")]
            elif method.endswith("_top1"):
                context_name = method[: -len("_top1")]
            else:
                context_name = method
            part = pred[pred["context_name"] == context_name].drop_duplicates("image_path")

        if part.empty or score_col not in part.columns:
            continue

        yy = part["gt_binary"].astype(int).to_numpy()
        ss = pd.to_numeric(part[score_col], errors="coerce").fillna(0.0).to_numpy()
        f1, acc, thr = best_f1_accuracy(yy, ss)

        rows.append({
            "dataset": "MVTec AD 2",
            "category": "vial",
            "method": method,
            "num_images": int(len(part)),
            "num_normal": int((yy == 0).sum()),
            "num_anomaly": int((yy == 1).sum()),
            "auroc": binary_auroc(yy, ss),
            "ap": average_precision(yy, ss),
            "best_f1": f1,
            "best_accuracy": acc,
            "best_threshold": thr,
        })

    summary = pd.DataFrame(rows)

    if "full_image" in set(summary["method"]):
        base = summary[summary["method"] == "full_image"][["auroc", "ap", "best_f1", "best_accuracy"]].iloc[0]
        summary["delta_auroc_vs_full"] = summary["auroc"] - float(base["auroc"])
        summary["delta_ap_vs_full"] = summary["ap"] - float(base["ap"])
        summary["delta_best_f1_vs_full"] = summary["best_f1"] - float(base["best_f1"])
        summary["delta_accuracy_vs_full"] = summary["best_accuracy"] - float(base["best_accuracy"])

    return summary.sort_values("auroc", ascending=False)


def write_report(summary: pd.DataFrame, score_df: pd.DataFrame) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    backend = score_df["clip_backend"].iloc[0] if len(score_df) else "unknown"

    lines: List[str] = []
    lines.append("# Stage 10-F Multiscale Context Crop Diagnostic")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("Stage 10-E showed that direct candidate crops underperform full-image VLM prompting.")
    lines.append("This stage tests whether adding spatial context around PatchCore candidate boxes improves VLM reasoning.")
    lines.append("")
    lines.append("## 2. Inputs")
    lines.append("")
    lines.append(f"- Candidate regions: `{IN_REGIONS.relative_to(ROOT)}`")
    lines.append(f"- Stage 10-E image predictions: `{IN_STAGE10E_IMAGES.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 3. Context Configurations")
    lines.append("")
    lines.append("| Context | Scale | Square |")
    lines.append("|---|---:|---:|")
    for cfg in CONTEXT_CONFIGS:
        lines.append(f"| {cfg['context_name']} | {cfg['scale']} | {cfg['square']} |")
    lines.append("")
    lines.append("## 4. VLM Backend")
    lines.append("")
    lines.append(f"- Backend: `{backend}`")
    lines.append("")
    lines.append("## 5. Output Files")
    lines.append("")
    lines.append(f"- Crop scores: `{OUT_CROP_SCORES.relative_to(ROOT)}`")
    lines.append(f"- Image predictions: `{OUT_IMAGE_PREDS.relative_to(ROOT)}`")
    lines.append(f"- Summary: `{OUT_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append(f"- Generated crops: `{CROP_DIR.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 6. Summary")
    lines.append("")
    lines.append("| Method | Images | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {int(r['num_images'])} | {float(r['auroc']):.4f} | "
            f"{float(r['ap']):.4f} | {float(r['best_f1']):.4f} | {float(r['best_accuracy']):.4f} | "
            f"{float(r.get('delta_auroc_vs_full', 0.0)):.4f} |"
        )

    lines.append("")
    lines.append("## 7. Decision Rule")
    lines.append("")
    lines.append("- If any context-crop method exceeds full_image, keep MVTec AD 2 as positive evidence for context-aware localization-guided VLM reasoning.")
    lines.append("- If context crops remain below full_image, treat vial as a negative case and move to either another AD2 category or logical-anomaly data such as MVTec LOCO AD.")
    lines.append("- PatchCore score is a detector reference, not VLM reasoning evidence.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    reset_crop_dir()

    print("[INFO] Load regions:", IN_REGIONS)
    regions = load_regions()
    print("[INFO] candidate rows:", len(regions))
    print("[INFO] images:", regions["image_path"].nunique())

    print("[INFO] Build multiscale context crops ...")
    crops = build_multiscale_crops(regions)
    print("[INFO] context crop rows:", len(crops))

    print("[INFO] Initialize CLIP scorer ...")
    scorer = ClipScorer()
    print("[INFO] backend:", scorer.backend)

    print("[INFO] Score context crops ...")
    score_df = score_crops(crops, scorer)
    image_pred = build_image_predictions(score_df)
    summary = summarize(image_pred)

    score_df.to_csv(OUT_CROP_SCORES, index=False)
    image_pred.to_csv(OUT_IMAGE_PREDS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    write_report(summary, score_df)

    print("[DONE]", OUT_CROP_SCORES)
    print("[DONE]", OUT_IMAGE_PREDS)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_REPORT)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
