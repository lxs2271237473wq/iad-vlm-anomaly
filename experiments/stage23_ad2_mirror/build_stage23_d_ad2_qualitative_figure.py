#!/usr/bin/env python3
"""Build a four-case AD2 qualitative figure from locked per-image predictions.

This script does not tune parameters and does not rerun VLM inference. It mines
the existing frozen prediction table and resolves original image paths.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

PREDICTION_CANDIDATES = [
    ROOT / "results/stage22_selective_qcr/stage22_b2b_ad2_frozen_predictions.csv",
    ROOT / "results/stage23_ad2_mirror/stage23_b1_ad2_frozen_predictions.csv",
    ROOT / "results/stage23_ad2_mirror/stage23_b1_ad2_frozen_mirror_predictions.csv",
]

OUT_DIR = ROOT / "results/stage23_ad2_mirror/qualitative"
DOC_DIR = ROOT / "docs/stage23_ad2_mirror"

ALIASES = {
    "label": ["Y", "gt_binary", "label", "target"],
    "image_path": ["image_path", "path", "filepath"],
    "D": ["D", "detector_score", "patchcore_score"],
    "M": ["M", "vlm_score", "context_topk_mean_score"],
    "Q": ["Q", "quality", "candidate_quality"],
    "S": ["score_S1", "S", "srb_score"],
    "gate": ["srb_pre_gate", "G_pre", "gate"],
    "weight": ["srb_weight", "w", "weight"],
}

def choose_column(df: pd.DataFrame, role: str, required: bool = True) -> str | None:
    for name in ALIASES[role]:
        if name in df.columns:
            return name
    if required:
        raise RuntimeError(f"Missing column for {role}; tried {ALIASES[role]}")
    return None

def find_predictions() -> Path:
    for path in PREDICTION_CANDIDATES:
        if path.exists():
            return path
    matches = list((ROOT / "results").rglob("*ad2*prediction*.csv"))
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError("No AD2 frozen prediction table found.")

def resolve_image(value: str) -> Path:
    p = Path(str(value))
    candidates = [p, ROOT / p]
    text = str(value).replace("\\", "/")
    for marker in ["datasets/", "MVTec_AD_2_anomalib_all/"]:
        if marker in text:
            suffix = text.split(marker, 1)[1]
            candidates.extend([
                ROOT / "datasets" / suffix,
                ROOT / "datasets/MVTec_AD_2_anomalib_all" / suffix,
            ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(value)

def first_distinct(part: pd.DataFrame, used: set[str]) -> pd.Series:
    for _, row in part.iterrows():
        key = str(row["_image_key"])
        if key not in used:
            used.add(key)
            return row
    raise RuntimeError("Not enough distinct qualitative cases.")

def select_cases(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    used: set[str] = set()
    gate_off = df[df["_gate"] == 0]
    gate_on = df[df["_gate"] == 1]

    case_a = first_distinct(
        gate_off[gate_off["_vlm_directional"] < 0].sort_values("_vlm_directional"),
        used,
    )
    case_b = first_distinct(
        gate_off[gate_off["_vlm_directional"] > 0].sort_values("_vlm_directional", ascending=False),
        used,
    )
    case_c = first_distinct(
        gate_on[gate_on["_srb_benefit"] > 0].sort_values("_srb_benefit", ascending=False),
        used,
    )
    case_d = first_distinct(
        gate_on[gate_on["_srb_benefit"] < 0].sort_values("_srb_benefit"),
        used,
    )
    return [
        ("A", "Gate OFF: harmful VLM rejected", case_a),
        ("B", "Gate OFF: useful VLM missed", case_b),
        ("C", "Gate ON: useful VLM accepted", case_c),
        ("D", "Gate ON: harmful VLM accepted", case_d),
    ]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()

    source = args.predictions or find_predictions()
    df = pd.read_csv(source)

    cols = {role: choose_column(df, role, required=(role != "weight")) for role in ALIASES}
    frame = pd.DataFrame({
        "category": df["category"].astype(str),
        "image_path": df[cols["image_path"]].astype(str),
        "Y": pd.to_numeric(df[cols["label"]], errors="raise").astype(int),
        "D": pd.to_numeric(df[cols["D"]], errors="raise"),
        "M": pd.to_numeric(df[cols["M"]], errors="raise"),
        "Q": pd.to_numeric(df[cols["Q"]], errors="raise"),
        "score_S1": pd.to_numeric(df[cols["S"]], errors="raise"),
        "srb_pre_gate": pd.to_numeric(df[cols["gate"]], errors="raise").astype(int),
    })
    frame["srb_weight"] = (
        pd.to_numeric(df[cols["weight"]], errors="coerce")
        if cols["weight"] else np.nan
    )
    frame["_image_key"] = frame["image_path"].str.replace("\\", "/", regex=False)
    frame["_gate"] = frame["srb_pre_gate"]
    frame["_vlm_directional"] = np.where(
        frame["Y"] == 1, frame["M"] - frame["D"], frame["D"] - frame["M"]
    )
    frame["_srb_benefit"] = np.where(
        frame["Y"] == 1,
        frame["score_S1"] - frame["D"],
        frame["D"] - frame["score_S1"],
    )

    cases = select_cases(frame)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    manifest = []
    for ax, (panel, headline, row) in zip(axes.ravel(), cases):
        path = resolve_image(row["image_path"])
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_axis_off()
        ax.set_title(
            f"({panel}) {headline}\n"
            f"{row['category']} | Y={int(row['Y'])} | "
            f"D={row['D']:.3f}, M={row['M']:.3f}, Q={row['Q']:.3f}, "
            f"SRB={row['score_S1']:.3f}",
            fontsize=10,
        )
        manifest.append({
            "panel": panel,
            "headline": headline,
            "category": row["category"],
            "image_path": str(path),
            "Y": int(row["Y"]),
            "D": float(row["D"]),
            "M": float(row["M"]),
            "Q": float(row["Q"]),
            "score_S1": float(row["score_S1"]),
            "srb_pre_gate": int(row["srb_pre_gate"]),
            "srb_weight": None if pd.isna(row["srb_weight"]) else float(row["srb_weight"]),
            "vlm_directional_advantage": float(row["_vlm_directional"]),
            "srb_benefit_vs_detector": float(row["_srb_benefit"]),
        })

    fig.suptitle("AD2 reliability-boundary cases under frozen SRB-QCR", fontsize=15)
    png = OUT_DIR / "stage23_d_ad2_reliability_boundary_2x2.png"
    svg = OUT_DIR / "stage23_d_ad2_reliability_boundary_2x2.svg"
    csv = OUT_DIR / "stage23_d_ad2_reliability_boundary_manifest.csv"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(manifest).to_csv(csv, index=False)

    caption = (
        "Four label-aware reliability-boundary cases on the AD2 four-category "
        "frozen-transfer protocol. Gate-off M values are retrospective audit values "
        "and are not computed during actual selective deployment."
    )
    (DOC_DIR / "stage23_d_ad2_qualitative_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    print("[DONE]", png)
    print("[DONE]", svg)
    print("[DONE]", csv)

if __name__ == "__main__":
    main()
