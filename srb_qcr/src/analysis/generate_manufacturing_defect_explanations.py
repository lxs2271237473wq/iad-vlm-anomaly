import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROCESS_PRIORS = {
    "grid": {
        "processes": ["molding", "surface forming", "pattern alignment", "visual inspection"],
        "default_causes": [
            "molding instability",
            "surface contamination",
            "pattern forming error",
            "mechanical impact during handling",
        ],
    },
    "screw": {
        "processes": ["metal forming", "threading", "surface treatment", "mechanical handling"],
        "default_causes": [
            "tool wear during threading",
            "mechanical collision",
            "surface treatment defect",
            "handling contamination",
        ],
    },
    "leather": {
        "processes": ["cutting", "pressing", "dyeing", "surface finishing"],
        "default_causes": [
            "cutting tool damage",
            "uneven dyeing",
            "surface abrasion",
            "finishing process defect",
        ],
    },
    "wood": {
        "processes": ["cutting", "polishing", "coating", "surface inspection"],
        "default_causes": [
            "cutting defect",
            "polishing defect",
            "coating irregularity",
            "mechanical scratch",
        ],
    },
}


DEFECT_FAMILY_RULES = [
    ("scratch", ["scratch"]),
    ("crack", ["crack", "broken", "hole"]),
    ("cut", ["cut", "poke"]),
    ("color", ["color", "liquid"]),
    ("contamination", ["contamination", "glue", "metal"]),
    ("deformation", ["bent", "fold", "manipulated", "thread"]),
]


VISUAL_EVIDENCE_RULES = {
    "scratch": "thin line-like surface damage or abrasion around the highlighted anomaly crop",
    "crack": "broken, missing, or hollow local structure around the detected anomaly region",
    "cut": "sharp local surface discontinuity or puncture-like damage",
    "color": "local color inconsistency, stain, or liquid-like appearance",
    "contamination": "foreign material, residue, or unexpected blob-like region",
    "deformation": "local shape distortion, bent structure, or abnormal thread-like geometry",
    "unknown": "local appearance inconsistent with the expected normal surface pattern",
}


def canonical_path(path):
    s = str(path).replace("\\", "/")
    marker = "datasets/MVTecAD/"
    if marker in s:
        return s[s.index(marker):]
    return s


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_defect_family(defect_type):
    defect_lower = str(defect_type).lower()

    for family, keys in DEFECT_FAMILY_RULES:
        if any(key in defect_lower for key in keys):
            return family

    return "unknown"


def get_category_knowledge(knowledge, category):
    categories = knowledge.get("categories", {})
    info = categories.get(category, {})

    priors = PROCESS_PRIORS.get(
        category,
        {
            "processes": ["manufacturing", "surface processing", "quality inspection"],
            "default_causes": [
                "manufacturing process variation",
                "surface damage",
                "handling damage",
                "material inconsistency",
            ],
        },
    )

    return {
        "object_type": info.get("object_type", category),
        "material_or_surface": info.get("material_or_surface", "industrial object surface"),
        "inspection_focus": info.get("inspection_focus", ["local visual anomaly"]),
        "normal_prompts": info.get("normal_prompts", []),
        "defect_prompts": info.get("defect_prompts", []),
        "processes": priors["processes"],
        "default_causes": priors["default_causes"],
    }


def get_defect_knowledge(knowledge, defect_family, category_info):
    generic = knowledge.get("generic_defect_types", {})
    info = generic.get(defect_family, {})

    visual_attributes = info.get("visual_attributes", [])
    possible_causes = info.get("possible_causes", [])

    if not possible_causes:
        possible_causes = category_info["default_causes"]

    if not visual_attributes:
        visual_attributes = [VISUAL_EVIDENCE_RULES.get(defect_family, VISUAL_EVIDENCE_RULES["unknown"])]

    return {
        "visual_attributes": visual_attributes,
        "possible_causes": possible_causes,
        "positive_prompts": info.get("positive_prompts", []),
        "negative_prompts": info.get("negative_prompts", []),
    }


def load_candidate_boxes(candidate_root, categories):
    boxes = {}

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

        for col in ["component_rank", "x1", "y1", "x2", "y2", "area", "mean_score", "max_score", "gt_iou", "gt_f1"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        valid_df = df[df["component_rank"] > 0].copy()

        for image_path, group in valid_df.groupby("image_path"):
            key = canonical_path(image_path)
            group = group.sort_values("component_rank")

            box_rows = []
            for _, row in group.iterrows():
                box_rows.append(
                    {
                        "rank": int(row["component_rank"]),
                        "x1": int(row["x1"]),
                        "y1": int(row["y1"]),
                        "x2": int(row["x2"]),
                        "y2": int(row["y2"]),
                        "area": float(row["area"]),
                        "mean_score": float(row["mean_score"]) if pd.notna(row["mean_score"]) else 0.0,
                        "max_score": float(row["max_score"]) if pd.notna(row["max_score"]) else 0.0,
                        "gt_iou": float(row["gt_iou"]) if "gt_iou" in row and pd.notna(row["gt_iou"]) else np.nan,
                        "gt_f1": float(row["gt_f1"]) if "gt_f1" in row and pd.notna(row["gt_f1"]) else np.nan,
                    }
                )

            boxes[key] = box_rows

    return boxes


def get_score(row, defect_type):
    col = f"score_{defect_type}"
    if col in row and pd.notna(row[col]):
        return float(row[col])
    return np.nan


def parse_top2(top2_text):
    if pd.isna(top2_text):
        return []
    return [x for x in str(top2_text).split("|") if x]


def confidence_proxy(row):
    pred = row["pred_defect_type"]
    pred_score = get_score(row, pred)

    top2 = parse_top2(row.get("top2_defect_types", ""))
    second_score = np.nan

    if len(top2) >= 2:
        second_score = get_score(row, top2[1])

    if np.isnan(pred_score) or np.isnan(second_score):
        return np.nan

    return float(pred_score - second_score)


def build_explanation(row, knowledge, candidate_boxes):
    category = row["category"]
    pred_defect = row["pred_defect_type"]
    true_defect = row.get("true_defect_type", "")

    category_info = get_category_knowledge(knowledge, category)
    defect_family = infer_defect_family(pred_defect)
    defect_info = get_defect_knowledge(knowledge, defect_family, category_info)

    key = canonical_path(row["image_path"])
    boxes = candidate_boxes.get(key, [])
    top_box = boxes[0] if boxes else None

    if top_box is None:
        bbox_text = "no candidate box available"
        region_text = "the suspicious region could not be localized by the candidate generator"
        anomaly_score_text = "unknown"
        gt_iou = np.nan
        gt_f1 = np.nan
    else:
        bbox_text = f"({top_box['x1']}, {top_box['y1']})-({top_box['x2']}, {top_box['y2']})"
        region_text = (
            f"the top PatchCore candidate region is located at {bbox_text}, "
            f"with area {top_box['area']:.0f}"
        )
        anomaly_score_text = f"mean={top_box['mean_score']:.4f}, max={top_box['max_score']:.4f}"
        gt_iou = top_box["gt_iou"]
        gt_f1 = top_box["gt_f1"]

    visual_evidence = VISUAL_EVIDENCE_RULES.get(defect_family, VISUAL_EVIDENCE_RULES["unknown"])
    visual_attributes = ", ".join(defect_info["visual_attributes"])
    processes = ", ".join(category_info["processes"])
    possible_causes = ", ".join(defect_info["possible_causes"])
    inspection_focus = ", ".join(category_info["inspection_focus"])

    explanation = (
        f"The image is predicted as defect type `{pred_defect}` on category `{category}`. "
        f"The product is described as a {category_info['object_type']} with "
        f"{category_info['material_or_surface']}. "
        f"Visual evidence for this prediction includes {visual_evidence}; "
        f"related attributes are {visual_attributes}. "
        f"According to the anomaly localization result, {region_text}. "
        f"Relevant manufacturing or inspection stages include {processes}. "
        f"Possible causes include {possible_causes}. "
        f"The inspection focus for this category includes {inspection_focus}."
    )

    return {
        "dataset": "MVTecAD",
        "category": category,
        "image_path": row["image_path"],
        "canonical_image_path": key,
        "true_defect_type": true_defect,
        "pred_defect_type": pred_defect,
        "top2_defect_types": row.get("top2_defect_types", ""),
        "top1_correct": row.get("top1_correct", ""),
        "top2_correct": row.get("top2_correct", ""),
        "defect_family": defect_family,
        "object_type": category_info["object_type"],
        "material_or_surface": category_info["material_or_surface"],
        "candidate_bbox": bbox_text,
        "candidate_anomaly_score": anomaly_score_text,
        "candidate_gt_iou": gt_iou,
        "candidate_gt_f1": gt_f1,
        "visual_evidence": visual_evidence,
        "visual_attributes": visual_attributes,
        "manufacturing_processes": processes,
        "possible_causes": possible_causes,
        "inspection_focus": inspection_focus,
        "clip_confidence_margin": confidence_proxy(row),
        "explanation": explanation,
    }


def write_sample_markdown(explanations_df, out_path, max_per_category=3):
    lines = [
        "# Sample Manufacturing-aware Defect Explanation Reports",
        "",
        "This file contains representative structured explanation examples generated from Stage 6.5.",
        "",
    ]

    for category in sorted(explanations_df["category"].unique()):
        sub = explanations_df[explanations_df["category"] == category].copy()
        sub = sub.sort_values(["top1_correct", "clip_confidence_margin"], ascending=[False, False])
        sub = sub.head(max_per_category)

        lines += [
            f"## {category}",
            "",
        ]

        for idx, row in enumerate(sub.itertuples(index=False), start=1):
            lines += [
                f"### Example {idx}",
                "",
                f"- Image: `{row.canonical_image_path}`",
                f"- True defect: `{row.true_defect_type}`",
                f"- Predicted defect: `{row.pred_defect_type}`",
                f"- Top-2 predictions: `{row.top2_defect_types}`",
                f"- Candidate region: `{row.candidate_bbox}`",
                f"- Candidate anomaly score: `{row.candidate_anomaly_score}`",
                f"- Defect family: `{row.defect_family}`",
                f"- Manufacturing processes: {row.manufacturing_processes}",
                f"- Possible causes: {row.possible_causes}",
                "",
                f"**Explanation:** {row.explanation}",
                "",
            ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_csv",
        type=str,
        default="results/analysis/real_anomaly_crop_visual_prompt_reasoning_full_test/real_anomaly_crop_visual_prompt_predictions.csv",
    )
    parser.add_argument(
        "--candidate_root",
        type=str,
        default="results/analysis/full_test_patchcore_candidate_regions",
    )
    parser.add_argument(
        "--knowledge_json",
        type=str,
        default="knowledge/mvtec_manufacturing_knowledge.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/analysis/manufacturing_defect_explanations",
    )
    parser.add_argument("--strategy", type=str, default="generic_label")
    parser.add_argument("--eval_mode", type=str, default="crop_topk_only")
    parser.add_argument("--sample_reports_per_category", type=int, default=3)
    args = parser.parse_args()

    prediction_csv = Path(args.prediction_csv)
    knowledge_json = Path(args.knowledge_json)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not prediction_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {prediction_csv}")

    if not knowledge_json.exists():
        raise FileNotFoundError(f"Missing knowledge JSON: {knowledge_json}")

    pred_df = pd.read_csv(prediction_csv)
    knowledge = load_json(knowledge_json)

    required_cols = ["strategy", "eval_mode", "category", "image_path", "true_defect_type", "pred_defect_type"]
    for col in required_cols:
        if col not in pred_df.columns:
            raise KeyError(f"Missing required prediction column: {col}")

    selected = pred_df[
        (pred_df["strategy"] == args.strategy)
        & (pred_df["eval_mode"] == args.eval_mode)
    ].copy()

    if len(selected) == 0:
        raise RuntimeError(
            f"No rows selected for strategy={args.strategy}, eval_mode={args.eval_mode}"
        )

    categories = sorted(selected["category"].unique().tolist())
    candidate_boxes = load_candidate_boxes(args.candidate_root, categories)

    explanation_rows = []
    for _, row in selected.iterrows():
        explanation_rows.append(build_explanation(row, knowledge, candidate_boxes))

    explanations_df = pd.DataFrame(explanation_rows)

    for col in ["top1_correct", "top2_correct"]:
        if col in explanations_df.columns:
            explanations_df[col] = explanations_df[col].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(explanations_df[col])

    out_csv = out_root / "manufacturing_defect_explanations.csv"
    explanations_df.to_csv(out_csv, index=False)

    summary_rows = []

    for category, group in explanations_df.groupby("category"):
        row = {
            "category": category,
            "num_reports": len(group),
            "num_pred_defect_types": group["pred_defect_type"].nunique(),
            "num_defect_families": group["defect_family"].nunique(),
            "mean_confidence_margin": pd.to_numeric(group["clip_confidence_margin"], errors="coerce").mean(),
        }

        if "top1_correct" in group.columns:
            row["top1_accuracy"] = pd.to_numeric(group["top1_correct"], errors="coerce").mean()
        if "top2_correct" in group.columns:
            row["top2_accuracy"] = pd.to_numeric(group["top2_correct"], errors="coerce").mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    mean_row = {
        "category": "MEAN",
        "num_reports": summary_df["num_reports"].sum(),
        "num_pred_defect_types": explanations_df["pred_defect_type"].nunique(),
        "num_defect_families": explanations_df["defect_family"].nunique(),
        "mean_confidence_margin": pd.to_numeric(explanations_df["clip_confidence_margin"], errors="coerce").mean(),
    }

    if "top1_accuracy" in summary_df.columns:
        mean_row["top1_accuracy"] = summary_df["top1_accuracy"].mean()
    if "top2_accuracy" in summary_df.columns:
        mean_row["top2_accuracy"] = summary_df["top2_accuracy"].mean()

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    summary_csv = out_root / "manufacturing_defect_explanation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    sample_md = out_root / "sample_manufacturing_defect_reports.md"
    write_sample_markdown(
        explanations_df=explanations_df,
        out_path=sample_md,
        max_per_category=args.sample_reports_per_category,
    )

    print("\n========== Explanation Summary ==========")
    print(summary_df.to_string(index=False))

    print(f"\n[DONE] Explanations saved to: {out_csv}")
    print(f"[DONE] Summary saved to: {summary_csv}")
    print(f"[DONE] Sample reports saved to: {sample_md}")


if __name__ == "__main__":
    main()
