import argparse
from pathlib import Path
import pandas as pd


MANUFACTURING_KNOWLEDGE = {
    "grid": {
        "object_type": "regular grid-like structure",
        "material_or_surface": "regular repeated texture surface",
        "processes": ["molding", "surface forming", "pattern alignment", "inspection"],
        "inspection_focus": [
            "broken repeated structure",
            "local contamination",
            "missing or deformed grid pattern",
            "irregular texture interruption",
        ],
        "possible_causes": [
            "molding defect",
            "surface contamination",
            "mechanical impact",
            "pattern forming error",
        ],
    },
    "screw": {
        "object_type": "metal screw",
        "material_or_surface": "metal threaded surface",
        "processes": ["metal forming", "threading", "surface treatment", "mechanical handling"],
        "inspection_focus": [
            "thread damage",
            "scratch",
            "deformation",
            "contamination",
            "local missing structure",
        ],
        "possible_causes": [
            "tool wear during threading",
            "mechanical collision",
            "surface treatment defect",
            "handling contamination",
        ],
    },
    "leather": {
        "object_type": "leather surface",
        "material_or_surface": "soft textured leather material",
        "processes": ["cutting", "pressing", "dyeing", "surface finishing"],
        "inspection_focus": [
            "cut mark",
            "color inconsistency",
            "scratch",
            "local surface damage",
        ],
        "possible_causes": [
            "cutting tool damage",
            "uneven dyeing",
            "surface abrasion",
            "finishing process defect",
        ],
    },
    "wood": {
        "object_type": "wooden surface",
        "material_or_surface": "natural wood grain texture",
        "processes": ["cutting", "polishing", "coating", "surface inspection"],
        "inspection_focus": [
            "scratch crossing wood grain",
            "local color change",
            "crack",
            "contamination",
            "texture interruption",
        ],
        "possible_causes": [
            "cutting defect",
            "polishing defect",
            "coating irregularity",
            "mechanical scratch",
        ],
    },
}


GENERIC_DEFECT_DESCRIPTIONS = {
    "scratch": "a thin line-like surface damage or abrasion",
    "crack": "a broken or split structure on the surface",
    "cut": "a sharp local surface cut or incision",
    "color": "a local abnormal color change or stain",
    "contamination": "foreign material or dirty region on the object surface",
    "hole": "a local missing or hollow region",
    "thread": "abnormal damage around threaded structure",
    "bent": "local shape deformation or bending",
    "broken": "broken or missing object structure",
    "metal": "metallic surface damage or contamination",
    "poke": "small local puncture-like defect",
    "glue": "abnormal glue-like contamination",
    "print": "abnormal printed pattern or local printing defect",
}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def defect_description(defect_type: str) -> str:
    lower = defect_type.lower()
    matched = []

    for key, desc in GENERIC_DEFECT_DESCRIPTIONS.items():
        if key in lower:
            matched.append(desc)

    if matched:
        return "; ".join(sorted(set(matched)))

    return f"visual defect of type {defect_type}"


def build_prompts(category: str, defect_type: str):
    info = MANUFACTURING_KNOWLEDGE.get(
        category,
        {
            "object_type": category,
            "material_or_surface": "industrial object surface",
            "processes": ["manufacturing", "surface processing", "quality inspection"],
            "inspection_focus": ["local visual anomaly", "surface defect", "structural inconsistency"],
            "possible_causes": ["manufacturing defect", "surface damage", "process variation"],
        },
    )

    defect_desc = defect_description(defect_type)

    generic_prompt = (
        f"a photo of an industrial defect: {defect_type}, "
        f"showing {defect_desc}"
    )

    category_aware_prompt = (
        f"a photo of a defective {info['object_type']} with {defect_type}, "
        f"showing {defect_desc}"
    )

    process_text = ", ".join(info["processes"])
    focus_text = ", ".join(info["inspection_focus"])
    cause_text = ", ".join(info["possible_causes"])

    manufacturing_aware_prompt = (
        f"a quality inspection image of a defective {info['object_type']} made of "
        f"{info['material_or_surface']}. The defect type is {defect_type}. "
        f"The visual evidence may include {defect_desc}. "
        f"The relevant manufacturing processes include {process_text}. "
        f"The inspection should focus on {focus_text}. "
        f"Possible manufacturing causes include {cause_text}."
    )

    normal_prompt = (
        f"a normal quality inspection image of a defect-free {info['object_type']} "
        f"with regular {info['material_or_surface']} and no visible defect"
    )

    return {
        "generic_prompt": generic_prompt,
        "category_aware_prompt": category_aware_prompt,
        "manufacturing_aware_prompt": manufacturing_aware_prompt,
        "normal_prompt": normal_prompt,
        "manufacturing_processes": process_text,
        "inspection_focus": focus_text,
        "possible_causes": cause_text,
        "material_or_surface": info["material_or_surface"],
        "object_type": info["object_type"],
    }


def scan_category(data_root: Path, category: str):
    test_dir = data_root / category / "test"

    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    rows = []

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue

        defect_type = defect_dir.name

        if defect_type == "good":
            continue

        image_files = [
            p for p in sorted(defect_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_files:
            prompt_info = build_prompts(category, defect_type)

            row = {
                "dataset": "MVTecAD",
                "category": category,
                "defect_type": defect_type,
                "image_path": str(image_path),
            }
            row.update(prompt_info)
            rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets/MVTecAD")
    parser.add_argument("--categories", nargs="+", default=["grid", "screw", "leather", "wood"])
    parser.add_argument("--output_root", type=str, default="results/analysis/defect_type_reasoning")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for category in args.categories:
        category_rows = scan_category(data_root, category)
        rows.extend(category_rows)
        print(f"[DONE] {category}: {len(category_rows)} abnormal test images")

    manifest_df = pd.DataFrame(rows)

    manifest_csv = out_root / "mvtec_defect_type_reasoning_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    summary_df = (
        manifest_df
        .groupby(["category", "defect_type"])
        .size()
        .reset_index(name="num_images")
        .sort_values(["category", "defect_type"])
    )

    summary_csv = out_root / "mvtec_defect_type_reasoning_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    category_summary = (
        manifest_df
        .groupby("category")
        .agg(num_images=("image_path", "count"), num_defect_types=("defect_type", "nunique"))
        .reset_index()
        .sort_values("category")
    )

    category_summary_csv = out_root / "mvtec_defect_type_category_summary.csv"
    category_summary.to_csv(category_summary_csv, index=False)

    print("\n========== Category Summary ==========")
    print(category_summary.to_string(index=False))

    print("\n========== Defect Type Summary ==========")
    print(summary_df.to_string(index=False))

    print(f"\n[DONE] Manifest saved to: {manifest_csv}")
    print(f"[DONE] Defect summary saved to: {summary_csv}")
    print(f"[DONE] Category summary saved to: {category_summary_csv}")


if __name__ == "__main__":
    main()
