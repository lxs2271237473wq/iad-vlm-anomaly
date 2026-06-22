import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_json", type=str, default="knowledge/mvtec_manufacturing_knowledge.json")
    parser.add_argument("--output_csv", type=str, default="results/analysis/semantic_prompt_bank/mvtec_prompt_bank.csv")
    args = parser.parse_args()

    knowledge_path = Path(args.knowledge_json)
    output_path = Path(args.output_csv)

    with knowledge_path.open("r", encoding="utf-8") as f:
        knowledge = json.load(f)

    rows = []

    for category, info in knowledge["categories"].items():
        for prompt in info.get("normal_prompts", []):
            rows.append({
                "dataset": knowledge["dataset"],
                "category": category,
                "prompt_type": "normal",
                "prompt": prompt,
                "object_type": info.get("object_type", ""),
                "material_or_surface": info.get("material_or_surface", ""),
                "inspection_focus": "; ".join(info.get("inspection_focus", []))
            })

        for prompt in info.get("defect_prompts", []):
            rows.append({
                "dataset": knowledge["dataset"],
                "category": category,
                "prompt_type": "defect",
                "prompt": prompt,
                "object_type": info.get("object_type", ""),
                "material_or_surface": info.get("material_or_surface", ""),
                "inspection_focus": "; ".join(info.get("inspection_focus", []))
            })

    for defect_name, defect_info in knowledge["generic_defect_types"].items():
        for prompt in defect_info.get("positive_prompts", []):
            rows.append({
                "dataset": knowledge["dataset"],
                "category": "generic",
                "prompt_type": f"defect_{defect_name}",
                "prompt": prompt,
                "object_type": "generic industrial defect",
                "material_or_surface": "",
                "inspection_focus": "; ".join(defect_info.get("visual_attributes", []))
            })

        for prompt in defect_info.get("negative_prompts", []):
            rows.append({
                "dataset": knowledge["dataset"],
                "category": "generic",
                "prompt_type": f"negative_{defect_name}",
                "prompt": prompt,
                "object_type": "generic industrial normal pattern",
                "material_or_surface": "",
                "inspection_focus": "; ".join(defect_info.get("possible_causes", []))
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "category",
        "prompt_type",
        "prompt",
        "object_type",
        "material_or_surface",
        "inspection_focus",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote prompt bank to: {output_path}")
    print(f"[DONE] Number of prompts: {len(rows)}")


if __name__ == "__main__":
    main()
