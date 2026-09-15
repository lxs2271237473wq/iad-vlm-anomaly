from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.patches import Rectangle


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

INVENTORY = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_case_analysis"
    / "stage22_d7b_label_aware_case_inventory.csv"
)

CANDIDATE_ROOT = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_rerun_patchcore"
    / "MVTecAD"
)

OUT_DIR = (
    ROOT
    / "results/stage22_selective_qcr"
    / "mvtec15_case_analysis"
    / "figures"
)

OUT_MANIFEST = (
    OUT_DIR
    / "stage22_d7c_figure_manifest.csv"
)

OUT_PNG = (
    OUT_DIR
    / "stage22_d7c_gate_behavior_grid.png"
)

OUT_SVG = (
    OUT_DIR
    / "stage22_d7c_gate_behavior_grid.svg"
)

CASE_TYPES = [
    "gate_off_protected_from_harmful_vlm",
    "gate_off_missed_vlm_opportunity",
    "gate_on_helpful",
    "gate_on_harmful",
]

CASE_LABELS = {
    "gate_off_protected_from_harmful_vlm": (
        "Gate off: harmful VLM rejected"
    ),
    "gate_off_missed_vlm_opportunity": (
        "Gate off: useful VLM missed"
    ),
    "gate_on_helpful": (
        "Gate on: useful VLM accepted"
    ),
    "gate_on_harmful": (
        "Gate on: harmful VLM accepted"
    ),
}


def candidate_csv(category: str) -> Path:
    return (
        CANDIDATE_ROOT
        / category
        / "candidate_regions.csv"
    )


def load_boxes(
    category: str,
    image_path: str,
    top_k: int = 3,
) -> list[dict]:
    path = candidate_csv(category)

    if not path.exists():
        raise FileNotFoundError(path)

    candidates = pd.read_csv(path)

    required = {
        "image_path",
        "component_rank",
        "candidate_available",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing = sorted(
        required - set(candidates.columns)
    )

    if missing:
        raise RuntimeError(
            f"{category}: candidate CSV missing {missing}"
        )

    candidates["component_rank"] = pd.to_numeric(
        candidates["component_rank"],
        errors="coerce",
    )

    candidates["candidate_available"] = pd.to_numeric(
        candidates["candidate_available"],
        errors="coerce",
    ).fillna(0)

    image_key = str(
        Path(image_path).resolve(strict=False)
    )

    candidates["_resolved"] = (
        candidates["image_path"]
        .astype(str)
        .map(
            lambda value: str(
                Path(value).resolve(strict=False)
            )
        )
    )

    selected = candidates[
        (candidates["_resolved"] == image_key)
        & (candidates["candidate_available"] == 1)
        & (candidates["component_rank"] > 0)
    ].copy()

    selected = (
        selected.sort_values("component_rank")
        .head(top_k)
    )

    boxes = []

    for _, row in selected.iterrows():
        boxes.append(
            {
                "rank": int(row["component_rank"]),
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
            }
        )

    return boxes


def select_figure_cases(
    inventory: pd.DataFrame,
    per_type: int = 2,
) -> pd.DataFrame:
    rows = []

    for case_type in CASE_TYPES:
        part = (
            inventory[
                inventory["case_type"] == case_type
            ]
            .sort_values("case_rank")
            .head(per_type)
            .copy()
        )

        if len(part) != per_type:
            raise RuntimeError(
                f"{case_type}: expected {per_type} rows, "
                f"found {len(part)}"
            )

        rows.append(part)

    selected = pd.concat(
        rows,
        ignore_index=True,
    )

    return selected


def add_candidate_boxes(
    axis,
    boxes: list[dict],
) -> None:
    for box in boxes:
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]

        rectangle = Rectangle(
            (box["x1"], box["y1"]),
            width,
            height,
            fill=False,
            linewidth=2.0,
        )

        axis.add_patch(rectangle)

        axis.text(
            box["x1"],
            max(0, box["y1"] - 4),
            f"R{box['rank']}",
            fontsize=8,
        )


def panel_title(row: pd.Series) -> str:
    label_text = (
        "anomaly"
        if int(row["Y"]) == 1
        else "normal"
    )

    benefit = float(
        row["srb_benefit_vs_detector"]
    )

    return (
        f"{row['category']} | {label_text}\n"
        f"D={row['D']:.3f}, M={row['M']:.3f}, "
        f"Q={row['Q']:.3f}, SRB={row['score_S1']:.3f}\n"
        f"label-aware benefit={benefit:+.3f}"
    )


def main() -> None:
    if not INVENTORY.exists():
        raise FileNotFoundError(INVENTORY)

    inventory = pd.read_csv(INVENTORY)

    required = {
        "case_type",
        "case_rank",
        "category",
        "image_path",
        "Y",
        "D",
        "M",
        "Q",
        "score_S1",
        "srb_pre_gate",
        "srb_weight",
        "srb_benefit_vs_detector",
        "naive_harm_vs_detector",
    }

    missing = sorted(
        required - set(inventory.columns)
    )

    if missing:
        raise RuntimeError(
            f"Inventory missing columns: {missing}"
        )

    selected = select_figure_cases(
        inventory,
        per_type=2,
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    figure, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(12, 18),
        constrained_layout=True,
    )

    for row_index, case_type in enumerate(
        CASE_TYPES
    ):
        part = (
            selected[
                selected["case_type"] == case_type
            ]
            .sort_values("case_rank")
            .reset_index(drop=True)
        )

        for column_index, (_, row) in enumerate(
            part.iterrows()
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            image_path = Path(
                str(row["image_path"])
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    image_path
                )

            with Image.open(image_path) as image:
                display_image = image.convert("RGB")
                axis.imshow(display_image)

            boxes = load_boxes(
                category=str(row["category"]),
                image_path=str(image_path),
                top_k=3,
            )

            add_candidate_boxes(
                axis,
                boxes,
            )

            axis.set_title(
                panel_title(row),
                fontsize=10,
            )

            axis.set_axis_off()

            if column_index == 0:
                axis.text(
                    -0.04,
                    0.5,
                    CASE_LABELS[case_type],
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axis.transAxes,
                    fontsize=11,
                )

            manifest_rows.append(
                {
                    "figure_row": row_index + 1,
                    "figure_column": column_index + 1,
                    "case_type": case_type,
                    "case_rank": int(
                        row["case_rank"]
                    ),
                    "category": row["category"],
                    "image_path": str(image_path),
                    "Y": int(row["Y"]),
                    "D": float(row["D"]),
                    "M": float(row["M"]),
                    "Q": float(row["Q"]),
                    "score_S1": float(
                        row["score_S1"]
                    ),
                    "srb_pre_gate": int(
                        row["srb_pre_gate"]
                    ),
                    "srb_weight": float(
                        row["srb_weight"]
                    ),
                    "srb_benefit_vs_detector": float(
                        row[
                            "srb_benefit_vs_detector"
                        ]
                    ),
                    "naive_harm_vs_detector": float(
                        row[
                            "naive_harm_vs_detector"
                        ]
                    ),
                    "candidate_boxes": len(boxes),
                }
            )

    figure.suptitle(
        "SRB-QCR gate behavior on MVTec AD",
        fontsize=16,
    )

    figure.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        OUT_SVG,
        bbox_inches="tight",
    )

    plt.close(figure)

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        OUT_MANIFEST,
        index=False,
        lineterminator="\n",
    )

    print("===== D7c FIGURE MANIFEST =====")
    print(
        manifest[
            [
                "figure_row",
                "figure_column",
                "case_type",
                "category",
                "Y",
                "D",
                "M",
                "Q",
                "score_S1",
                "srb_benefit_vs_detector",
                "candidate_boxes",
                "image_path",
            ]
        ].to_string(index=False)
    )

    print()
    print("[DONE]", OUT_PNG)
    print("[DONE]", OUT_SVG)
    print("[DONE]", OUT_MANIFEST)


if __name__ == "__main__":
    main()
