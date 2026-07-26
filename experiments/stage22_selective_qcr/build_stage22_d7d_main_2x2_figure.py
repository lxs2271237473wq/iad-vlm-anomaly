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

OUT_PNG = (
    OUT_DIR
    / "stage22_d7d_main_gate_behavior_2x2.png"
)

OUT_SVG = (
    OUT_DIR
    / "stage22_d7d_main_gate_behavior_2x2.svg"
)

OUT_MANIFEST = (
    OUT_DIR
    / "stage22_d7d_main_figure_manifest.csv"
)

OUT_CAPTION = (
    ROOT
    / "docs/stage22_selective_qcr"
    / "stage22_d7d_main_figure_caption.md"
)

# Four paper-facing cases selected from the validated D7b inventory.
SELECTIONS = [
    {
        "panel": "A",
        "case_type": "gate_off_protected_from_harmful_vlm",
        "category": "metal_nut",
        "case_rank": 1,
        "headline": "Gate OFF — harmful VLM rejected",
    },
    {
        "panel": "B",
        "case_type": "gate_off_missed_vlm_opportunity",
        "category": "tile",
        "case_rank": 1,
        "headline": "Gate OFF — useful VLM missed",
    },
    {
        "panel": "C",
        "case_type": "gate_on_helpful",
        "category": "tile",
        "case_rank": 2,
        "headline": "Gate ON — useful VLM accepted",
    },
    {
        "panel": "D",
        "case_type": "gate_on_harmful",
        "category": "capsule",
        "case_rank": 1,
        "headline": "Gate ON — harmful VLM accepted",
    },
]


def candidate_csv(category: str) -> Path:
    return (
        CANDIDATE_ROOT
        / category
        / "candidate_regions.csv"
    )


def resolve_path(value: str) -> str:
    return str(
        Path(str(value)).resolve(strict=False)
    )


def load_boxes(
    category: str,
    image_path: str,
    top_k: int = 3,
) -> list[dict]:
    path = candidate_csv(category)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

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
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{category}: candidate CSV missing {missing}"
        )

    df["component_rank"] = pd.to_numeric(
        df["component_rank"],
        errors="coerce",
    )

    df["candidate_available"] = pd.to_numeric(
        df["candidate_available"],
        errors="coerce",
    ).fillna(0)

    target = resolve_path(image_path)

    df["_resolved_path"] = (
        df["image_path"]
        .astype(str)
        .map(resolve_path)
    )

    selected = df[
        (df["_resolved_path"] == target)
        & (df["candidate_available"] == 1)
        & (df["component_rank"] > 0)
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


def select_case(
    inventory: pd.DataFrame,
    selection: dict,
) -> pd.Series:
    part = inventory[
        (
            inventory["case_type"]
            == selection["case_type"]
        )
        & (
            inventory["category"]
            == selection["category"]
        )
        & (
            pd.to_numeric(
                inventory["case_rank"],
                errors="coerce",
            )
            == selection["case_rank"]
        )
    ].copy()

    if len(part) != 1:
        raise RuntimeError(
            "Expected exactly one case for "
            f"{selection}, found {len(part)}."
        )

    return part.iloc[0]


def metric_text(
    row: pd.Series,
    case_type: str,
) -> str:
    if case_type == "gate_off_protected_from_harmful_vlm":
        value = float(
            row["naive_harm_vs_detector"]
        )
        return f"Naive harm avoided={value:+.3f}"

    if case_type == "gate_off_missed_vlm_opportunity":
        value = float(
            row["vlm_directional_advantage"]
        )
        return f"Missed VLM advantage={value:+.3f}"

    if case_type == "gate_on_helpful":
        value = float(
            row["srb_benefit_vs_detector"]
        )
        return f"SRB benefit={value:+.3f}"

    if case_type == "gate_on_harmful":
        value = -float(
            row["srb_benefit_vs_detector"]
        )
        return f"SRB harm={value:+.3f}"

    raise ValueError(case_type)


def panel_title(
    row: pd.Series,
    selection: dict,
) -> str:
    label = (
        "anomaly"
        if int(row["Y"]) == 1
        else "normal"
    )

    return (
        f"({selection['panel']}) {selection['headline']} "
        f"| {row['category']}, {label}\n"
        f"D={row['D']:.3f}  M={row['M']:.3f}  "
        f"Q={row['Q']:.3f}  SRB={row['score_S1']:.3f}  "
        f"{metric_text(row, selection['case_type'])}"
    )


def add_candidate_boxes(
    axis,
    boxes: list[dict],
) -> None:
    for box in boxes:
        width = (
            box["x2"] - box["x1"]
        )

        height = (
            box["y2"] - box["y1"]
        )

        rectangle = Rectangle(
            (box["x1"], box["y1"]),
            width,
            height,
            fill=False,
            linewidth=2.5,
        )

        axis.add_patch(rectangle)

        axis.text(
            box["x1"],
            max(0, box["y1"] - 5),
            f"R{box['rank']}",
            fontsize=9,
            bbox={
                "alpha": 0.65,
                "pad": 1.5,
            },
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
        "vlm_directional_advantage",
    }

    missing = sorted(
        required - set(inventory.columns)
    )

    if missing:
        raise RuntimeError(
            f"Inventory missing columns: {missing}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_CAPTION.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 10),
        constrained_layout=True,
    )

    axes_flat = axes.ravel()
    manifest_rows = []

    for axis, selection in zip(
        axes_flat,
        SELECTIONS,
    ):
        row = select_case(
            inventory,
            selection,
        )

        image_path = Path(
            str(row["image_path"])
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        with Image.open(image_path) as image:
            axis.imshow(
                image.convert("RGB")
            )

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
            panel_title(
                row,
                selection,
            ),
            fontsize=10,
        )

        axis.set_axis_off()

        manifest_rows.append(
            {
                "panel": selection["panel"],
                "headline": selection["headline"],
                "case_type": selection["case_type"],
                "case_rank": int(
                    row["case_rank"]
                ),
                "category": row["category"],
                "image_path": str(
                    image_path
                ),
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
                "vlm_directional_advantage": float(
                    row[
                        "vlm_directional_advantage"
                    ]
                ),
                "candidate_boxes": len(
                    boxes
                ),
            }
        )

    figure.suptitle(
        "Reliability boundaries of SRB-QCR on MVTec AD",
        fontsize=15,
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

    caption = """**Qualitative analysis of the frozen SRB-QCR gate on MVTec AD.**
Each panel reports the detector score \(D\), crop-based VLM score \(M\),
candidate quality \(Q\), and final SRB-QCR score. The cases illustrate:
(A) successful rejection of harmful VLM evidence, (B) a conservative missed
opportunity, (C) successful acceptance of useful VLM evidence, and
(D) erroneous acceptance of harmful VLM evidence. Candidate boxes are proposed
by PatchCore. For gate-off cases, the displayed \(M\) is taken retrospectively
from the full-inference audit and is not computed during actual selective
deployment.
"""

    OUT_CAPTION.write_text(
        caption,
        encoding="utf-8",
        newline="\n",
    )

    print(
        "===== D7d MAIN FIGURE MANIFEST ====="
    )

    print(
        manifest[
            [
                "panel",
                "headline",
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
    print("[DONE]", OUT_CAPTION)


if __name__ == "__main__":
    main()
