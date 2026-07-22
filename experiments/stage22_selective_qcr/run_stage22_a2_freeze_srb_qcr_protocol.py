from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

INPUT = (
    ROOT
    / "results/stage9_qcr_u"
    / "stage9_a1_qcr_u_fusion_predictions.csv"
)

OUT_JSON = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_a2_srb_qcr_frozen_protocol.json"
)

OUT_REPORT = (
    ROOT
    / "docs/stage22_selective_qcr"
    / "stage22_a2_srb_qcr_frozen_protocol.md"
)

REQUIRED_COLUMNS = [
    "backbone",
    "dataset",
    "category",
    "strategy",
    "eval_mode",
    "image_key",
    "is_anomaly_final",
    "fallback",
    "has_candidate",
    "num_candidates",
    "vlm_score_norm",
    "detector_score_norm",
    "candidate_quality_norm",
]

OPTIONAL_COLUMNS = [
    "high_high_consistency",
    "image_path",
    "gt_label",
    "defect_type",
]

DEDUP_KEYS = [
    "backbone",
    "dataset",
    "category",
    "strategy",
    "eval_mode",
    "image_key",
]

PRIMARY_SCOPE = {
    "dataset": "VisA",
    "strategy": "inspection_binary",
    "eval_mode": "crop_topk_ensemble",
}

GRID = {
    "w_max": [0.15, 0.25, 0.35],
    "tau_q": [0.25, 0.50, 0.75],
    "tau_delta": [0.25, 0.50, 0.75],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def read_bool_summary(series: pd.Series) -> list[str]:
    values = (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    return sorted(values)


def numeric_summary(
    df: pd.DataFrame,
    column: str,
) -> dict:
    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    return {
        "non_null": int(values.notna().sum()),
        "minimum": (
            float(values.min())
            if values.notna().any()
            else None
        ),
        "maximum": (
            float(values.max())
            if values.notna().any()
            else None
        ),
        "mean": (
            float(values.mean())
            if values.notna().any()
            else None
        ),
        "unique": int(values.nunique(dropna=True)),
    }


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    original_rows = len(df)

    base = (
        df[
            REQUIRED_COLUMNS
            + [
                column
                for column in OPTIONAL_COLUMNS
                if column in df.columns
            ]
        ]
        .drop_duplicates(subset=DEDUP_KEYS)
        .reset_index(drop=True)
    )

    primary = base.copy()

    for column, value in PRIMARY_SCOPE.items():
        primary = primary[
            primary[column].astype(str) == value
        ]

    if primary.empty:
        raise RuntimeError(
            "The frozen VisA primary protocol is empty."
        )

    patchcore_backbones = sorted(
        value
        for value in primary["backbone"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if "patchcore" in value.lower()
    )

    if not patchcore_backbones:
        raise RuntimeError(
            "No PatchCore backbone found in the "
            "frozen VisA primary protocol."
        )

    categories = sorted(
        primary["category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(categories) < 2:
        raise RuntimeError(
            "At least two categories are required "
            "for category-LOCO selection."
        )

    labels = sorted(
        pd.to_numeric(
            primary["is_anomaly_final"],
            errors="coerce",
        )
        .dropna()
        .unique()
        .tolist()
    )

    if not set(labels).issubset({0, 1, 0.0, 1.0}):
        raise RuntimeError(
            f"Unexpected binary labels: {labels}"
        )

    signal_summary = {
        column: numeric_summary(primary, column)
        for column in [
            "detector_score_norm",
            "vlm_score_norm",
            "candidate_quality_norm",
            "num_candidates",
        ]
    }

    if "high_high_consistency" in primary.columns:
        signal_summary["high_high_consistency"] = (
            numeric_summary(
                primary,
                "high_high_consistency",
            )
        )

    protocol = {
        "protocol_id": "stage22_a2_srb_qcr_v1",
        "status": "frozen_before_stage22_b_results",
        "method_name": (
            "Selective Reliability-Bounded QCR"
        ),
        "short_name": "SRB-QCR",
        "source": {
            "path": str(INPUT.relative_to(ROOT)),
            "sha256": sha256(INPUT),
            "original_rows": original_rows,
            "deduplicated_rows": len(base),
            "required_columns": REQUIRED_COLUMNS,
            "optional_columns_present": [
                column
                for column in OPTIONAL_COLUMNS
                if column in base.columns
            ],
        },
        "signal_mapping": {
            "D": "detector_score_norm",
            "M": "vlm_score_norm",
            "Q": "candidate_quality_norm",
            "Y": "is_anomaly_final",
            "sample_id": "image_key",
            "group": "category",
            "old_K_diagnostic_only": (
                "high_high_consistency"
                if "high_high_consistency"
                in base.columns
                else None
            ),
        },
        "frozen_primary_scope": PRIMARY_SCOPE,
        "development_backbone_rule": (
            "case-insensitive backbone name "
            "containing 'patchcore'"
        ),
        "development_backbones_found": (
            patchcore_backbones
        ),
        "categories_found": categories,
        "method": {
            "pre_gate": (
                "G_pre = I(has_candidate) * "
                "I(not fallback) * I(Q >= tau_q)"
            ),
            "agreement": (
                "A = clip(1 - abs(D-M)/tau_delta, 0, 1)"
            ),
            "weight": (
                "w = w_max * G_pre * Q * A"
            ),
            "score": (
                "S_SRB = D + w * (M-D)"
            ),
            "missing_candidate_rule": (
                "If no valid candidate or M is unavailable, "
                "set w=0 and S_SRB=D."
            ),
            "boundedness": [
                "0 <= w <= w_max < 0.5",
                "S_SRB lies between D and M.",
                (
                    "abs(S_SRB-D) <= "
                    "w_max * abs(M-D) <= w_max."
                ),
                (
                    "The fused score remains closer to D "
                    "than to M whenever D != M."
                ),
            ],
            "old_consistency_usage": (
                "The binary high_high_consistency signal "
                "is retained only for comparison with the "
                "old Adaptive QCR and is not used by SRB-QCR."
            ),
        },
        "hyperparameter_grid": GRID,
        "number_of_grid_configurations": (
            len(GRID["w_max"])
            * len(GRID["tau_q"])
            * len(GRID["tau_delta"])
        ),
        "selection_protocol": {
            "type": (
                "leave-one-category-out development "
                "on VisA PatchCore"
            ),
            "for_each_fold": [
                (
                    "Hold out one VisA category without using "
                    "its labels for configuration selection."
                ),
                (
                    "Evaluate all 27 configurations on the "
                    "remaining categories."
                ),
                (
                    "Compute macro mean image AUROC across "
                    "the remaining categories."
                ),
                (
                    "A configuration is eligible when its "
                    "development macro AUROC is no worse than "
                    "detector-only by more than 0.002."
                ),
                (
                    "Among eligible configurations, select "
                    "the highest development macro AUROC."
                ),
                (
                    "Treat AUROC differences <=0.001 as ties; "
                    "then prefer lower potential VLM call rate."
                ),
                (
                    "Remaining ties use, in order: lower w_max, "
                    "higher tau_q, lower tau_delta."
                ),
                (
                    "If no configuration is eligible, use "
                    "detector-only fallback for that fold."
                ),
            ],
            "global_frozen_configuration": [
                (
                    "Take the most frequently selected "
                    "configuration across held-out folds."
                ),
                (
                    "Break frequency ties by better average "
                    "development rank, then lower call rate, "
                    "then the conservative tie order."
                ),
                (
                    "Apply the resulting single configuration "
                    "unchanged to transfer and external tests."
                ),
            ],
        },
        "evaluation": {
            "primary_metric": "image AUROC",
            "secondary_metrics": [
                "image AP",
                "best F1",
                "best accuracy",
            ],
            "efficiency_metrics": [
                "potential VLM invocation rate",
                "potential VLM calls saved",
            ],
            "potential_call_definition": (
                "I(has_candidate) * I(not fallback) "
                "* I(Q >= tau_q)"
            ),
            "efficiency_caveat": (
                "Offline cached-score evaluation estimates "
                "potential calls. Actual wall-clock savings "
                "must be verified before making a runtime claim."
            ),
        },
        "comparisons": [
            "detector only",
            "crop VLM only",
            "naive detector-crop fusion",
            "old Quality-Calibrated QCR",
            "old Adaptive QCR",
            "SRB-QCR",
        ],
        "transfer_and_external_validation": [
            (
                "VisA FastFlow under the same primary protocol"
            ),
            (
                "AD2 four-category fixed protocol"
            ),
            (
                "complete MVTec AD fixed protocol when "
                "sample-level cached signals are available"
            ),
        ],
        "prohibited_actions": [
            (
                "Do not choose parameters using the held-out "
                "category labels."
            ),
            (
                "Do not reselect parameters on FastFlow, AD2, "
                "or MVTec AD."
            ),
            (
                "Do not use GT mask coverage or any label-derived "
                "candidate statistic as Q."
            ),
            (
                "Do not alter the grid after Stage 22-B results "
                "have been inspected."
            ),
            (
                "Do not claim actual runtime reduction from "
                "offline call-rate estimates alone."
            ),
        ],
        "data_audit": {
            "primary_rows": len(primary),
            "primary_categories": len(categories),
            "primary_backbones": sorted(
                primary["backbone"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            "label_values": labels,
            "fallback_values": read_bool_summary(
                primary["fallback"]
            ),
            "has_candidate_values": read_bool_summary(
                primary["has_candidate"]
            ),
            "signals": signal_summary,
        },
    }

    OUT_JSON.write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Stage 22-A2: Frozen SRB-QCR Protocol",
        "",
        "## Status",
        "",
        "- status: `frozen_before_stage22_b_results`",
        "- method: `Selective Reliability-Bounded QCR`",
        "- short name: `SRB-QCR`",
        f"- source: `{INPUT.relative_to(ROOT)}`",
        f"- source SHA-256: `{protocol['source']['sha256']}`",
        "",
        "## Signal mapping",
        "",
        "- `D = detector_score_norm`",
        "- `M = vlm_score_norm`",
        "- `Q = candidate_quality_norm`",
        "- `Y = is_anomaly_final`",
        "- sample ID: `image_key`",
        "- group: `category`",
        "",
        "## Frozen method",
        "",
        "```text",
        "G_pre = I(has_candidate) · I(not fallback) · I(Q >= tau_q)",
        "A     = clip(1 - |D-M| / tau_delta, 0, 1)",
        "w     = w_max · G_pre · Q · A",
        "S_SRB = D + w(M-D)",
        "```",
        "",
        "with:",
        "",
        "```text",
        "0 <= w <= w_max < 0.5",
        "|S_SRB-D| <= w_max |M-D| <= w_max",
        "```",
        "",
        "A missing or invalid candidate forces `w=0` and",
        "`S_SRB=D`.",
        "",
        "## Frozen grid",
        "",
        f"- `w_max`: `{GRID['w_max']}`",
        f"- `tau_q`: `{GRID['tau_q']}`",
        f"- `tau_delta`: `{GRID['tau_delta']}`",
        "- total configurations: `27`",
        "",
        "## Selection",
        "",
        "- development source: VisA PatchCore",
        "- selection: leave one category out",
        "- primary criterion: development macro image AUROC",
        "- detector non-inferiority tolerance: `0.002`",
        "- AUROC tie tolerance: `0.001`",
        "- tie priority: lower call rate, lower `w_max`,",
        "  higher `tau_q`, lower `tau_delta`",
        "- no eligible configuration: detector-only fallback",
        "",
        "## Frozen transfer tests",
        "",
        "- VisA FastFlow",
        "- AD2 four categories",
        "- complete MVTec AD when cached signals are available",
        "",
        "## Data audit",
        "",
        f"- original rows: `{original_rows}`",
        f"- deduplicated rows: `{len(base)}`",
        f"- primary rows: `{len(primary)}`",
        f"- primary categories: `{len(categories)}`",
        "- PatchCore development backbone(s): "
        + ", ".join(
            f"`{value}`"
            for value in patchcore_backbones
        ),
        "",
        "## Claim restriction",
        "",
        "Offline evaluation may report potential VLM call rate.",
        "It may not claim measured runtime acceleration until",
        "an execution-level timing experiment is completed.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print("===== STAGE 22-A2 PROTOCOL FROZEN =====")
    print("source:", INPUT)
    print("source SHA256:", protocol["source"]["sha256"])
    print("original rows:", original_rows)
    print("deduplicated rows:", len(base))
    print("primary rows:", len(primary))
    print("categories:", len(categories))
    print("PatchCore backbones:", patchcore_backbones)
    print("grid configurations: 27")
    print()
    print("[DONE]", OUT_JSON)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
