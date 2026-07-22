from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

B1_SCRIPT = (
    ROOT
    / "experiments/stage22_selective_qcr"
    / "run_stage22_b1_visa_patchcore_loco_selection.py"
)

STAGE18_SCRIPT = (
    ROOT
    / "experiments/stage18_ad2_qcr_ablation"
    / "run_stage18_b2_ad2_qcr_ablation.py"
)

PROTOCOL_PATH = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_a2_srb_qcr_frozen_protocol.json"
)

CONFIG_PATH = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_b1_visa_patchcore_global_config.json"
)

OUT_DIR = ROOT / "results/stage22_selective_qcr"
DOC_DIR = ROOT / "docs/stage22_selective_qcr"

OUT_PRED = (
    OUT_DIR
    / "stage22_b2b_ad2_frozen_predictions.csv"
)

OUT_PER_CATEGORY = (
    OUT_DIR
    / "stage22_b2b_ad2_per_category.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "stage22_b2b_ad2_summary.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "stage22_b2b_ad2_transfer_metadata.json"
)

OUT_REPORT = (
    DOC_DIR
    / "stage22_b2b_ad2_frozen_transfer_report.md"
)

EXPECTED_CATEGORIES = [
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "walnuts",
]

VARIANTS = {
    "D0": (
        "Detector only",
        "score_D0",
    ),
    "V3": (
        "Naive detector-crop fusion",
        "score_V3",
    ),
    "V4": (
        "Old Quality-Calibrated QCR",
        "score_V4",
    ),
    "V6": (
        "Old Adaptive QCR",
        "score_V6",
    ),
    "S1": (
        "SRB-QCR frozen transfer",
        "score_S1",
    ),
}


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_frozen_config() -> dict:
    for path in [
        PROTOCOL_PATH,
        CONFIG_PATH,
        B1_SCRIPT,
        STAGE18_SCRIPT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    config_payload = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    selected = config_payload[
        "global_configuration"
    ]["selected"]

    config = {
        "w_max": float(
            selected["w_max"]
        ),
        "q_quantile": float(
            selected["q_quantile"]
        ),
        "tau_delta": float(
            selected["tau_delta"]
        ),
    }

    expected = {
        "w_max": 0.35,
        "q_quantile": 0.25,
        "tau_delta": 0.75,
    }

    if config != expected:
        raise RuntimeError(
            "Frozen configuration does not match "
            "the preregistered Stage 22-B1 result.\n"
            f"Expected: {expected}\n"
            f"Actual:   {config}"
        )

    return {
        "protocol": protocol,
        "config": config,
    }


def prepare_ad2(stage18) -> pd.DataFrame:
    # Uses the already-existing Stage 18 assembly logic.
    # No detector or VLM inference is executed.
    pred = stage18.build_predictions().copy()

    required = [
        "category",
        "image_path",
        "gt_binary",
        "D",
        "M",
        "Q",
        "V0_detector_only",
        "V3_naive_detector_crop_fusion",
        "V4_quality_calibrated_qcr",
        "V6_adaptive_qcr_refinement",
    ]

    missing = [
        column
        for column in required
        if column not in pred.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing Stage 18 columns: {missing}"
        )

    pred = pred[
        pred["category"].isin(
            EXPECTED_CATEGORIES
        )
    ].copy()

    pred["Y"] = pd.to_numeric(
        pred["gt_binary"],
        errors="coerce",
    )

    pred["D"] = pd.to_numeric(
        pred["D"],
        errors="coerce",
    )

    pred["M"] = pd.to_numeric(
        pred["M"],
        errors="coerce",
    )

    pred["Q"] = pd.to_numeric(
        pred["Q"],
        errors="coerce",
    )

    pred["M_available"] = np.isfinite(
        pred["M"]
    )

    if "num_candidates" in pred.columns:
        num_candidates = pd.to_numeric(
            pred["num_candidates"],
            errors="coerce",
        ).fillna(0.0)

        pred["has_candidate_bool"] = (
            num_candidates > 0
        )
    else:
        pred["has_candidate_bool"] = (
            pred["M_available"]
        )

    # Stage 18 AD2 assembly has no separate fallback flag.
    pred["fallback_bool"] = False

    pred = pred[
        np.isfinite(pred["Y"])
        & np.isfinite(pred["D"])
    ].copy()

    pred["Y"] = pred["Y"].astype(int)

    pred["M"] = pred["M"].fillna(0.0)
    pred["Q"] = pred["Q"].fillna(0.0)

    for column in ["D", "M", "Q"]:
        pred[column] = pred[column].clip(
            lower=0.0,
            upper=1.0,
        )

    pred["score_D0"] = pd.to_numeric(
        pred["V0_detector_only"],
        errors="coerce",
    )

    pred["score_V3"] = pd.to_numeric(
        pred[
            "V3_naive_detector_crop_fusion"
        ],
        errors="coerce",
    )

    pred["score_V4"] = pd.to_numeric(
        pred[
            "V4_quality_calibrated_qcr"
        ],
        errors="coerce",
    )

    pred["score_V6"] = pd.to_numeric(
        pred[
            "V6_adaptive_qcr_refinement"
        ],
        errors="coerce",
    )

    return pred.reset_index(drop=True)


def target_q_threshold(
    pred: pd.DataFrame,
    quantile: float,
) -> tuple[float, int]:
    eligible = (
        pred["has_candidate_bool"]
        & ~pred["fallback_bool"]
        & pred["M_available"]
        & np.isfinite(pred["Q"])
    )

    values = pred.loc[
        eligible,
        "Q",
    ].dropna()

    if values.empty:
        return 1.0, 0

    return (
        float(values.quantile(quantile)),
        len(values),
    )


def macro_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.mean())


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    frozen = load_frozen_config()
    config = frozen["config"]

    b1 = load_module(
        "stage22_b1_module",
        B1_SCRIPT,
    )

    stage18 = load_module(
        "stage18_b2_module",
        STAGE18_SCRIPT,
    )

    pred = prepare_ad2(stage18)

    categories = sorted(
        pred["category"]
        .astype(str)
        .unique()
        .tolist()
    )

    q_threshold, eligible_q_count = (
        target_q_threshold(
            pred,
            config["q_quantile"],
        )
    )

    print(
        "===== STAGE 22-B2b AD2 VALIDATION ====="
    )
    print("rows:", len(pred))
    print("categories:", categories)
    print(
        "labels:",
        sorted(pred["Y"].unique().tolist()),
    )
    print(
        "w_max:",
        config["w_max"],
    )
    print(
        "q_quantile:",
        config["q_quantile"],
    )
    print(
        "target q threshold:",
        q_threshold,
    )
    print(
        "eligible q samples:",
        eligible_q_count,
    )
    print(
        "tau_delta:",
        config["tau_delta"],
    )
    print()

    if categories != sorted(
        EXPECTED_CATEGORIES
    ):
        raise RuntimeError(
            "Unexpected AD2 categories.\n"
            f"Expected: {sorted(EXPECTED_CATEGORIES)}\n"
            f"Actual:   {categories}"
        )

    if sorted(
        pred["Y"].unique().tolist()
    ) != [0, 1]:
        raise RuntimeError(
            "Expected binary labels [0, 1]."
        )

    if args.validate_only:
        print(
            "[OK] Validation-only run passed."
        )
        return

    scored = b1.apply_srb(
        pred,
        w_max=config["w_max"],
        q_threshold=q_threshold,
        tau_delta=config["tau_delta"],
    )

    per_category_rows = []

    for category, group in scored.groupby(
        "category",
        dropna=False,
    ):
        row = {
            "category": str(category),
            "num_images": len(group),
            "num_normal": int(
                (group["Y"] == 0).sum()
            ),
            "num_anomaly": int(
                (group["Y"] == 1).sum()
            ),
            "potential_call_rate": float(
                group[
                    "srb_pre_gate"
                ].mean()
            ),
            "active_weight_rate": float(
                group[
                    "srb_active"
                ].mean()
            ),
        }

        for variant_id, (
            _,
            score_col,
        ) in VARIANTS.items():
            metrics = b1.evaluate_binary(
                group["Y"],
                group[score_col],
            )

            row[
                f"{variant_id}_auroc"
            ] = metrics["auroc"]

            row[
                f"{variant_id}_ap"
            ] = metrics["ap"]

            row[
                f"{variant_id}_best_f1"
            ] = metrics["best_f1"]

            row[
                f"{variant_id}_best_accuracy"
            ] = metrics["best_accuracy"]

        row["S1_minus_D0"] = (
            row["S1_auroc"]
            - row["D0_auroc"]
        )

        row["S1_minus_V3"] = (
            row["S1_auroc"]
            - row["V3_auroc"]
        )

        row["S1_minus_V4"] = (
            row["S1_auroc"]
            - row["V4_auroc"]
        )

        row["S1_minus_V6"] = (
            row["S1_auroc"]
            - row["V6_auroc"]
        )

        per_category_rows.append(row)

    per_category = pd.DataFrame(
        per_category_rows
    ).sort_values(
        "S1_minus_D0",
        ascending=False,
    )

    summary_rows = []

    for variant_id, (
        method,
        score_col,
    ) in VARIANTS.items():
        category_auroc_col = (
            f"{variant_id}_auroc"
        )

        category_ap_col = (
            f"{variant_id}_ap"
        )

        category_f1_col = (
            f"{variant_id}_best_f1"
        )

        pooled = b1.evaluate_binary(
            scored["Y"],
            scored[score_col],
        )

        summary_rows.append(
            {
                "variant_id": variant_id,
                "method": method,
                "mean_image_auroc": (
                    macro_mean(
                        per_category,
                        category_auroc_col,
                    )
                ),
                "pooled_image_auroc": (
                    pooled["auroc"]
                ),
                "mean_image_ap": (
                    macro_mean(
                        per_category,
                        category_ap_col,
                    )
                ),
                "mean_best_f1": (
                    macro_mean(
                        per_category,
                        category_f1_col,
                    )
                ),
                "num_categories": len(
                    per_category
                ),
                "potential_call_rate": (
                    float(
                        scored[
                            "srb_pre_gate"
                        ].mean()
                    )
                    if variant_id == "S1"
                    else np.nan
                ),
                "active_weight_rate": (
                    float(
                        scored[
                            "srb_active"
                        ].mean()
                    )
                    if variant_id == "S1"
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    scores = {
        row["variant_id"]: float(
            row["mean_image_auroc"]
        )
        for _, row in summary.iterrows()
    }

    srb_vs_detector = (
        scores["S1"] - scores["D0"]
    )

    category_wins = int(
        (
            per_category["S1_minus_D0"]
            > 0
        ).sum()
    )

    worst_delta = float(
        per_category[
            "S1_minus_D0"
        ].min()
    )

    # Frozen interpretation rule declared before inspecting results.
    if (
        srb_vs_detector >= 0
        and category_wins >= 3
        and worst_delta >= -0.005
    ):
        decision = (
            "retain_as_cross_dataset_deployment_mode"
        )
    elif (
        srb_vs_detector >= -0.005
        and worst_delta >= -0.010
    ):
        decision = (
            "retain_only_as_noninferior_efficiency_mode"
        )
    else:
        decision = (
            "reject_cross_dataset_reliability_claim"
        )

    metadata = {
        "status": "success",
        "selection_source": (
            "VisA PatchCore category-LOCO"
        ),
        "transfer_target": (
            "AD2 four categories"
        ),
        "target_labels_used_for_parameters": False,
        "configuration": config,
        "target_quality_rule": (
            "25th percentile of unlabeled "
            "eligible AD2 Q values"
        ),
        "target_q_threshold": q_threshold,
        "eligible_q_samples": (
            eligible_q_count
        ),
        "rows": len(scored),
        "categories": categories,
        "adapter": {
            "has_candidate": (
                "num_candidates > 0, or "
                "finite M if unavailable"
            ),
            "fallback": False,
        },
        "frozen_decision_rule": {
            "strong_retain": (
                "mean delta >= 0, wins >= 3/4, "
                "worst category delta >= -0.005"
            ),
            "noninferior_only": (
                "mean delta >= -0.005 and "
                "worst category delta >= -0.010"
            ),
        },
        "decision": decision,
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_csv(
        OUT_PRED,
        index=False,
        lineterminator="\n",
    )

    per_category.to_csv(
        OUT_PER_CATEGORY,
        index=False,
        lineterminator="\n",
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        lineterminator="\n",
    )

    OUT_METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Stage 22-B2b: Frozen SRB-QCR Transfer to AD2",
        "",
        "## Protocol",
        "",
        "- parameter source: `VisA PatchCore category-LOCO`",
        "- transfer target: `AD2 four categories`",
        "- AD2 labels used for parameter selection: `none`",
        f"- `w_max = {config['w_max']}`",
        f"- `q_quantile = {config['q_quantile']}`",
        f"- target unlabeled Q threshold: `{q_threshold:.6f}`",
        f"- `tau_delta = {config['tau_delta']}`",
        "",
        "## Summary",
        "",
        "| Variant | Mean category AUROC | Pooled AUROC | Mean AP | Mean Best F1 | Potential call rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        call_rate = (
            f"{row['potential_call_rate']:.4f}"
            if pd.notna(
                row["potential_call_rate"]
            )
            else "-"
        )

        lines.append(
            f"| {row['method']} | "
            f"{row['mean_image_auroc']:.4f} | "
            f"{row['pooled_image_auroc']:.4f} | "
            f"{row['mean_image_ap']:.4f} | "
            f"{row['mean_best_f1']:.4f} | "
            f"{call_rate} |"
        )

    lines += [
        "",
        "## Frozen-transfer deltas",
        "",
        f"- SRB minus detector: `{scores['S1'] - scores['D0']:+.4f}`",
        f"- SRB minus naive: `{scores['S1'] - scores['V3']:+.4f}`",
        f"- SRB minus old Quality QCR: `{scores['S1'] - scores['V4']:+.4f}`",
        f"- SRB minus old Adaptive QCR: `{scores['S1'] - scores['V6']:+.4f}`",
        f"- categories SRB > detector: `{category_wins}/4`",
        f"- worst category delta vs detector: `{worst_delta:+.4f}`",
        f"- potential calls saved: `{1.0 - float(scored['srb_pre_gate'].mean()):.4f}`",
        f"- frozen decision: `{decision}`",
        "",
        "Potential call saving remains an offline estimate.",
        "It is not a measured runtime speedup.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print(
        "===== STAGE 22-B2b SUCCESS ====="
    )
    print()

    print(
        summary[
            [
                "variant_id",
                "method",
                "mean_image_auroc",
                "pooled_image_auroc",
                "mean_image_ap",
                "mean_best_f1",
                "potential_call_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "SRB - detector:",
        f"{scores['S1'] - scores['D0']:+.6f}",
    )
    print(
        "SRB - naive:",
        f"{scores['S1'] - scores['V3']:+.6f}",
    )
    print(
        "SRB - old quality:",
        f"{scores['S1'] - scores['V4']:+.6f}",
    )
    print(
        "SRB - old adaptive:",
        f"{scores['S1'] - scores['V6']:+.6f}",
    )
    print(
        "category wins vs detector:",
        f"{category_wins}/4",
    )
    print(
        "worst category delta vs detector:",
        f"{worst_delta:+.6f}",
    )
    print(
        "potential call rate:",
        f"{scored['srb_pre_gate'].mean():.6f}",
    )
    print(
        "potential calls saved:",
        f"{1.0 - scored['srb_pre_gate'].mean():.6f}",
    )
    print(
        "frozen decision:",
        decision,
    )
    print()
    print("[DONE]", OUT_PRED)
    print("[DONE]", OUT_PER_CATEGORY)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_METADATA)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
