from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

OUT_DIR = ROOT / "results/stage22_selective_qcr"
DOC_DIR = ROOT / "docs/stage22_selective_qcr"

OUT_SUMMARY = (
    OUT_DIR
    / "stage22_c1_paired_bootstrap_summary.csv"
)

OUT_DISTRIBUTIONS = (
    OUT_DIR
    / "stage22_c1_paired_bootstrap_distributions.npz"
)

OUT_METADATA = (
    OUT_DIR
    / "stage22_c1_paired_bootstrap_metadata.json"
)

OUT_REPORT = (
    DOC_DIR
    / "stage22_c1_paired_bootstrap_report.md"
)

PROTOCOLS = [
    {
        "protocol": "visa_patchcore_loco",
        "path": (
            OUT_DIR
            / "stage22_b1_visa_patchcore_loco_predictions.csv"
        ),
        "aggregation": "mean_pooled_auc_over_backbones",
    },
    {
        "protocol": "visa_fastflow_frozen_transfer",
        "path": (
            OUT_DIR
            / "stage22_b2a_visa_fastflow_frozen_predictions.csv"
        ),
        "aggregation": "mean_pooled_auc_over_backbones",
    },
    {
        "protocol": "ad2_frozen_transfer",
        "path": (
            OUT_DIR
            / "stage22_b2b_ad2_frozen_predictions.csv"
        ),
        "aggregation": "mean_category_auc",
    },
]

METHODS = {
    "D0": "score_D0",
    "V3": "score_V3",
    "V4": "score_V4",
    "V6": "score_V6",
    "S1": "score_S1",
}

PAIRS = [
    {
        "comparison": "SRB-QCR vs detector",
        "left": "S1",
        "right": "D0",
        "claim_role": "reliability_mode_primary",
    },
    {
        "comparison": "Quality QCR vs naive fusion",
        "left": "V4",
        "right": "V3",
        "claim_role": "high_accuracy_mode_primary",
    },
    {
        "comparison": "Adaptive QCR vs Quality QCR",
        "left": "V6",
        "right": "V4",
        "claim_role": "adaptive_refinement_secondary",
    },
    {
        "comparison": "SRB-QCR vs Quality QCR",
        "left": "S1",
        "right": "V4",
        "claim_role": "accuracy_efficiency_tradeoff",
    },
]

SEED = 20260722
DEFAULT_ITERATIONS = 5000


def safe_auc(
    y: np.ndarray,
    score: np.ndarray,
) -> float:
    valid = (
        np.isfinite(y)
        & np.isfinite(score)
    )

    y_valid = y[valid]
    score_valid = score[valid]

    if (
        len(y_valid) == 0
        or len(np.unique(y_valid)) < 2
    ):
        return float("nan")

    return float(
        roc_auc_score(
            y_valid,
            score_valid,
        )
    )


def load_protocol(
    specification: dict,
) -> pd.DataFrame:
    path = specification["path"]

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "category",
        "Y",
        *METHODS.values(),
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path} missing columns: {missing}"
        )

    result = df.copy()

    result["category"] = (
        result["category"].astype(str)
    )

    result["Y"] = pd.to_numeric(
        result["Y"],
        errors="coerce",
    )

    for score_column in METHODS.values():
        result[score_column] = pd.to_numeric(
            result[score_column],
            errors="coerce",
        )

    if (
        specification["aggregation"]
        == "mean_pooled_auc_over_backbones"
    ):
        if "backbone" not in result.columns:
            result["backbone"] = (
                specification["protocol"]
            )
        else:
            result["backbone"] = (
                result["backbone"].astype(str)
            )
    else:
        result["backbone"] = "not_applicable"

    result = result[
        np.isfinite(result["Y"])
    ].copy()

    result["Y"] = result["Y"].astype(int)

    if sorted(
        result["Y"].unique().tolist()
    ) != [0, 1]:
        raise RuntimeError(
            f"{path} does not contain labels [0, 1]."
        )

    return result.reset_index(drop=True)


def make_strata(
    df: pd.DataFrame,
    aggregation: str,
) -> list[dict]:
    if (
        aggregation
        == "mean_pooled_auc_over_backbones"
    ):
        group_columns = [
            "backbone",
            "category",
        ]
    elif aggregation == "mean_category_auc":
        group_columns = ["category"]
    else:
        raise ValueError(
            f"Unknown aggregation: {aggregation}"
        )

    strata = []

    grouper = (
        group_columns[0]
        if len(group_columns) == 1
        else group_columns
    )

    for key, group in df.groupby(
        grouper,
        sort=True,
        dropna=False,
    ):
        indices = group.index.to_numpy(
            dtype=int
        )

        y = df.loc[indices, "Y"].to_numpy(
            dtype=int
        )

        positive = indices[y == 1]
        negative = indices[y == 0]

        if (
            len(positive) == 0
            or len(negative) == 0
        ):
            raise RuntimeError(
                f"Invalid bootstrap stratum {key}: "
                f"positive={len(positive)}, "
                f"negative={len(negative)}"
            )

        if not isinstance(key, tuple):
            key = (key,)

        strata.append(
            {
                "key": tuple(str(value) for value in key),
                "positive": positive,
                "negative": negative,
            }
        )

    return strata


def evaluate_indices(
    df: pd.DataFrame,
    indices_by_stratum: list[np.ndarray],
    strata: list[dict],
    aggregation: str,
) -> dict[str, float]:
    score_results = {
        method_id: []
        for method_id in METHODS
    }

    if (
        aggregation
        == "mean_pooled_auc_over_backbones"
    ):
        backbone_indices: dict[
            str,
            list[np.ndarray],
        ] = {}

        for sampled_indices, stratum in zip(
            indices_by_stratum,
            strata,
        ):
            backbone = stratum["key"][0]

            backbone_indices.setdefault(
                backbone,
                [],
            ).append(sampled_indices)

        for index_parts in (
            backbone_indices.values()
        ):
            indices = np.concatenate(index_parts)

            y = df.loc[
                indices,
                "Y",
            ].to_numpy(dtype=int)

            for method_id, score_column in (
                METHODS.items()
            ):
                score = df.loc[
                    indices,
                    score_column,
                ].to_numpy(dtype=float)

                score_results[method_id].append(
                    safe_auc(y, score)
                )

    elif aggregation == "mean_category_auc":
        for sampled_indices in (
            indices_by_stratum
        ):
            y = df.loc[
                sampled_indices,
                "Y",
            ].to_numpy(dtype=int)

            for method_id, score_column in (
                METHODS.items()
            ):
                score = df.loc[
                    sampled_indices,
                    score_column,
                ].to_numpy(dtype=float)

                score_results[method_id].append(
                    safe_auc(y, score)
                )

    else:
        raise ValueError(aggregation)

    output = {}

    for method_id, values in (
        score_results.items()
    ):
        values_array = np.asarray(
            values,
            dtype=float,
        )

        output[method_id] = float(
            np.nanmean(values_array)
        )

    return output


def point_scores(
    df: pd.DataFrame,
    strata: list[dict],
    aggregation: str,
) -> dict[str, float]:
    original_indices = []

    for stratum in strata:
        original_indices.append(
            np.concatenate(
                [
                    stratum["positive"],
                    stratum["negative"],
                ]
            )
        )

    return evaluate_indices(
        df=df,
        indices_by_stratum=original_indices,
        strata=strata,
        aggregation=aggregation,
    )


def bootstrap_scores(
    df: pd.DataFrame,
    strata: list[dict],
    aggregation: str,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    distributions = {
        method_id: np.empty(
            iterations,
            dtype=np.float64,
        )
        for method_id in METHODS
    }

    for iteration in range(iterations):
        sampled_indices = []

        for stratum in strata:
            sampled_positive = rng.choice(
                stratum["positive"],
                size=len(stratum["positive"]),
                replace=True,
            )

            sampled_negative = rng.choice(
                stratum["negative"],
                size=len(stratum["negative"]),
                replace=True,
            )

            sampled_indices.append(
                np.concatenate(
                    [
                        sampled_positive,
                        sampled_negative,
                    ]
                )
            )

        scores = evaluate_indices(
            df=df,
            indices_by_stratum=sampled_indices,
            strata=strata,
            aggregation=aggregation,
        )

        for method_id in METHODS:
            distributions[method_id][
                iteration
            ] = scores[method_id]

        if (
            iteration == 0
            or (iteration + 1) % 500 == 0
            or iteration + 1 == iterations
        ):
            print(
                f"  bootstrap "
                f"{iteration + 1}/{iterations}"
            )

    return distributions


def summarize_delta(
    delta: np.ndarray,
) -> dict:
    valid = delta[np.isfinite(delta)]

    if len(valid) == 0:
        raise RuntimeError(
            "Empty bootstrap delta distribution."
        )

    ci_low, ci_high = np.quantile(
        valid,
        [0.025, 0.975],
    )

    positive_probability = float(
        np.mean(valid > 0)
    )

    nonnegative_probability = float(
        np.mean(valid >= 0)
    )

    p_lower = (
        np.count_nonzero(valid <= 0) + 1
    ) / (len(valid) + 1)

    p_upper = (
        np.count_nonzero(valid >= 0) + 1
    ) / (len(valid) + 1)

    p_two_sided = float(
        min(
            1.0,
            2.0 * min(p_lower, p_upper),
        )
    )

    return {
        "bootstrap_mean_delta": float(
            np.mean(valid)
        ),
        "bootstrap_median_delta": float(
            np.median(valid)
        ),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "probability_delta_gt_zero": (
            positive_probability
        ),
        "probability_delta_ge_zero": (
            nonnegative_probability
        ),
        "bootstrap_two_sided_p": (
            p_two_sided
        ),
        "ci_strictly_positive": bool(
            ci_low > 0
        ),
        "ci_contains_zero": bool(
            ci_low <= 0 <= ci_high
        ),
        "noninferior_at_minus_0_005": bool(
            ci_low > -0.005
        ),
    }


def safe_key(value: str) -> str:
    return re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        value,
    ).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.iterations < 100:
        raise ValueError(
            "At least 100 bootstrap iterations "
            "are required."
        )

    loaded = []

    print("===== STAGE 22-C1 DATA VALIDATION =====")

    for specification in PROTOCOLS:
        df = load_protocol(specification)

        strata = make_strata(
            df,
            specification["aggregation"],
        )

        points = point_scores(
            df,
            strata,
            specification["aggregation"],
        )

        loaded.append(
            {
                "specification": specification,
                "data": df,
                "strata": strata,
                "point_scores": points,
            }
        )

        print()
        print(
            "protocol:",
            specification["protocol"],
        )
        print("rows:", len(df))
        print(
            "categories:",
            df["category"].nunique(),
        )
        print(
            "backbones:",
            df["backbone"].nunique(),
        )
        print("strata:", len(strata))
        print(
            "point scores:",
            {
                key: round(value, 6)
                for key, value
                in points.items()
            },
        )

    if args.validate_only:
        print()
        print("[OK] Validation-only run passed.")
        return

    rng = np.random.default_rng(SEED)

    summary_rows = []
    saved_arrays = {}

    for item in loaded:
        specification = item["specification"]
        protocol_name = specification["protocol"]

        print()
        print(
            "===== BOOTSTRAP:",
            protocol_name,
            "=====",
        )

        distributions = bootstrap_scores(
            df=item["data"],
            strata=item["strata"],
            aggregation=(
                specification["aggregation"]
            ),
            iterations=args.iterations,
            rng=rng,
        )

        for method_id, values in (
            distributions.items()
        ):
            saved_arrays[
                f"{safe_key(protocol_name)}"
                f"__{method_id}"
            ] = values

        for pair in PAIRS:
            left_id = pair["left"]
            right_id = pair["right"]

            delta = (
                distributions[left_id]
                - distributions[right_id]
            )

            point_delta = (
                item["point_scores"][left_id]
                - item["point_scores"][right_id]
            )

            statistics = summarize_delta(delta)

            summary_rows.append(
                {
                    "protocol": protocol_name,
                    "aggregation": (
                        specification[
                            "aggregation"
                        ]
                    ),
                    "comparison": (
                        pair["comparison"]
                    ),
                    "claim_role": (
                        pair["claim_role"]
                    ),
                    "left_method": left_id,
                    "right_method": right_id,
                    "left_point_auroc": (
                        item[
                            "point_scores"
                        ][left_id]
                    ),
                    "right_point_auroc": (
                        item[
                            "point_scores"
                        ][right_id]
                    ),
                    "point_delta": point_delta,
                    "iterations": (
                        args.iterations
                    ),
                    **statistics,
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        lineterminator="\n",
    )

    np.savez_compressed(
        OUT_DISTRIBUTIONS,
        **saved_arrays,
    )

    metadata = {
        "status": "success",
        "seed": SEED,
        "iterations": args.iterations,
        "bootstrap_type": (
            "paired class-stratified bootstrap"
        ),
        "resampling": (
            "normal and anomaly samples are "
            "resampled separately within each "
            "backbone-category or category stratum"
        ),
        "parameter_reselection": False,
        "protocols": [
            {
                "protocol": item[
                    "specification"
                ]["protocol"],
                "path": str(
                    item[
                        "specification"
                    ]["path"].relative_to(ROOT)
                ),
                "rows": len(item["data"]),
                "strata": len(item["strata"]),
                "aggregation": item[
                    "specification"
                ]["aggregation"],
            }
            for item in loaded
        ],
    }

    OUT_METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Stage 22-C1: Paired Bootstrap Statistical Analysis",
        "",
        "## Protocol",
        "",
        f"- bootstrap iterations: `{args.iterations}`",
        f"- random seed: `{SEED}`",
        "- paired resampling: `yes`",
        "- class-stratified resampling: `yes`",
        "- parameter reselection during bootstrap: `none`",
        "",
        "## Results",
        "",
        "| Protocol | Comparison | Point delta | 95% CI | P(delta>0) | Two-sided p | Interpretation |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for _, row in summary.iterrows():
        if row["ci95_low"] > 0:
            interpretation = (
                "positive CI excludes zero"
            )
        elif row["ci95_high"] < 0:
            interpretation = (
                "negative CI excludes zero"
            )
        else:
            interpretation = (
                "CI includes zero"
            )

        lines.append(
            f"| {row['protocol']} | "
            f"{row['comparison']} | "
            f"{row['point_delta']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, "
            f"{row['ci95_high']:+.4f}] | "
            f"{row['probability_delta_gt_zero']:.4f} | "
            f"{row['bootstrap_two_sided_p']:.4f} | "
            f"{interpretation} |"
        )

    lines += [
        "",
        "## Claim rule",
        "",
        "- Use `significant improvement` only when the",
        "  complete 95% confidence interval is above zero.",
        "- When the interval includes zero, report the point",
        "  estimate and interval without a significance claim.",
        "- Offline potential call-rate results are not part",
        "  of this statistical test.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("===== STAGE 22-C1 SUCCESS =====")
    print()

    display = summary[
        [
            "protocol",
            "comparison",
            "point_delta",
            "ci95_low",
            "ci95_high",
            "probability_delta_gt_zero",
            "bootstrap_two_sided_p",
            "noninferior_at_minus_0_005",
        ]
    ]

    print(
        display.to_string(index=False)
    )

    print()
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_DISTRIBUTIONS)
    print("[DONE]", OUT_METADATA)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
