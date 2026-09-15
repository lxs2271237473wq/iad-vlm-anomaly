from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

PROTOCOL_PATH = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_a2_srb_qcr_frozen_protocol.json"
)

OUT_DIR = ROOT / "results/stage22_selective_qcr"
DOC_DIR = ROOT / "docs/stage22_selective_qcr"

OUT_GRID = (
    OUT_DIR
    / "stage22_b1_visa_patchcore_loco_grid.csv"
)

OUT_FOLDS = (
    OUT_DIR
    / "stage22_b1_visa_patchcore_loco_selected_folds.csv"
)

OUT_PRED = (
    OUT_DIR
    / "stage22_b1_visa_patchcore_loco_predictions.csv"
)

OUT_METRICS = (
    OUT_DIR
    / "stage22_b1_visa_patchcore_loco_metrics.csv"
)

OUT_GLOBAL = (
    OUT_DIR
    / "stage22_b1_visa_patchcore_global_config.json"
)

OUT_REPORT = (
    DOC_DIR
    / "stage22_b1_visa_patchcore_loco_report.md"
)


TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "t",
}

VARIANTS = {
    "D0": "Detector only",
    "V3": "Naive detector-crop fusion",
    "V4": "Old Quality-Calibrated QCR",
    "V6": "Old Adaptive QCR",
    "S1": "SRB-QCR",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(TRUE_VALUES)
    )


def safe_auroc(
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)

    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid]
    s = s[valid]

    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")

    return float(roc_auc_score(y, s))


def safe_ap(
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)

    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid]
    s = s[valid]

    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")

    return float(average_precision_score(y, s))


def best_threshold_metrics(
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)

    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid]
    s = s[valid]

    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")

    thresholds = np.unique(s)

    if len(thresholds) > 2000:
        thresholds = np.quantile(
            s,
            np.linspace(0.0, 1.0, 2000),
        )

    best_f1 = -1.0
    best_acc = -1.0
    best_threshold = float("nan")

    for threshold in thresholds:
        pred = (s >= threshold).astype(int)

        f1 = float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        )

        acc = float(accuracy_score(y, pred))

        if (
            f1 > best_f1 + 1e-12
            or (
                abs(f1 - best_f1) <= 1e-12
                and acc > best_acc
            )
        ):
            best_f1 = f1
            best_acc = acc
            best_threshold = float(threshold)

    return best_f1, best_acc, best_threshold


def evaluate_binary(
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
) -> dict:
    best_f1, best_acc, threshold = (
        best_threshold_metrics(y_true, score)
    )

    return {
        "auroc": safe_auroc(y_true, score),
        "ap": safe_ap(y_true, score),
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "best_threshold": threshold,
    }


def macro_auroc(
    df: pd.DataFrame,
    score_col: str,
) -> tuple[float, int]:
    values = []

    for _, group in df.groupby(
        ["backbone", "category"],
        dropna=False,
    ):
        value = safe_auroc(
            group["Y"],
            group[score_col],
        )

        if np.isfinite(value):
            values.append(value)

    if not values:
        return float("nan"), 0

    return float(np.mean(values)), len(values)


def quality_threshold(
    df: pd.DataFrame,
    quantile_level: float,
) -> float:
    eligible = (
        df["has_candidate_bool"]
        & ~df["fallback_bool"]
        & df["M_available"]
        & np.isfinite(df["Q"])
    )

    values = df.loc[eligible, "Q"].dropna()

    if values.empty:
        return 1.0

    return float(values.quantile(quantile_level))


def add_old_scores(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["score_D0"] = result["D"]

    result["score_V3"] = (
        0.5 * result["D"]
        + 0.5 * result["M"]
    )

    result["score_V4"] = (
        0.5 * result["D"]
        + 0.5
        * (
            result["M"]
            * (0.5 + 0.5 * result["Q"])
        )
    )

    agreement = (
        1.0 - (result["D"] - result["M"]).abs()
    ).clip(lower=0.0, upper=1.0)

    mutual_evidence = np.minimum(
        result["D"],
        result["M"],
    )

    adaptive_gate = (
        result["Q"]
        * result["K"]
        * agreement
        * mutual_evidence
    )

    result["score_V6"] = (
        result["score_V4"]
        + 0.05 * adaptive_gate
    )

    return result


def apply_srb(
    df: pd.DataFrame,
    w_max: float,
    q_threshold: float,
    tau_delta: float,
) -> pd.DataFrame:
    result = df.copy()

    pre_gate = (
        result["has_candidate_bool"]
        & ~result["fallback_bool"]
        & result["M_available"]
        & (result["Q"] >= q_threshold)
    )

    agreement = (
        1.0
        - (
            (result["D"] - result["M"]).abs()
            / tau_delta
        )
    ).clip(lower=0.0, upper=1.0)

    weight = (
        w_max
        * pre_gate.astype(float)
        * result["Q"]
        * agreement
    )

    weight = weight.clip(
        lower=0.0,
        upper=w_max,
    )

    result["srb_pre_gate"] = pre_gate.astype(int)
    result["srb_agreement"] = agreement
    result["srb_weight"] = weight

    result["score_S1"] = (
        result["D"]
        + weight * (result["M"] - result["D"])
    )

    result["srb_active"] = (
        weight > 0
    ).astype(int)

    return result


def prepare_base(
    protocol: dict,
) -> pd.DataFrame:
    source = ROOT / protocol["source"]["path"]

    if not source.exists():
        raise FileNotFoundError(source)

    expected_hash = protocol["source"]["sha256"]
    actual_hash = sha256(source)

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Stage 9 source CSV changed after protocol freeze.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    df = pd.read_csv(source)

    required = (
        protocol["source"]["required_columns"]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing source columns: {missing}"
        )

    keep = list(required)

    if "high_high_consistency" in df.columns:
        keep.append("high_high_consistency")

    if "image_path" in df.columns:
        keep.append("image_path")

    base = (
        df[keep]
        .drop_duplicates(
            subset=[
                "backbone",
                "dataset",
                "category",
                "strategy",
                "eval_mode",
                "image_key",
            ]
        )
        .reset_index(drop=True)
    )

    scope = protocol["frozen_primary_scope"]

    for column, value in scope.items():
        base = base[
            base[column].astype(str) == str(value)
        ]

    base = base[
        base["backbone"]
        .astype(str)
        .str.contains(
            "patchcore",
            case=False,
            regex=False,
        )
    ].copy()

    if base.empty:
        raise RuntimeError(
            "No VisA PatchCore rows found."
        )

    base["Y"] = pd.to_numeric(
        base["is_anomaly_final"],
        errors="coerce",
    )

    base["D"] = pd.to_numeric(
        base["detector_score_norm"],
        errors="coerce",
    )

    m_numeric = pd.to_numeric(
        base["vlm_score_norm"],
        errors="coerce",
    )

    base["M_available"] = np.isfinite(m_numeric)
    base["M"] = m_numeric.fillna(0.0)

    base["Q"] = pd.to_numeric(
        base["candidate_quality_norm"],
        errors="coerce",
    ).fillna(0.0)

    if "high_high_consistency" in base.columns:
        base["K"] = pd.to_numeric(
            base["high_high_consistency"],
            errors="coerce",
        ).fillna(0.0)
    else:
        base["K"] = 0.0

    base["fallback_bool"] = as_bool(
        base["fallback"]
    )

    base["has_candidate_bool"] = as_bool(
        base["has_candidate"]
    )

    base = base[
        np.isfinite(base["Y"])
        & np.isfinite(base["D"])
    ].copy()

    base["Y"] = base["Y"].astype(int)

    for column in ["D", "M", "Q", "K"]:
        base[column] = base[column].clip(
            lower=0.0,
            upper=1.0,
        )

    base = add_old_scores(base)

    return base.reset_index(drop=True)


def select_fold_configuration(
    fold_grid: pd.DataFrame,
    auroc_tolerance: float,
) -> tuple[pd.Series | None, bool]:
    eligible = fold_grid[
        fold_grid["eligible_noninferior"]
    ].copy()

    if eligible.empty:
        return None, True

    best_auroc = float(
        eligible["dev_macro_auroc"].max()
    )

    tied = eligible[
        eligible["dev_macro_auroc"]
        >= best_auroc - auroc_tolerance
    ].copy()

    tied = tied.sort_values(
        [
            "dev_potential_call_rate",
            "w_max",
            "q_quantile",
            "tau_delta",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    )

    return tied.iloc[0], False


def evaluate_fold_variant(
    fold_df: pd.DataFrame,
    score_col: str,
) -> dict:
    per_backbone = []

    for backbone, group in fold_df.groupby(
        "backbone",
        dropna=False,
    ):
        metrics = evaluate_binary(
            group["Y"],
            group[score_col],
        )

        per_backbone.append(
            {
                "backbone": str(backbone),
                **metrics,
            }
        )

    frame = pd.DataFrame(per_backbone)

    if frame.empty:
        return {
            "auroc": float("nan"),
            "ap": float("nan"),
            "best_f1": float("nan"),
            "best_accuracy": float("nan"),
            "num_backbones": 0,
        }

    return {
        "auroc": float(frame["auroc"].mean()),
        "ap": float(frame["ap"].mean()),
        "best_f1": float(
            frame["best_f1"].mean()
        ),
        "best_accuracy": float(
            frame["best_accuracy"].mean()
        ),
        "num_backbones": len(frame),
    }


def global_config_from_folds(
    selected_folds: pd.DataFrame,
) -> dict:
    valid = selected_folds[
        ~selected_folds["detector_only_fallback"]
    ].copy()

    if valid.empty:
        return {
            "method": "detector_only",
            "reason": (
                "No fold selected an eligible SRB-QCR "
                "configuration."
            ),
        }

    key_cols = [
        "selected_w_max",
        "selected_q_quantile",
        "selected_tau_delta",
    ]

    keys = [
        tuple(row)
        for row in valid[key_cols].to_numpy()
    ]

    counts = Counter(keys)

    candidates = []

    for key, count in counts.items():
        mask = np.ones(len(valid), dtype=bool)

        for column, value in zip(
            key_cols,
            key,
        ):
            mask &= np.isclose(
                valid[column].astype(float),
                float(value),
            )

        subset = valid.loc[mask]

        candidates.append(
            {
                "w_max": float(key[0]),
                "q_quantile": float(key[1]),
                "tau_delta": float(key[2]),
                "selection_count": int(count),
                "mean_dev_rank": float(
                    subset[
                        "selected_dev_rank"
                    ].mean()
                ),
                "mean_dev_call_rate": float(
                    subset[
                        "selected_dev_potential_call_rate"
                    ].mean()
                ),
            }
        )

    candidates.sort(
        key=lambda row: (
            -row["selection_count"],
            row["mean_dev_rank"],
            row["mean_dev_call_rate"],
            row["w_max"],
            -row["q_quantile"],
            row["tau_delta"],
        )
    )

    return {
        "method": "SRB-QCR",
        "selection_rule": (
            "frequency, mean development rank, "
            "call rate, conservative tie order"
        ),
        "selected": candidates[0],
        "all_candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8",
        )
    )

    if (
        protocol.get("status")
        != "frozen_before_stage22_b_results"
    ):
        raise RuntimeError(
            "Unexpected frozen protocol status."
        )

    base = prepare_base(protocol)

    categories = sorted(
        base["category"]
        .astype(str)
        .unique()
        .tolist()
    )

    print("===== STAGE 22-B1 DATA VALIDATION =====")
    print("rows:", len(base))
    print("categories:", len(categories))
    print("backbones:", sorted(
        base["backbone"]
        .astype(str)
        .unique()
        .tolist()
    ))
    print("labels:", sorted(
        base["Y"].unique().tolist()
    ))
    print("protocol:", PROTOCOL_PATH)
    print()

    if args.validate_only:
        print("[OK] Validation-only run passed.")
        return

    grid = protocol["hyperparameter_grid"]

    configs = list(
        itertools.product(
            grid["w_max"],
            grid["tau_q"],
            grid["tau_delta"],
        )
    )

    if len(configs) != 27:
        raise RuntimeError(
            f"Expected 27 configurations, "
            f"found {len(configs)}."
        )

    noninferiority = 0.002
    auroc_tolerance = 0.001

    grid_rows = []
    selected_rows = []
    prediction_frames = []
    metric_rows = []

    for heldout_category in categories:
        dev = base[
            base["category"].astype(str)
            != heldout_category
        ].copy()

        test = base[
            base["category"].astype(str)
            == heldout_category
        ].copy()

        detector_macro, detector_groups = (
            macro_auroc(dev, "score_D0")
        )

        fold_grid_rows = []

        q_thresholds = {
            float(q_level): quality_threshold(
                dev,
                float(q_level),
            )
            for q_level in grid["tau_q"]
        }

        for w_max, q_level, tau_delta in configs:
            q_level = float(q_level)

            candidate = apply_srb(
                dev,
                w_max=float(w_max),
                q_threshold=q_thresholds[q_level],
                tau_delta=float(tau_delta),
            )

            dev_macro, valid_groups = macro_auroc(
                candidate,
                "score_S1",
            )

            call_rate = float(
                candidate[
                    "srb_pre_gate"
                ].mean()
            )

            active_rate = float(
                candidate["srb_active"].mean()
            )

            row = {
                "heldout_category": (
                    heldout_category
                ),
                "w_max": float(w_max),
                "q_quantile": q_level,
                "q_threshold_from_dev": (
                    q_thresholds[q_level]
                ),
                "tau_delta": float(tau_delta),
                "dev_macro_auroc": dev_macro,
                "dev_detector_macro_auroc": (
                    detector_macro
                ),
                "dev_delta_vs_detector": (
                    dev_macro - detector_macro
                ),
                "eligible_noninferior": bool(
                    np.isfinite(dev_macro)
                    and np.isfinite(
                        detector_macro
                    )
                    and dev_macro
                    >= detector_macro
                    - noninferiority
                ),
                "dev_potential_call_rate": (
                    call_rate
                ),
                "dev_active_weight_rate": (
                    active_rate
                ),
                "valid_dev_groups": valid_groups,
                "detector_dev_groups": (
                    detector_groups
                ),
            }

            fold_grid_rows.append(row)

        fold_grid = pd.DataFrame(
            fold_grid_rows
        )

        fold_grid["dev_rank"] = (
            fold_grid["dev_macro_auroc"]
            .rank(
                method="min",
                ascending=False,
            )
        )

        selected, detector_fallback = (
            select_fold_configuration(
                fold_grid,
                auroc_tolerance=auroc_tolerance,
            )
        )

        if detector_fallback:
            test_scored = test.copy()
            test_scored["score_S1"] = (
                test_scored["score_D0"]
            )
            test_scored["srb_pre_gate"] = 0
            test_scored["srb_agreement"] = 0.0
            test_scored["srb_weight"] = 0.0
            test_scored["srb_active"] = 0

            selected_row = {
                "heldout_category": (
                    heldout_category
                ),
                "detector_only_fallback": True,
                "selected_w_max": np.nan,
                "selected_q_quantile": np.nan,
                "selected_q_threshold": np.nan,
                "selected_tau_delta": np.nan,
                "selected_dev_macro_auroc": (
                    detector_macro
                ),
                "selected_dev_detector_auroc": (
                    detector_macro
                ),
                "selected_dev_delta_vs_detector": (
                    0.0
                ),
                "selected_dev_potential_call_rate": (
                    0.0
                ),
                "selected_dev_rank": np.nan,
            }

        else:
            q_threshold = float(
                selected[
                    "q_threshold_from_dev"
                ]
            )

            test_scored = apply_srb(
                test,
                w_max=float(
                    selected["w_max"]
                ),
                q_threshold=q_threshold,
                tau_delta=float(
                    selected["tau_delta"]
                ),
            )

            selected_row = {
                "heldout_category": (
                    heldout_category
                ),
                "detector_only_fallback": False,
                "selected_w_max": float(
                    selected["w_max"]
                ),
                "selected_q_quantile": float(
                    selected["q_quantile"]
                ),
                "selected_q_threshold": (
                    q_threshold
                ),
                "selected_tau_delta": float(
                    selected["tau_delta"]
                ),
                "selected_dev_macro_auroc": (
                    float(
                        selected[
                            "dev_macro_auroc"
                        ]
                    )
                ),
                "selected_dev_detector_auroc": (
                    detector_macro
                ),
                "selected_dev_delta_vs_detector": (
                    float(
                        selected[
                            "dev_delta_vs_detector"
                        ]
                    )
                ),
                "selected_dev_potential_call_rate": (
                    float(
                        selected[
                            "dev_potential_call_rate"
                        ]
                    )
                ),
                "selected_dev_rank": float(
                    selected["dev_rank"]
                ),
            }

        selected_rows.append(selected_row)

        test_scored["heldout_category"] = (
            heldout_category
        )

        test_scored["selected_w_max"] = (
            selected_row["selected_w_max"]
        )
        test_scored[
            "selected_q_quantile"
        ] = selected_row[
            "selected_q_quantile"
        ]
        test_scored[
            "selected_q_threshold"
        ] = selected_row[
            "selected_q_threshold"
        ]
        test_scored[
            "selected_tau_delta"
        ] = selected_row[
            "selected_tau_delta"
        ]

        prediction_frames.append(test_scored)

        for variant_id, method in VARIANTS.items():
            score_col = f"score_{variant_id}"

            metrics = evaluate_fold_variant(
                test_scored,
                score_col,
            )

            metric_rows.append(
                {
                    "scope": "heldout_fold",
                    "heldout_category": (
                        heldout_category
                    ),
                    "variant_id": variant_id,
                    "method": method,
                    **metrics,
                    "potential_call_rate": (
                        float(
                            test_scored[
                                "srb_pre_gate"
                            ].mean()
                        )
                        if variant_id == "S1"
                        else np.nan
                    ),
                    "active_weight_rate": (
                        float(
                            test_scored[
                                "srb_active"
                            ].mean()
                        )
                        if variant_id == "S1"
                        else np.nan
                    ),
                }
            )

        grid_rows.extend(
            fold_grid.to_dict(
                orient="records"
            )
        )

    grid_df = pd.DataFrame(grid_rows)
    selected_df = pd.DataFrame(selected_rows)
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    for variant_id, method in VARIANTS.items():
        score_col = f"score_{variant_id}"

        per_backbone = []

        for backbone, group in predictions.groupby(
            "backbone",
            dropna=False,
        ):
            metrics = evaluate_binary(
                group["Y"],
                group[score_col],
            )

            per_backbone.append(
                {
                    "backbone": str(backbone),
                    **metrics,
                }
            )

            metric_rows.append(
                {
                    "scope": "loco_per_backbone",
                    "heldout_category": "all",
                    "variant_id": variant_id,
                    "method": method,
                    **metrics,
                    "potential_call_rate": (
                        float(
                            group[
                                "srb_pre_gate"
                            ].mean()
                        )
                        if variant_id == "S1"
                        else np.nan
                    ),
                    "active_weight_rate": (
                        float(
                            group[
                                "srb_active"
                            ].mean()
                        )
                        if variant_id == "S1"
                        else np.nan
                    ),
                    "backbone": str(backbone),
                }
            )

        backbone_df = pd.DataFrame(
            per_backbone
        )

        metric_rows.append(
            {
                "scope": "loco_mean_over_backbones",
                "heldout_category": "all",
                "variant_id": variant_id,
                "method": method,
                "auroc": float(
                    backbone_df["auroc"].mean()
                ),
                "ap": float(
                    backbone_df["ap"].mean()
                ),
                "best_f1": float(
                    backbone_df["best_f1"].mean()
                ),
                "best_accuracy": float(
                    backbone_df[
                        "best_accuracy"
                    ].mean()
                ),
                "num_backbones": len(
                    backbone_df
                ),
                "potential_call_rate": (
                    float(
                        predictions[
                            "srb_pre_gate"
                        ].mean()
                    )
                    if variant_id == "S1"
                    else np.nan
                ),
                "active_weight_rate": (
                    float(
                        predictions[
                            "srb_active"
                        ].mean()
                    )
                    if variant_id == "S1"
                    else np.nan
                ),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)

    global_config = global_config_from_folds(
        selected_df
    )

    global_payload = {
        "protocol_id": protocol["protocol_id"],
        "source_sha256": (
            protocol["source"]["sha256"]
        ),
        "selection_source": (
            "VisA PatchCore category-LOCO"
        ),
        "num_categories": len(categories),
        "global_configuration": (
            global_config
        ),
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_df.to_csv(
        OUT_GRID,
        index=False,
        lineterminator="\n",
    )

    selected_df.to_csv(
        OUT_FOLDS,
        index=False,
        lineterminator="\n",
    )

    predictions.to_csv(
        OUT_PRED,
        index=False,
        lineterminator="\n",
    )

    metrics_df.to_csv(
        OUT_METRICS,
        index=False,
        lineterminator="\n",
    )

    OUT_GLOBAL.write_text(
        json.dumps(
            global_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    main_metrics = metrics_df[
        metrics_df["scope"]
        == "loco_mean_over_backbones"
    ].copy()

    srb = main_metrics[
        main_metrics["variant_id"] == "S1"
    ].iloc[0]

    detector = main_metrics[
        main_metrics["variant_id"] == "D0"
    ].iloc[0]

    naive = main_metrics[
        main_metrics["variant_id"] == "V3"
    ].iloc[0]

    old_quality = main_metrics[
        main_metrics["variant_id"] == "V4"
    ].iloc[0]

    old_adaptive = main_metrics[
        main_metrics["variant_id"] == "V6"
    ].iloc[0]

    lines = [
        "# Stage 22-B1: VisA PatchCore Category-LOCO SRB-QCR",
        "",
        "## Protocol",
        "",
        "- method: `Selective Reliability-Bounded QCR`",
        "- development source: `VisA PatchCore`",
        "- selection: leave one category out",
        "- configurations per fold: `27`",
        "- target-label use during selection: `none`",
        "- detector/VLM inference rerun: `none`",
        "",
        "## LOCO comparison",
        "",
        "| Variant | Mean image AUROC | AP | Best F1 | Potential VLM call rate |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in main_metrics.iterrows():
        call_rate = (
            f"{row['potential_call_rate']:.4f}"
            if pd.notna(
                row["potential_call_rate"]
            )
            else "-"
        )

        lines.append(
            f"| {row['method']} | "
            f"{row['auroc']:.4f} | "
            f"{row['ap']:.4f} | "
            f"{row['best_f1']:.4f} | "
            f"{call_rate} |"
        )

    lines += [
        "",
        "## Main deltas",
        "",
        f"- SRB-QCR minus detector: `{srb['auroc'] - detector['auroc']:+.4f}`",
        f"- SRB-QCR minus naive fusion: `{srb['auroc'] - naive['auroc']:+.4f}`",
        f"- SRB-QCR minus old Quality QCR: `{srb['auroc'] - old_quality['auroc']:+.4f}`",
        f"- SRB-QCR minus old Adaptive QCR: `{srb['auroc'] - old_adaptive['auroc']:+.4f}`",
        f"- potential VLM call rate: `{srb['potential_call_rate']:.4f}`",
        f"- potential calls saved: `{1.0 - srb['potential_call_rate']:.4f}`",
        "",
        "## Fold selections",
        "",
        "| Held-out category | Fallback | w_max | q quantile | q threshold | tau_delta | Dev AUROC | Dev delta vs detector |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in selected_df.iterrows():
        lines.append(
            f"| {row['heldout_category']} | "
            f"{bool(row['detector_only_fallback'])} | "
            f"{row['selected_w_max']} | "
            f"{row['selected_q_quantile']} | "
            f"{row['selected_q_threshold']} | "
            f"{row['selected_tau_delta']} | "
            f"{row['selected_dev_macro_auroc']:.4f} | "
            f"{row['selected_dev_delta_vs_detector']:+.4f} |"
        )

    lines += [
        "",
        "## Frozen transfer configuration",
        "",
        "```json",
        json.dumps(
            global_config,
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "## Interpretation restriction",
        "",
        "Potential call rate is an offline estimate based on",
        "the pre-VLM quality gate. It is not yet a measured",
        "wall-clock speedup.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print("===== STAGE 22-B1 SUCCESS =====")
    print()
    print(
        main_metrics[
            [
                "variant_id",
                "method",
                "auroc",
                "ap",
                "best_f1",
                "potential_call_rate",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "SRB - detector:",
        f"{srb['auroc'] - detector['auroc']:+.4f}",
    )
    print(
        "SRB - naive:",
        f"{srb['auroc'] - naive['auroc']:+.4f}",
    )
    print(
        "SRB - old quality:",
        f"{srb['auroc'] - old_quality['auroc']:+.4f}",
    )
    print(
        "SRB - old adaptive:",
        f"{srb['auroc'] - old_adaptive['auroc']:+.4f}",
    )
    print(
        "potential call rate:",
        f"{srb['potential_call_rate']:.4f}",
    )
    print(
        "potential calls saved:",
        f"{1.0 - srb['potential_call_rate']:.4f}",
    )
    print()
    print("[DONE]", OUT_GRID)
    print("[DONE]", OUT_FOLDS)
    print("[DONE]", OUT_PRED)
    print("[DONE]", OUT_METRICS)
    print("[DONE]", OUT_GLOBAL)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
