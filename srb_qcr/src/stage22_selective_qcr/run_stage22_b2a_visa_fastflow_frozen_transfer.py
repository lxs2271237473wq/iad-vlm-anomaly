from __future__ import annotations

import argparse
import hashlib
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
    / "stage22_b2a_visa_fastflow_frozen_predictions.csv"
)

OUT_PER_CATEGORY = (
    OUT_DIR
    / "stage22_b2a_visa_fastflow_per_category.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "stage22_b2a_visa_fastflow_summary.csv"
)

OUT_TRANSFER = (
    OUT_DIR
    / "stage22_b2a_visa_fastflow_transfer_metadata.json"
)

OUT_REPORT = (
    DOC_DIR
    / "stage22_b2a_visa_fastflow_frozen_transfer_report.md"
)

TRUE_VALUES = {"1", "true", "yes", "y", "t"}

VARIANTS = {
    "D0": ("Detector only", "score_D0"),
    "V3": ("Naive detector-crop fusion", "score_V3"),
    "V4": ("Old Quality-Calibrated QCR", "score_V4"),
    "V6": ("Old Adaptive QCR", "score_V6"),
    "S1": ("SRB-QCR frozen transfer", "score_S1"),
}


def load_b1_module():
    if not B1_SCRIPT.exists():
        raise FileNotFoundError(B1_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "stage22_b1_module",
        B1_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {B1_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def prepare_fastflow(
    protocol: dict,
    b1,
) -> pd.DataFrame:
    source = ROOT / protocol["source"]["path"]

    if not source.exists():
        raise FileNotFoundError(source)

    actual_hash = sha256(source)
    expected_hash = protocol["source"]["sha256"]

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen source changed after Stage 22-A2.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    df = pd.read_csv(source)

    required = protocol["source"]["required_columns"]

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

    for optional in [
        "high_high_consistency",
        "image_path",
        "gt_label",
        "defect_type",
    ]:
        if optional in df.columns:
            keep.append(optional)

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

    for column, value in (
        protocol["frozen_primary_scope"].items()
    ):
        base = base[
            base[column].astype(str) == str(value)
        ]

    base = base[
        base["backbone"]
        .astype(str)
        .str.contains(
            "fastflow",
            case=False,
            regex=False,
        )
    ].copy()

    if base.empty:
        raise RuntimeError(
            "No VisA FastFlow rows were found in "
            "the frozen Stage 9 prediction table."
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

    base = b1.add_old_scores(base)

    return base.reset_index(drop=True)


def target_quality_threshold(
    df: pd.DataFrame,
    quantile: float,
) -> tuple[float, int]:
    eligible = (
        df["has_candidate_bool"]
        & ~df["fallback_bool"]
        & df["M_available"]
        & np.isfinite(df["Q"])
    )

    values = df.loc[eligible, "Q"].dropna()

    if values.empty:
        return 1.0, 0

    return float(values.quantile(quantile)), len(values)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    for path in [
        PROTOCOL_PATH,
        CONFIG_PATH,
        B1_SCRIPT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    config_payload = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    selected = (
        config_payload[
            "global_configuration"
        ]["selected"]
    )

    w_max = float(selected["w_max"])
    q_quantile = float(
        selected["q_quantile"]
    )
    tau_delta = float(
        selected["tau_delta"]
    )

    # Exact frozen configuration expected from Stage 22-B1.
    expected = {
        "w_max": 0.35,
        "q_quantile": 0.25,
        "tau_delta": 0.75,
    }

    actual = {
        "w_max": w_max,
        "q_quantile": q_quantile,
        "tau_delta": tau_delta,
    }

    if actual != expected:
        raise RuntimeError(
            "Unexpected frozen global configuration.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    b1 = load_b1_module()
    base = prepare_fastflow(protocol, b1)

    q_threshold, q_count = (
        target_quality_threshold(
            base,
            q_quantile,
        )
    )

    print(
        "===== STAGE 22-B2a FASTFLOW VALIDATION ====="
    )
    print("rows:", len(base))
    print(
        "categories:",
        base["category"].nunique(),
    )
    print(
        "backbones:",
        sorted(
            base["backbone"]
            .astype(str)
            .unique()
            .tolist()
        ),
    )
    print(
        "labels:",
        sorted(base["Y"].unique().tolist()),
    )
    print("w_max:", w_max)
    print("q_quantile:", q_quantile)
    print("target q threshold:", q_threshold)
    print("eligible q samples:", q_count)
    print("tau_delta:", tau_delta)
    print()

    if base["category"].nunique() != 12:
        raise RuntimeError(
            "Expected 12 VisA categories."
        )

    if sorted(
        base["Y"].unique().tolist()
    ) != [0, 1]:
        raise RuntimeError(
            "Expected binary labels [0, 1]."
        )

    if args.validate_only:
        print("[OK] Validation-only run passed.")
        return

    scored = b1.apply_srb(
        base,
        w_max=w_max,
        q_threshold=q_threshold,
        tau_delta=tau_delta,
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
                group["srb_pre_gate"].mean()
            ),
            "active_weight_rate": float(
                group["srb_active"].mean()
            ),
        }

        for variant_id, (_, score_col) in (
            VARIANTS.items()
        ):
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
        backbone_rows = []

        for backbone, group in scored.groupby(
            "backbone",
            dropna=False,
        ):
            metrics = b1.evaluate_binary(
                group["Y"],
                group[score_col],
            )

            backbone_rows.append(
                {
                    "backbone": str(backbone),
                    **metrics,
                }
            )

        backbone_frame = pd.DataFrame(
            backbone_rows
        )

        summary_rows.append(
            {
                "variant_id": variant_id,
                "method": method,
                "mean_image_auroc": float(
                    backbone_frame["auroc"].mean()
                ),
                "mean_image_ap": float(
                    backbone_frame["ap"].mean()
                ),
                "mean_best_f1": float(
                    backbone_frame[
                        "best_f1"
                    ].mean()
                ),
                "num_backbones": len(
                    backbone_frame
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

    summary = pd.DataFrame(summary_rows)

    scores = {
        row["variant_id"]: float(
            row["mean_image_auroc"]
        )
        for _, row in summary.iterrows()
    }

    transfer_metadata = {
        "status": "success",
        "source": protocol["source"]["path"],
        "source_sha256": (
            protocol["source"]["sha256"]
        ),
        "selection_source": (
            "VisA PatchCore category-LOCO"
        ),
        "transfer_target": "VisA FastFlow",
        "target_labels_used_for_parameters": False,
        "target_quality_rule": (
            "25th percentile of unlabeled eligible "
            "target candidate-quality values"
        ),
        "configuration": actual,
        "target_q_threshold": q_threshold,
        "eligible_q_samples": q_count,
        "rows": len(scored),
        "categories": int(
            scored["category"].nunique()
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

    OUT_TRANSFER.write_text(
        json.dumps(
            transfer_metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    srb = scores["S1"]

    lines = [
        "# Stage 22-B2a: Frozen SRB-QCR Transfer to VisA FastFlow",
        "",
        "## Protocol",
        "",
        "- parameters selected on: `VisA PatchCore category-LOCO`",
        "- target: `VisA FastFlow`",
        "- target labels used for parameter selection: `none`",
        f"- `w_max = {w_max}`",
        f"- `q_quantile = {q_quantile}`",
        f"- target unlabeled Q threshold: `{q_threshold:.6f}`",
        f"- `tau_delta = {tau_delta}`",
        "",
        "## Summary",
        "",
        "| Variant | Image AUROC | AP | Best F1 | Potential call rate |",
        "|---|---:|---:|---:|---:|",
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
            f"{row['mean_image_ap']:.4f} | "
            f"{row['mean_best_f1']:.4f} | "
            f"{call_rate} |"
        )

    lines += [
        "",
        "## Frozen-transfer deltas",
        "",
        f"- SRB minus detector: `{srb - scores['D0']:+.4f}`",
        f"- SRB minus naive: `{srb - scores['V3']:+.4f}`",
        f"- SRB minus old Quality QCR: `{srb - scores['V4']:+.4f}`",
        f"- SRB minus old Adaptive QCR: `{srb - scores['V6']:+.4f}`",
        f"- categories SRB > detector: `{int((per_category['S1_minus_D0'] > 0).sum())}/12`",
        f"- worst category delta vs detector: `{per_category['S1_minus_D0'].min():+.4f}`",
        f"- potential calls saved: `{1.0 - float(scored['srb_pre_gate'].mean()):.4f}`",
        "",
        "Potential call saving remains an offline estimate,",
        "not a measured runtime speedup.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print(
        "===== STAGE 22-B2a SUCCESS ====="
    )
    print()
    print(
        summary[
            [
                "variant_id",
                "method",
                "mean_image_auroc",
                "mean_image_ap",
                "mean_best_f1",
                "potential_call_rate",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "SRB - detector:",
        f"{srb - scores['D0']:+.6f}",
    )
    print(
        "SRB - naive:",
        f"{srb - scores['V3']:+.6f}",
    )
    print(
        "SRB - old quality:",
        f"{srb - scores['V4']:+.6f}",
    )
    print(
        "SRB - old adaptive:",
        f"{srb - scores['V6']:+.6f}",
    )
    print(
        "category wins vs detector:",
        f"{int((per_category['S1_minus_D0'] > 0).sum())}/12",
    )
    print(
        "worst category delta vs detector:",
        f"{per_category['S1_minus_D0'].min():+.6f}",
    )
    print(
        "potential call rate:",
        f"{scored['srb_pre_gate'].mean():.6f}",
    )
    print(
        "potential calls saved:",
        f"{1.0 - scored['srb_pre_gate'].mean():.6f}",
    )
    print()
    print("[DONE]", OUT_PRED)
    print("[DONE]", OUT_PER_CATEGORY)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_TRANSFER)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
