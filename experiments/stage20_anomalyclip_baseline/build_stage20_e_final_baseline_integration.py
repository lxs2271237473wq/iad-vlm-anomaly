from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


ROOT = Path(".").resolve()

ANOMALYCLIP_METRICS = (
    ROOT
    / "results/stage20_anomalyclip_baseline"
    / "stage20_d_anomalyclip_ad2four_metrics.csv"
)

ANOMALYCLIP_RAW = (
    ROOT
    / "results/stage20_anomalyclip_baseline"
    / "stage20_d_anomalyclip_ad2four_full_raw.json"
)

SYSTEM_TABLE = (
    ROOT
    / "results/stage16_qcru_ablation"
    / "stage16_d_paper_facing_system_baseline_table.csv"
)

OUT_DIR = ROOT / "results/stage20_anomalyclip_baseline"
DOC_DIR = ROOT / "docs/stage20_anomalyclip_baseline"

OUT_TABLE = (
    OUT_DIR
    / "stage20_e_final_system_baseline_with_anomalyclip.csv"
)

OUT_CLAIMS = (
    OUT_DIR
    / "stage20_e_anomalyclip_claim_audit.csv"
)

OUT_REPORT = (
    DOC_DIR
    / "stage20_e_anomalyclip_final_integration_report.md"
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(f"Empty CSV: {path}")

    return df


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"None of {candidates} found. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_anomalyclip_mean() -> dict:
    df = read_csv(ANOMALYCLIP_METRICS)

    category_col = find_column(
        df,
        ["category", "objects", "object"],
    )

    image_auroc_col = find_column(
        df,
        [
            "image_AUROC",
            "image_auroc",
            "image_AUROC_percent",
        ],
    )

    image_ap_col = find_column(
        df,
        [
            "image_AP",
            "image_ap",
            "image_AP_percent",
        ],
        required=False,
    )

    pixel_auroc_col = find_column(
        df,
        [
            "pixel_AUROC",
            "pixel_auroc",
            "pixel_AUROC_percent",
        ],
        required=False,
    )

    pixel_aupro_col = find_column(
        df,
        [
            "pixel_AUPRO",
            "pixel_aupro",
            "pixel_AUPRO_percent",
        ],
        required=False,
    )

    mean_rows = df[
        df[category_col].astype(str).str.lower() == "mean"
    ]

    if len(mean_rows) != 1:
        raise RuntimeError(
            f"Expected one mean row, found {len(mean_rows)}"
        )

    row = mean_rows.iloc[0]

    def normalized_metric(column: str | None) -> float | None:
        if column is None:
            return None

        value = float(row[column])

        # 防止输入是百分数形式，例如 82.15
        if value > 1.0:
            value /= 100.0

        if not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"Invalid metric {column}={value}"
            )

        return value

    return {
        "image_AUROC": normalized_metric(image_auroc_col),
        "image_AP": normalized_metric(image_ap_col),
        "pixel_AUROC": normalized_metric(pixel_auroc_col),
        "pixel_AUPRO": normalized_metric(pixel_aupro_col),
    }


def validate_run_status() -> dict:
    if not ANOMALYCLIP_RAW.exists():
        raise FileNotFoundError(ANOMALYCLIP_RAW)

    data = json.loads(
        ANOMALYCLIP_RAW.read_text(encoding="utf-8")
    )

    if data.get("status") != "success":
        raise RuntimeError(
            f"Stage 20-D status is not success: {data}"
        )

    if int(data.get("return_code", -1)) != 0:
        raise RuntimeError(
            f"Stage 20-D return_code is not zero: {data}"
        )

    return data


def normalize_system_table(df: pd.DataFrame) -> pd.DataFrame:
    method_col = find_column(
        df,
        ["method", "Method"],
    )

    auroc_col = find_column(
        df,
        [
            "mean_image_AUROC",
            "mean_AUROC",
            "image_AUROC",
            "mean_auroc",
        ],
    )

    role_col = find_column(
        df,
        ["paper_role", "role", "Role"],
        required=False,
    )

    protocol_col = find_column(
        df,
        ["protocol", "Protocol"],
        required=False,
    )

    rows = []

    for _, row in df.iterrows():
        value = float(row[auroc_col])

        if value > 1.0:
            value /= 100.0

        rows.append(
            {
                "method": str(row[method_col]),
                "image_AUROC": value,
                "paper_role": (
                    str(row[role_col])
                    if role_col is not None
                    else ""
                ),
                "protocol": (
                    str(row[protocol_col])
                    if protocol_col is not None
                    else ""
                ),
                "source_stage": "Stage16",
            }
        )

    return pd.DataFrame(rows)


def classify_result(
    anomalyclip: float,
    values: dict[str, float],
) -> tuple[str, str]:
    loco = values.get(
        "PatchCore + context VLM, LOCO"
    )

    patchcore = values.get("PatchCore")
    efficientad = values.get(
        "EfficientAD-30 fixed-budget"
    )
    winclip = values.get("WinCLIP fixed protocol")

    if loco is None:
        raise RuntimeError(
            "LOCO result missing from system table."
        )

    if anomalyclip > loco:
        decision = "anomalyclip_above_loco"
        interpretation = (
            "AnomalyCLIP is stronger than the current "
            "LOCO fusion on mean image AUROC. "
            "The paper must not claim best VLM-system "
            "performance on the AD2-four protocol."
        )

    elif abs(anomalyclip - loco) <= 0.005:
        decision = "approximately_tied_with_loco"
        interpretation = (
            "AnomalyCLIP and LOCO fusion are approximately "
            "tied within 0.005 AUROC. The contribution should "
            "be framed as a complementary detector-guided "
            "reasoning mechanism, not a superiority claim."
        )

    else:
        decision = "loco_above_anomalyclip"
        interpretation = (
            "The LOCO fusion is stronger than the fixed "
            "AnomalyCLIP baseline on this adapted AD2-four "
            "protocol. The claim must remain protocol-specific."
        )

    details = [
        interpretation,
        (
            f"AnomalyCLIP={anomalyclip:.4f}, "
            f"LOCO={loco:.4f}, "
            f"delta_LOCO_minus_AnomalyCLIP="
            f"{loco-anomalyclip:+.4f}."
        ),
    ]

    if patchcore is not None:
        details.append(
            f"delta_AnomalyCLIP_minus_PatchCore="
            f"{anomalyclip-patchcore:+.4f}."
        )

    if efficientad is not None:
        details.append(
            f"delta_AnomalyCLIP_minus_EfficientAD30="
            f"{anomalyclip-efficientad:+.4f}."
        )

    if winclip is not None:
        details.append(
            f"delta_AnomalyCLIP_minus_WinCLIP="
            f"{anomalyclip-winclip:+.4f}."
        )

    return decision, " ".join(details)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    run_info = validate_run_status()
    anomalyclip_metrics = load_anomalyclip_mean()

    system_raw = read_csv(SYSTEM_TABLE)
    system = normalize_system_table(system_raw)

    anomalyclip_row = pd.DataFrame(
        [
            {
                "method": "AnomalyCLIP fixed checkpoint",
                "image_AUROC": (
                    anomalyclip_metrics["image_AUROC"]
                ),
                "paper_role": (
                    "external VLM anomaly baseline"
                ),
                "protocol": (
                    "official implementation, fixed checkpoint, "
                    "AD2-four adapted metadata, no AD2 training"
                ),
                "source_stage": "Stage20",
            }
        ]
    )

    # 避免重复运行脚本时产生重复行
    system = system[
        system["method"]
        != "AnomalyCLIP fixed checkpoint"
    ]

    combined = pd.concat(
        [system, anomalyclip_row],
        ignore_index=True,
    )

    combined = combined.sort_values(
        "image_AUROC",
        ascending=False,
    ).reset_index(drop=True)

    combined.insert(
        0,
        "rank",
        range(1, len(combined) + 1),
    )

    values = {
        row["method"]: float(row["image_AUROC"])
        for _, row in combined.iterrows()
    }

    decision, interpretation = classify_result(
        anomalyclip_metrics["image_AUROC"],
        values,
    )

    claim_rows = [
        {
            "claim_id": "C20-1",
            "status": "allowed",
            "claim": (
                "AnomalyCLIP was evaluated as a fixed-checkpoint "
                "external VLM anomaly baseline on the adapted "
                "AD2-four protocol."
            ),
            "reason": (
                "The official implementation and checkpoint were "
                "used without AD2-specific training."
            ),
        },
        {
            "claim_id": "C20-2",
            "status": "allowed_with_protocol_qualifier",
            "claim": (
                "Numerical comparison between AnomalyCLIP and "
                "the proposed LOCO fusion."
            ),
            "reason": interpretation,
        },
        {
            "claim_id": "C20-3",
            "status": "forbidden",
            "claim": (
                "The proposed method universally outperforms "
                "AnomalyCLIP."
            ),
            "reason": (
                "Only one adapted four-category protocol was "
                "evaluated."
            ),
        },
        {
            "claim_id": "C20-4",
            "status": "forbidden",
            "claim": (
                "The Stage20 score is an exact reproduction of "
                "the official AnomalyCLIP benchmark."
            ),
            "reason": (
                "AD2-four is a custom adapted dataset view, not "
                "an official reported benchmark configuration."
            ),
        },
    ]

    claims = pd.DataFrame(claim_rows)

    combined.to_csv(
        OUT_TABLE,
        index=False,
        lineterminator="\n",
    )

    claims.to_csv(
        OUT_CLAIMS,
        index=False,
        lineterminator="\n",
    )

    anomalyclip = anomalyclip_metrics["image_AUROC"]
    loco = values["PatchCore + context VLM, LOCO"]

    lines = [
        "# Stage 20-E: Final AnomalyCLIP Integration",
        "",
        "## 1. Locked protocol",
        "",
        "- Stage 20-D status: `success`",
        f"- return code: `{run_info.get('return_code')}`",
        "- dataset: `AD2-four`",
        "- categories: `fruit_jelly`, `sheet_metal`, `vial`, `walnuts`",
        "- AD2-specific AnomalyCLIP training: `none`",
        "- checkpoint: fixed and recorded in Stage 20-D",
        "",
        "## 2. Locked AnomalyCLIP result",
        "",
        f"- mean image AUROC: `{anomalyclip:.4f}`",
        f"- mean image AP: `{anomalyclip_metrics['image_AP']}`",
        f"- mean pixel AUROC: `{anomalyclip_metrics['pixel_AUROC']}`",
        f"- mean pixel AUPRO: `{anomalyclip_metrics['pixel_AUPRO']}`",
        "",
        "## 3. Main comparison",
        "",
        f"- LOCO fusion image AUROC: `{loco:.4f}`",
        f"- AnomalyCLIP image AUROC: `{anomalyclip:.4f}`",
        f"- LOCO minus AnomalyCLIP: `{loco-anomalyclip:+.4f}`",
        f"- decision: `{decision}`",
        "",
        interpretation,
        "",
        "## 4. Final system table",
        "",
        "| Rank | Method | Image AUROC | Role |",
        "|---:|---|---:|---|",
    ]

    for _, row in combined.iterrows():
        lines.append(
            f"| {int(row['rank'])} | "
            f"{row['method']} | "
            f"{row['image_AUROC']:.4f} | "
            f"{row['paper_role']} |"
        )

    lines += [
        "",
        "## 5. Claim decision",
        "",
        "The paper should report the numerical comparison directly,",
        "but all superiority wording must remain limited to the",
        "implemented AD2-four fixed-checkpoint protocol.",
        "",
        "## 6. Next step",
        "",
        "Update the paper-facing baseline table, Results section,",
        "abstract, and limitations using this locked result.",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print("[DONE]", OUT_TABLE)
    print("[DONE]", OUT_CLAIMS)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== FINAL SYSTEM TABLE =====")
    print(
        combined[
            [
                "rank",
                "method",
                "image_AUROC",
                "paper_role",
            ]
        ].to_string(index=False)
    )
    print()
    print("===== DECISION =====")
    print("decision:", decision)
    print("AnomalyCLIP:", f"{anomalyclip:.4f}")
    print("LOCO:", f"{loco:.4f}")
    print(
        "LOCO - AnomalyCLIP:",
        f"{loco-anomalyclip:+.4f}",
    )
    print()
    print(interpretation)


if __name__ == "__main__":
    main()
