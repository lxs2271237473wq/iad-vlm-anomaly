from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_PER_CONFIG = ROOT / "results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_config.csv"
IN_PER_CATEGORY = ROOT / "results/stage16_qcru_ablation/stage16_a1_qcru_fixed_ablation_per_category.csv"

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_DELTA = OUT_DIR / "stage16_a2_qcru_variant_delta_by_protocol.csv"
OUT_SUMMARY = OUT_DIR / "stage16_a2_qcru_robustness_summary.csv"
OUT_FAILURES = OUT_DIR / "stage16_a2_qcru_failure_cases.csv"
OUT_DOC = DOC_DIR / "stage16_a2_qcru_robustness_check_report.md"


VARIANT_NAMES = {
    "V0": "detector_only",
    "V2": "crop_topk_vlm",
    "V3": "naive_detector_crop_fusion",
    "V4": "quality_weighted_crop",
    "V5": "quality_consistency_fusion",
}


def read_csv_robust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Fix CSV line breaks before running Stage 16-A2.")
    return df


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    idx = ["backbone", "dataset", "strategy", "eval_mode"]
    piv = df.pivot_table(index=idx, columns="variant_id", values=metric, aggfunc="first").reset_index()
    piv.columns.name = None
    return piv


def compute_protocol_deltas(per_config: pd.DataFrame) -> pd.DataFrame:
    piv = pivot_metric(per_config, "auroc")

    required = ["V0", "V2", "V3", "V4", "V5"]
    missing = [c for c in required if c not in piv.columns]
    if missing:
        raise RuntimeError(f"Missing variant columns in pivot: {missing}")

    out = piv.copy()
    out["delta_v5_minus_v3_naive"] = out["V5"] - out["V3"]
    out["delta_v5_minus_v4_quality"] = out["V5"] - out["V4"]
    out["delta_v5_minus_v0_detector"] = out["V5"] - out["V0"]
    out["delta_v5_minus_v2_crop"] = out["V5"] - out["V2"]
    out["delta_v4_minus_v3_naive"] = out["V4"] - out["V3"]

    out["v5_beats_naive"] = out["delta_v5_minus_v3_naive"] > 0
    out["v5_beats_quality_only"] = out["delta_v5_minus_v4_quality"] > 0
    out["v5_beats_detector"] = out["delta_v5_minus_v0_detector"] > 0
    out["v5_beats_crop"] = out["delta_v5_minus_v2_crop"] > 0
    out["quality_beats_naive"] = out["delta_v4_minus_v3_naive"] > 0

    return out


def summarize_boolean(df: pd.DataFrame, col: str) -> tuple[int, int, float]:
    total = len(df)
    wins = int(df[col].sum())
    rate = wins / total if total else 0.0
    return wins, total, rate


def make_summary(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []

    checks = [
        ("V5 > V3 naive fusion", "v5_beats_naive", "delta_v5_minus_v3_naive"),
        ("V5 > V4 quality-only", "v5_beats_quality_only", "delta_v5_minus_v4_quality"),
        ("V5 > V0 detector-only", "v5_beats_detector", "delta_v5_minus_v0_detector"),
        ("V5 > V2 crop-VLM-only", "v5_beats_crop", "delta_v5_minus_v2_crop"),
        ("V4 > V3 naive fusion", "quality_beats_naive", "delta_v4_minus_v3_naive"),
    ]

    for name, bool_col, delta_col in checks:
        wins, total, rate = summarize_boolean(delta, bool_col)
        rows.append(
            {
                "check": name,
                "wins": wins,
                "total_protocols": total,
                "win_rate": rate,
                "mean_delta": delta[delta_col].mean(),
                "median_delta": delta[delta_col].median(),
                "min_delta": delta[delta_col].min(),
                "max_delta": delta[delta_col].max(),
            }
        )

    # By eval mode, because full_all may behave differently from crop protocols.
    for eval_mode, g in delta.groupby("eval_mode"):
        wins, total, rate = summarize_boolean(g, "v5_beats_naive")
        rows.append(
            {
                "check": f"V5 > V3 naive fusion by eval_mode={eval_mode}",
                "wins": wins,
                "total_protocols": total,
                "win_rate": rate,
                "mean_delta": g["delta_v5_minus_v3_naive"].mean(),
                "median_delta": g["delta_v5_minus_v3_naive"].median(),
                "min_delta": g["delta_v5_minus_v3_naive"].min(),
                "max_delta": g["delta_v5_minus_v3_naive"].max(),
            }
        )

        wins, total, rate = summarize_boolean(g, "v5_beats_quality_only")
        rows.append(
            {
                "check": f"V5 > V4 quality-only by eval_mode={eval_mode}",
                "wins": wins,
                "total_protocols": total,
                "win_rate": rate,
                "mean_delta": g["delta_v5_minus_v4_quality"].mean(),
                "median_delta": g["delta_v5_minus_v4_quality"].median(),
                "min_delta": g["delta_v5_minus_v4_quality"].min(),
                "max_delta": g["delta_v5_minus_v4_quality"].max(),
            }
        )

    return pd.DataFrame(rows)


def make_failures(delta: pd.DataFrame) -> pd.DataFrame:
    failures = delta[
        (~delta["v5_beats_naive"])
        | (~delta["v5_beats_quality_only"])
        | (~delta["v5_beats_detector"])
        | (~delta["v5_beats_crop"])
    ].copy()

    if failures.empty:
        return failures

    failures["failure_reason"] = ""
    reasons = []

    for _, r in failures.iterrows():
        rs = []
        if not bool(r["v5_beats_naive"]):
            rs.append("V5_not_better_than_naive")
        if not bool(r["v5_beats_quality_only"]):
            rs.append("V5_not_better_than_quality_only")
        if not bool(r["v5_beats_detector"]):
            rs.append("V5_not_better_than_detector")
        if not bool(r["v5_beats_crop"]):
            rs.append("V5_not_better_than_crop")
        reasons.append(";".join(rs))

    failures["failure_reason"] = reasons
    return failures.sort_values(["eval_mode", "delta_v5_minus_v3_naive"])


def write_report(delta: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame) -> None:
    lines = []
    lines += [
        "# Stage 16-A2 QCR-U Robustness Check",
        "",
        "## 1. Purpose",
        "",
        "Stage 16-A1 showed that fixed quality-consistency fusion can improve the best protocol.",
        "",
        "Stage 16-A2 checks whether that gain is robust across all protocols, instead of only appearing in the best protocol.",
        "",
        "## 2. Overall Robustness Summary",
        "",
        "| Check | Wins | Total | Win Rate | Mean Delta | Median Delta | Min Delta | Max Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['check']} | {int(r['wins'])} | {int(r['total_protocols'])} | "
            f"{r['win_rate']:.4f} | {r['mean_delta']:.4f} | {r['median_delta']:.4f} | "
            f"{r['min_delta']:.4f} | {r['max_delta']:.4f} |"
        )

    lines += [
        "",
        "## 3. Protocol-level Deltas",
        "",
        "| Backbone | Strategy | Eval Mode | V5 AUROC | V3 AUROC | V4 AUROC | V5-V3 | V5-V4 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    display = delta.sort_values("delta_v5_minus_v3_naive", ascending=False)
    for _, r in display.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
            f"{r['V5']:.4f} | {r['V3']:.4f} | {r['V4']:.4f} | "
            f"{r['delta_v5_minus_v3_naive']:+.4f} | {r['delta_v5_minus_v4_quality']:+.4f} |"
        )

    lines += [
        "",
        "## 4. Failure / Weakness Cases",
        "",
    ]

    if failures.empty:
        lines.append("No failure case found under the current checks.")
    else:
        lines += [
            "| Backbone | Strategy | Eval Mode | V5-V3 | V5-V4 | V5-V0 | V5-V2 | Reason |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for _, r in failures.iterrows():
            lines.append(
                f"| {r['backbone']} | {r['strategy']} | {r['eval_mode']} | "
                f"{r['delta_v5_minus_v3_naive']:+.4f} | "
                f"{r['delta_v5_minus_v4_quality']:+.4f} | "
                f"{r['delta_v5_minus_v0_detector']:+.4f} | "
                f"{r['delta_v5_minus_v2_crop']:+.4f} | "
                f"{r['failure_reason']} |"
            )

    lines += [
        "",
        "## 5. Decision Rule",
        "",
        "If V5 is consistently better than V3 naive fusion but often worse than V4 quality-only, the consistency term should not be claimed as universally beneficial.",
        "",
        "In that case, the next method should be revised from fixed Q+C fusion to adaptive QCR-U:",
        "",
        "```text",
        "use quality-weighted crop as the stable core;",
        "apply consistency only when detector and VLM evidence are both reliable;",
        "avoid adding consistency under weak/full-image protocols where it hurts.",
        "```",
        "",
        "## 6. Outputs",
        "",
        f"- `{OUT_DELTA.relative_to(ROOT)}`",
        f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
        f"- `{OUT_FAILURES.relative_to(ROOT)}`",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    per_config = read_csv_robust(IN_PER_CONFIG)
    _ = read_csv_robust(IN_PER_CATEGORY)

    delta = compute_protocol_deltas(per_config)
    summary = make_summary(delta)
    failures = make_failures(delta)

    delta.to_csv(OUT_DELTA, index=False, lineterminator="\n")
    summary.to_csv(OUT_SUMMARY, index=False, lineterminator="\n")
    failures.to_csv(OUT_FAILURES, index=False, lineterminator="\n")

    write_report(delta, summary, failures)

    print("[DONE]", OUT_DELTA)
    print("[DONE]", OUT_SUMMARY)
    print("[DONE]", OUT_FAILURES)
    print("[DONE]", OUT_DOC)
    print()
    print("===== summary =====")
    print(summary.to_string(index=False))
    print()
    print("===== failures =====")
    if failures.empty:
        print("none")
    else:
        print(
            failures[
                [
                    "backbone",
                    "strategy",
                    "eval_mode",
                    "delta_v5_minus_v3_naive",
                    "delta_v5_minus_v4_quality",
                    "delta_v5_minus_v0_detector",
                    "delta_v5_minus_v2_crop",
                    "failure_reason",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
