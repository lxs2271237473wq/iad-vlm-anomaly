from __future__ import annotations

from pathlib import Path
from io import StringIO
import re
import pandas as pd


ROOT = Path(".").resolve()

IN_STAGE15E = ROOT / "results/stage15_modern_detector_baselines/stage15_e_primary_unified_baseline_comparison.csv"
IN_STAGE16B_PRIMARY = ROOT / "results/stage16_qcru_ablation/stage16_b_adaptive_qcru_primary_protocol_table.csv"
IN_STAGE16B_DECISION = ROOT / "results/stage16_qcru_ablation/stage16_b_adaptive_qcru_final_method_decision.csv"
IN_STAGE16C_CLAIMS = ROOT / "results/stage16_qcru_ablation/stage16_c_final_method_claims.csv"

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"

OUT_SYSTEM = OUT_DIR / "stage16_d_paper_facing_system_baseline_table.csv"
OUT_QCR = OUT_DIR / "stage16_d_paper_facing_qcr_ablation_table.csv"
OUT_DELTAS = OUT_DIR / "stage16_d_paper_facing_claim_ready_deltas.csv"
OUT_DOC = DOC_DIR / "stage16_d_paper_facing_final_comparison_report.md"


STAGE15_HEADER = "category,method_group,method,image_auroc,image_ap,pixel_auroc,pixel_f1,protocol,fairness_tag"


def read_csv_robust(path: Path, known_header: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    raw = path.read_text(encoding="utf-8").strip()

    if known_header is not None and raw.startswith(known_header) and "\n" not in raw:
        body = raw[len(known_header):].strip()
        rows = re.split(
            r"\s+(?=(?:fruit_jelly|sheet_metal|vial|walnuts|MEAN),)",
            body,
        )
        rows = [r.strip() for r in rows if r.strip()]
        raw = known_header + "\n" + "\n".join(rows) + "\n"

    df = pd.read_csv(StringIO(raw))
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair local CSV formatting first.")
    return df


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_system_table(stage15e: pd.DataFrame) -> pd.DataFrame:
    df = stage15e.copy()
    df = to_num(df, ["image_auroc", "image_ap", "pixel_auroc", "pixel_f1"])

    mean = df[df["category"] == "MEAN"].copy()
    if mean.empty:
        raise RuntimeError("Stage 15-E has no MEAN rows.")

    rows = []
    for _, r in mean.iterrows():
        method = str(r["method"])
        fairness_tag = str(r.get("fairness_tag", ""))
        protocol = str(r.get("protocol", ""))

        if "same-set" in method:
            paper_role = "upper_bound_diagnostic_only"
            use_in_main_claim = False
        elif "LOCO" in method:
            paper_role = "primary_fair_system_result"
            use_in_main_claim = True
        elif "EfficientAD" in method:
            paper_role = "modern_detector_fixed_budget_baseline"
            use_in_main_claim = True
        elif "WinCLIP" in method:
            paper_role = "external_vlm_anomaly_baseline"
            use_in_main_claim = True
        elif method == "PatchCore":
            paper_role = "classic_detector_baseline"
            use_in_main_claim = True
        elif "VLM" in method:
            paper_role = "vlm_baseline"
            use_in_main_claim = True
        else:
            paper_role = "baseline"
            use_in_main_claim = True

        rows.append(
            {
                "panel": "A_system_level_strong_baselines",
                "method": method,
                "mean_image_auroc": float(r["image_auroc"]),
                "mean_image_ap": r.get("image_ap", ""),
                "mean_pixel_auroc": r.get("pixel_auroc", ""),
                "mean_pixel_f1": r.get("pixel_f1", ""),
                "protocol": protocol,
                "fairness_tag": fairness_tag,
                "paper_role": paper_role,
                "use_in_main_claim": use_in_main_claim,
                "comparison_scope": "Stage15 primary four-category system comparison",
                "directly_comparable_with_qcr_panel": False,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("mean_image_auroc", ascending=False).reset_index(drop=True)
    out["rank_by_mean_image_auroc"] = range(1, len(out) + 1)
    return out


def rename_qcr_variant(variant_id: str, variant: str) -> tuple[str, str, bool]:
    if variant_id == "V0":
        return "Detector only", "anchor_baseline", True
    if variant_id == "V2":
        return "Crop VLM only", "vlm_crop_baseline", True
    if variant_id == "V3":
        return "Naive detector-crop fusion", "naive_fusion_baseline", True
    if variant_id == "V4":
        return "Quality-Calibrated QCR", "main_effective_method_core", True
    if variant_id == "V5":
        return "Fixed Q+C fusion", "diagnostic_not_final", False
    if variant_id == "V6":
        return "Quality-Calibrated QCR + adaptive consistency refinement", "final_refinement_variant", True
    return variant, "other", True


def build_qcr_table(primary: pd.DataFrame) -> pd.DataFrame:
    df = primary.copy()
    df = to_num(df, ["auroc", "ap", "best_f1", "best_accuracy", "best_threshold"])

    rows = []
    for _, r in df.iterrows():
        display_name, paper_role, use_in_main_claim = rename_qcr_variant(
            str(r["variant_id"]),
            str(r["variant"]),
        )

        rows.append(
            {
                "panel": "B_qcr_primary_protocol_ablation",
                "backbone": r["backbone"],
                "dataset": r["dataset"],
                "strategy": r["strategy"],
                "eval_mode": r["eval_mode"],
                "variant_id": r["variant_id"],
                "method": display_name,
                "source_variant_name": r["variant"],
                "image_auroc": float(r["auroc"]),
                "image_ap": float(r["ap"]),
                "best_f1": float(r["best_f1"]),
                "best_accuracy": float(r["best_accuracy"]),
                "paper_role": paper_role,
                "use_in_main_claim": use_in_main_claim,
                "comparison_scope": "Stage16-B QCR primary protocol",
                "directly_comparable_with_system_panel": False,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["backbone", "variant_id"]).reset_index(drop=True)
    return out


def build_claim_deltas(system_table: pd.DataFrame, qcr_table: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def system_score(method: str) -> float | None:
        r = system_table[system_table["method"] == method]
        if r.empty:
            return None
        return float(r.iloc[0]["mean_image_auroc"])

    loco = system_score("PatchCore + context VLM, LOCO")
    same = system_score("PatchCore + context VLM, same-set")
    patch = system_score("PatchCore")
    ead = system_score("EfficientAD-30 fixed-budget")
    winclip = system_score("WinCLIP fixed protocol")
    context = system_score("context-aware VLM")
    full = system_score("full-image VLM")

    system_pairs = [
        ("LOCO fusion vs PatchCore", loco, patch, "system_level_main_delta"),
        ("LOCO fusion vs EfficientAD-30 fixed-budget", loco, ead, "system_level_main_delta"),
        ("LOCO fusion vs WinCLIP fixed protocol", loco, winclip, "system_level_main_delta"),
        ("LOCO fusion vs context-aware VLM", loco, context, "system_level_main_delta"),
        ("context-aware VLM vs full-image VLM", context, full, "vlm_localization_delta"),
        ("same-set upper bound vs LOCO fair result", same, loco, "upper_bound_gap"),
    ]

    for name, left, right, delta_type in system_pairs:
        if left is None or right is None:
            continue
        rows.append(
            {
                "delta_type": delta_type,
                "scope": "system_panel",
                "comparison": name,
                "left_score": left,
                "right_score": right,
                "delta": left - right,
                "paper_interpretation": interpret_system_delta(name),
            }
        )

    # QCR primary protocol mean over backbones.
    piv = qcr_table.pivot_table(
        index=["dataset", "strategy", "eval_mode", "backbone"],
        columns="variant_id",
        values="image_auroc",
        aggfunc="first",
    ).reset_index()
    piv.columns.name = None

    qcr_pairs = [
        ("Quality-Calibrated QCR vs naive fusion", "V4", "V3", "qcr_core_delta"),
        ("Adaptive refinement vs Quality-Calibrated QCR", "V6", "V4", "adaptive_refinement_delta"),
        ("Adaptive refinement vs naive fusion", "V6", "V3", "qcr_final_delta"),
        ("Fixed Q+C vs Quality-Calibrated QCR", "V5", "V4", "diagnostic_fixed_consistency_delta"),
        ("Adaptive refinement vs fixed Q+C", "V6", "V5", "robustness_tradeoff_delta"),
    ]

    for name, left_col, right_col, delta_type in qcr_pairs:
        if left_col not in piv.columns or right_col not in piv.columns:
            continue
        d = piv[left_col] - piv[right_col]
        rows.append(
            {
                "delta_type": delta_type,
                "scope": "qcr_primary_protocol",
                "comparison": name,
                "left_score": float(piv[left_col].mean()),
                "right_score": float(piv[right_col].mean()),
                "delta": float(d.mean()),
                "paper_interpretation": interpret_qcr_delta(name, float(d.mean())),
            }
        )

    # Include Stage 16-B decision rows as evidence, but not as final table entries.
    if not decision.empty:
        for _, r in decision.iterrows():
            rows.append(
                {
                    "delta_type": "stage16b_decision_summary",
                    "scope": r.get("scope", ""),
                    "comparison": r.get("comparison", ""),
                    "left_score": "",
                    "right_score": "",
                    "delta": r.get("mean_delta", ""),
                    "paper_interpretation": (
                        f"wins={r.get('wins', '')}/{r.get('num_protocols', '')}, "
                        f"win_rate={r.get('win_rate', '')}"
                    ),
                }
            )

    out = pd.DataFrame(rows)
    return out


def interpret_system_delta(name: str) -> str:
    if "EfficientAD" in name:
        return "LOCO fusion remains above the fixed-budget modern detector baseline; do not claim full EfficientAD defeat."
    if "PatchCore" in name:
        return "Localization-guided VLM evidence complements the detector baseline."
    if "WinCLIP" in name:
        return "The proposed localization-guided route is stronger than this fixed WinCLIP protocol."
    if "full-image" in name:
        return "Localization/context improves over full-image VLM."
    if "upper bound" in name:
        return "Same-set is diagnostic upper bound only; LOCO is the fair result."
    return "Claim-supporting delta."


def interpret_qcr_delta(name: str, delta: float) -> str:
    if "Quality-Calibrated QCR vs naive" in name:
        return "Candidate quality calibration is the main method gain."
    if "Adaptive refinement vs Quality" in name:
        if abs(delta) < 0.005:
            return "Adaptive consistency is only a small refinement, not a main contribution."
        return "Adaptive consistency provides a meaningful refinement."
    if "Adaptive refinement vs naive" in name:
        return "Final refinement variant improves over naive fusion."
    if "Fixed Q+C" in name:
        return "Fixed consistency is diagnostic only because robustness is not stable across protocols."
    if "Adaptive refinement vs fixed" in name:
        return "Adaptive refinement trades peak primary-protocol AUROC for robustness."
    return "QCR delta."


def write_report(
    system_table: pd.DataFrame,
    qcr_table: pd.DataFrame,
    deltas: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    lines = []
    lines += [
        "# Stage 16-D Paper-facing Final Comparison",
        "",
        "## 1. Purpose",
        "",
        "This stage creates the final paper-facing comparison tables after the method claim was locked in Stage 16-C.",
        "",
        "The final method family is:",
        "",
        "```text",
        "Quality-Calibrated QCR",
        "```",
        "",
        "The adaptive consistency term is treated only as a conservative refinement, not as the main performance source.",
        "",
        "## 2. Important Comparison Rule",
        "",
        "This report uses two panels because Stage 15 system baselines and Stage 16 QCR ablations are not the same protocol.",
        "",
        "- Panel A compares system-level baselines from Stage 15.",
        "- Panel B compares QCR variants under the Stage 16-B QCR primary protocol.",
        "",
        "Do not merge the two panels into a single global ranking.",
        "",
        "## 3. Panel A: System-level Strong Baseline Comparison",
        "",
        "| Rank | Method | Mean Image AUROC | Role | Fairness Tag |",
        "|---:|---|---:|---|---|",
    ]

    for _, r in system_table.iterrows():
        lines.append(
            f"| {int(r['rank_by_mean_image_auroc'])} | {r['method']} | "
            f"{float(r['mean_image_auroc']):.4f} | {r['paper_role']} | {r['fairness_tag']} |"
        )

    lines += [
        "",
        "Paper use:",
        "",
        "- Use `PatchCore + context VLM, LOCO` as the fair system-level result.",
        "- Use `same-set` only as an upper-bound diagnostic.",
        "- Keep `EfficientAD-30` explicitly labeled as fixed-budget.",
        "",
        "## 4. Panel B: QCR Primary-protocol Ablation",
        "",
        "| Backbone | Method | Variant | Image AUROC | AP | Best F1 | Role |",
        "|---|---|---|---:|---:|---:|---|",
    ]

    for _, r in qcr_table.iterrows():
        lines.append(
            f"| {r['backbone']} | {r['method']} | {r['variant_id']} | "
            f"{float(r['image_auroc']):.4f} | {float(r['image_ap']):.4f} | "
            f"{float(r['best_f1']):.4f} | {r['paper_role']} |"
        )

    lines += [
        "",
        "Paper use:",
        "",
        "- Treat `Quality-Calibrated QCR` as the main effective method core.",
        "- Treat `Quality-Calibrated QCR + adaptive consistency refinement` as the final conservative refinement.",
        "- Treat `Fixed Q+C fusion` as diagnostic only, because it is not robust across protocols.",
        "",
        "## 5. Claim-ready Deltas",
        "",
        "| Scope | Comparison | Left Score | Right Score | Delta | Interpretation |",
        "|---|---|---:|---:|---:|---|",
    ]

    for _, r in deltas.iterrows():
        left = r["left_score"]
        right = r["right_score"]
        delta = r["delta"]

        left_s = "" if left == "" else f"{float(left):.4f}"
        right_s = "" if right == "" else f"{float(right):.4f}"
        try:
            delta_s = f"{float(delta):+.4f}"
        except Exception:
            delta_s = str(delta)

        lines.append(
            f"| {r['scope']} | {r['comparison']} | {left_s} | {right_s} | "
            f"{delta_s} | {r['paper_interpretation']} |"
        )

    lines += [
        "",
        "## 6. Final Paper Claims",
        "",
    ]

    if claims.empty:
        lines.append("Stage 16-C claims file was not available.")
    else:
        lines += [
            "| Claim ID | Type | Claim | Status |",
            "|---|---|---|---|",
        ]
        for _, r in claims.iterrows():
            lines.append(
                f"| {r['claim_id']} | {r['claim_type']} | {r['claim']} | {r['paper_status']} |"
            )

    lines += [
        "",
        "## 7. Safe Main Claim",
        "",
        "Use this as the central claim:",
        "",
        "```text",
        "Localization-guided VLM anomaly recognition becomes more reliable when crop-level VLM evidence is calibrated by candidate quality. Adaptive consistency is retained as a conservative refinement, but the main effective component is candidate quality calibration.",
        "```",
        "",
        "## 8. Claims to Avoid",
        "",
        "- Do not claim fixed Q+C fusion as the final method.",
        "- Do not claim consistency is universally beneficial.",
        "- Do not claim adaptive consistency is the main source of improvement.",
        "- Do not claim full industrial anomaly understanding.",
        "- Do not claim pixel-level segmentation SOTA.",
        "",
        "## 9. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Stage 16-E: failure cases and boundary analysis",
        "```",
        "",
        "Stage 16-E should explain where quality calibration helps, where fixed consistency fails, and where detector localization errors mislead VLM reasoning.",
        "",
        "## 10. Outputs",
        "",
        f"- `{OUT_SYSTEM.relative_to(ROOT)}`",
        f"- `{OUT_QCR.relative_to(ROOT)}`",
        f"- `{OUT_DELTAS.relative_to(ROOT)}`",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    stage15e = read_csv_robust(IN_STAGE15E, STAGE15_HEADER)
    primary = read_csv_robust(IN_STAGE16B_PRIMARY)
    decision = read_csv_robust(IN_STAGE16B_DECISION)
    claims = read_csv_robust(IN_STAGE16C_CLAIMS)

    system_table = build_system_table(stage15e)
    qcr_table = build_qcr_table(primary)
    deltas = build_claim_deltas(system_table, qcr_table, decision)

    system_table.to_csv(OUT_SYSTEM, index=False, lineterminator="\n")
    qcr_table.to_csv(OUT_QCR, index=False, lineterminator="\n")
    deltas.to_csv(OUT_DELTAS, index=False, lineterminator="\n")

    write_report(system_table, qcr_table, deltas, claims)

    print("[DONE]", OUT_SYSTEM)
    print("[DONE]", OUT_QCR)
    print("[DONE]", OUT_DELTAS)
    print("[DONE]", OUT_DOC)
    print()
    print("===== Panel A: system baselines =====")
    print(system_table[["rank_by_mean_image_auroc", "method", "mean_image_auroc", "paper_role"]].to_string(index=False))
    print()
    print("===== Panel B: QCR ablation =====")
    print(qcr_table[["backbone", "variant_id", "method", "image_auroc", "paper_role"]].to_string(index=False))
    print()
    print("===== claim-ready deltas =====")
    print(deltas[["scope", "comparison", "delta", "paper_interpretation"]].to_string(index=False))


if __name__ == "__main__":
    main()
