from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()

IN_CLAIM_MAP = ROOT / "results/stage16_qcru_ablation/stage16_f_final_claim_evidence_map.csv"
IN_P2_CLAIM_USAGE = ROOT / "results/paper_p2/paper_p2_claim_usage_map.csv"
IN_P4_POSITIONING = ROOT / "results/paper_p4/paper_p4_positioning_map.csv"

OUT_DIR = ROOT / "results/paper_p5"
DOC_DIR = ROOT / "docs/paper_p5"

OUT_NOTATION = OUT_DIR / "paper_p5_notation_table.csv"
OUT_COMPONENTS = OUT_DIR / "paper_p5_method_components.csv"
OUT_ALGORITHM = OUT_DIR / "paper_p5_algorithm_steps.csv"
OUT_BOUNDARIES = OUT_DIR / "paper_p5_method_claim_boundaries.csv"
OUT_DOC = DOC_DIR / "paper_p5_method_full_draft.md"


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def build_notation() -> pd.DataFrame:
    rows = [
        {
            "symbol": "x",
            "name": "input image",
            "definition": "industrial test image",
            "range_or_type": "image",
            "used_in": "all steps",
        },
        {
            "symbol": "A",
            "name": "localization evidence",
            "definition": "detector-produced anomaly localization map or candidate evidence",
            "range_or_type": "map / spatial evidence",
            "used_in": "candidate generation",
        },
        {
            "symbol": "D",
            "name": "detector anomaly score",
            "definition": "normalized image-level anomaly score from detector evidence",
            "range_or_type": "[0, 1]",
            "used_in": "naive fusion; quality-calibrated QCR; adaptive refinement",
        },
        {
            "symbol": "C = {c_i}",
            "name": "candidate crop set",
            "definition": "candidate regions generated from localization evidence",
            "range_or_type": "set of image crops",
            "used_in": "crop-level VLM scoring",
        },
        {
            "symbol": "m_i",
            "name": "candidate VLM abnormality score",
            "definition": "VLM abnormality score for candidate crop c_i",
            "range_or_type": "[0, 1]",
            "used_in": "crop aggregation",
        },
        {
            "symbol": "M",
            "name": "aggregated crop VLM score",
            "definition": "fixed aggregation of crop-level VLM abnormality evidence under the selected protocol",
            "range_or_type": "[0, 1]",
            "used_in": "naive fusion; quality-calibrated QCR; adaptive refinement",
        },
        {
            "symbol": "Q",
            "name": "candidate quality",
            "definition": "localization-derived reliability of the candidate crop evidence",
            "range_or_type": "[0, 1]",
            "used_in": "quality calibration",
        },
        {
            "symbol": "K",
            "name": "detector-VLM high-high consistency",
            "definition": "consistency signal indicating jointly high detector and VLM abnormal evidence",
            "range_or_type": "[0, 1]",
            "used_in": "adaptive consistency refinement only",
        },
        {
            "symbol": "S_naive",
            "name": "naive detector-crop fusion score",
            "definition": "unreliability-aware baseline score",
            "range_or_type": "[0, 1]",
            "used_in": "ablation baseline",
        },
        {
            "symbol": "S_quality",
            "name": "Quality-Calibrated QCR score",
            "definition": "main quality-calibrated anomaly score",
            "range_or_type": "[0, 1] after normalization",
            "used_in": "main method core",
        },
        {
            "symbol": "S_adaptive",
            "name": "adaptive-refinement score",
            "definition": "quality-calibrated score plus conservative gated consistency bonus",
            "range_or_type": "[0, 1] after normalization",
            "used_in": "optional final refinement",
        },
    ]
    return pd.DataFrame(rows)


def build_components() -> pd.DataFrame:
    rows = [
        {
            "component_id": "M1",
            "component": "Detector localization",
            "role": "Generate localization evidence and detector anomaly score.",
            "main_formula_or_operation": "Detector(x) -> A, D",
            "paper_status": "use",
            "claim_boundary": "Detector is used as localization/evidence provider, not replaced.",
        },
        {
            "component_id": "M2",
            "component": "Candidate crop generation",
            "role": "Convert localization evidence into candidate regions for VLM scoring.",
            "main_formula_or_operation": "A -> C = {c_i}",
            "paper_status": "use",
            "claim_boundary": "Do not claim cropping alone is the main novelty.",
        },
        {
            "component_id": "M3",
            "component": "Crop-level VLM scoring",
            "role": "Obtain localized visual-language abnormality evidence.",
            "main_formula_or_operation": "VLM(c_i, prompts) -> m_i; aggregate {m_i} -> M",
            "paper_status": "use",
            "claim_boundary": "Do not claim full VLM anomaly understanding.",
        },
        {
            "component_id": "M4",
            "component": "Candidate quality calibration",
            "role": "Main effective method component; calibrates crop-level VLM evidence.",
            "main_formula_or_operation": "S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)",
            "paper_status": "main_method_core",
            "claim_boundary": "Quality calibration is the main contribution; not universal per category.",
        },
        {
            "component_id": "M5",
            "component": "Fixed Q+C fusion",
            "role": "Diagnostic comparison only.",
            "main_formula_or_operation": "S_fixed = 0.4D + 0.4M + 0.1Q + 0.1K",
            "paper_status": "diagnostic_only",
            "claim_boundary": "Do not use as final method; fixed consistency is not robust.",
        },
        {
            "component_id": "M6",
            "component": "Adaptive consistency refinement",
            "role": "Conservative optional refinement on top of quality-calibrated core.",
            "main_formula_or_operation": "S_adaptive = S_quality + 0.05 * Q*K*(1-|D-M|)*min(D,M)",
            "paper_status": "use_with_caution",
            "claim_boundary": "Small refinement only; not the main performance source.",
        },
    ]
    return pd.DataFrame(rows)


def build_algorithm() -> pd.DataFrame:
    rows = [
        {
            "step": 1,
            "name": "Detector inference",
            "input": "image x",
            "operation": "Run anomaly detector to obtain localization evidence A and detector anomaly score D.",
            "output": "A, D",
        },
        {
            "step": 2,
            "name": "Candidate generation",
            "input": "localization evidence A",
            "operation": "Extract candidate regions C = {c_i} using the fixed candidate protocol.",
            "output": "candidate crop set C",
        },
        {
            "step": 3,
            "name": "Crop-level VLM scoring",
            "input": "candidate crops C",
            "operation": "Score candidate crops with VLM abnormality prompts and aggregate crop scores.",
            "output": "aggregated crop VLM score M",
        },
        {
            "step": 4,
            "name": "Candidate quality estimation",
            "input": "localization/candidate evidence",
            "operation": "Compute candidate quality Q as localization-derived reliability of the crop evidence.",
            "output": "candidate quality Q",
        },
        {
            "step": 5,
            "name": "Quality-calibrated scoring",
            "input": "D, M, Q",
            "operation": "Compute S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q).",
            "output": "Quality-Calibrated QCR score",
        },
        {
            "step": 6,
            "name": "Optional adaptive consistency refinement",
            "input": "D, M, Q, K",
            "operation": "Compute gated consistency bonus and add it conservatively to S_quality.",
            "output": "S_adaptive",
        },
        {
            "step": 7,
            "name": "Image-level anomaly decision",
            "input": "S_quality or S_adaptive",
            "operation": "Use the selected score for image-level anomaly recognition and evaluation.",
            "output": "image-level anomaly score",
        },
    ]
    return pd.DataFrame(rows)


def build_boundaries() -> pd.DataFrame:
    rows = [
        {
            "boundary_id": "B1",
            "topic": "method name",
            "safe_statement": "Use Quality-Calibrated QCR as the main method family.",
            "forbidden_statement": "Use fixed Q+C QCR-U as the final method.",
        },
        {
            "boundary_id": "B2",
            "topic": "quality calibration",
            "safe_statement": "Candidate quality calibration is the main effective component.",
            "forbidden_statement": "Candidate quality improves every category and every case.",
        },
        {
            "boundary_id": "B3",
            "topic": "adaptive consistency",
            "safe_statement": "Adaptive consistency is a conservative refinement.",
            "forbidden_statement": "Adaptive consistency is the main source of improvement.",
        },
        {
            "boundary_id": "B4",
            "topic": "fixed consistency",
            "safe_statement": "Fixed Q+C is diagnostic because robustness is insufficient.",
            "forbidden_statement": "Fixed consistency is universally beneficial.",
        },
        {
            "boundary_id": "B5",
            "topic": "localization",
            "safe_statement": "Localization evidence is used for candidate generation and reliability calibration.",
            "forbidden_statement": "The method achieves pixel-level segmentation SOTA.",
        },
        {
            "boundary_id": "B6",
            "topic": "VLM reasoning",
            "safe_statement": "The method performs localization-guided VLM anomaly recognition.",
            "forbidden_statement": "The method explains manufacturing causes or full anomaly understanding.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    _ = read_csv_optional(IN_CLAIM_MAP)
    _ = read_csv_optional(IN_P2_CLAIM_USAGE)
    _ = read_csv_optional(IN_P4_POSITIONING)

    notation = build_notation()
    components = build_components()
    algorithm = build_algorithm()
    boundaries = build_boundaries()

    notation.to_csv(OUT_NOTATION, index=False, lineterminator="\n")
    components.to_csv(OUT_COMPONENTS, index=False, lineterminator="\n")
    algorithm.to_csv(OUT_ALGORITHM, index=False, lineterminator="\n")
    boundaries.to_csv(OUT_BOUNDARIES, index=False, lineterminator="\n")

    lines = []
    lines += [
        "# Paper Stage P5: Method Section Full Draft",
        "",
        "## 1. Method Overview",
        "",
        "We propose **Quality-Calibrated QCR**, a localization-guided VLM reasoning framework for image-level industrial anomaly recognition. "
        "The method starts from detector localization evidence, converts this evidence into candidate image crops, obtains crop-level VLM abnormality scores, and calibrates those scores using candidate quality. "
        "The key design principle is that crop-level VLM evidence should not be trusted uniformly: it should contribute strongly only when the candidate region is reliable.",
        "",
        "The method family contains two paper-facing variants. "
        "The main method core is **Quality-Calibrated QCR**, which uses candidate quality to calibrate crop-level VLM evidence. "
        "An optional full variant, **Quality-Calibrated QCR with adaptive consistency refinement**, adds a small reliability-gated consistency bonus. "
        "Fixed Q+C fusion is retained only as a diagnostic ablation and is not the final method.",
        "",
        "## 2. Notation",
        "",
        "| Symbol | Name | Definition | Range / Type |",
        "|---|---|---|---|",
    ]

    for _, r in notation.iterrows():
        lines.append(
            f"| `{r['symbol']}` | {r['name']} | {r['definition']} | {r['range_or_type']} |"
        )

    lines += [
        "",
        "## 3. Localization-guided Candidate Generation",
        "",
        "Given an input image `x`, an anomaly detector produces localization evidence `A` and a normalized detector anomaly score `D`. "
        "The localization evidence is used to generate a set of candidate crops:",
        "",
        "```text",
        "Detector(x) -> A, D",
        "A -> C = {c_i}",
        "```",
        "",
        "The candidate set focuses the VLM on spatial regions where abnormal evidence is likely to appear. "
        "This step is not claimed as the main novelty by itself. Its role is to convert detector localization into candidate-level visual evidence.",
        "",
        "## 4. Crop-level VLM Anomaly Evidence",
        "",
        "Each candidate crop `c_i` is evaluated by the VLM using abnormality-oriented prompts, producing a crop-level score `m_i`. "
        "The crop scores are aggregated under a fixed protocol to produce `M`, the aggregated crop-level VLM abnormality score:",
        "",
        "```text",
        "VLM(c_i, prompts) -> m_i",
        "Aggregate({m_i}) -> M",
        "```",
        "",
        "The aggregated score `M` is useful but not sufficient. A high VLM score can be unreliable if the crop is poorly localized, too broad, too small, or visually ambiguous. "
        "Therefore, the method calibrates VLM evidence using candidate quality.",
        "",
        "## 5. Candidate Quality Calibration",
        "",
        "Candidate quality `Q` measures the reliability of the selected candidate evidence. "
        "The naive detector-crop fusion baseline is:",
        "",
        "```text",
        "S_naive = 0.5D + 0.5M",
        "```",
        "",
        "This baseline treats detector and VLM evidence as equally reliable. "
        "Quality-Calibrated QCR instead modulates the VLM contribution by candidate quality:",
        "",
        "```text",
        "S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)",
        "```",
        "",
        "When `Q` is high, the crop-level VLM evidence contributes more strongly. "
        "When `Q` is low, the VLM contribution is reduced, preventing unreliable crops from dominating the image-level anomaly score. "
        "This quality-calibrated score is the main method core.",
        "",
        "## 6. Diagnostic Fixed Q+C Fusion",
        "",
        "We also evaluate a fixed quality-consistency fusion variant:",
        "",
        "```text",
        "S_fixed = 0.4D + 0.4M + 0.1Q + 0.1K",
        "```",
        "",
        "where `K` is a detector-VLM high-high consistency signal. "
        "This variant is useful diagnostically because it tests whether adding consistency can improve peak performance. "
        "However, robustness analysis shows that fixed consistency is not stable across protocols. "
        "Therefore, `S_fixed` is not used as the final method.",
        "",
        "## 7. Adaptive Consistency Refinement",
        "",
        "To avoid the instability of fixed consistency, the final refinement applies consistency only through a conservative reliability gate. "
        "We define:",
        "",
        "```text",
        "agreement = 1 - |D - M|",
        "mutual_anomaly_evidence = min(D, M)",
        "gate = Q * K * agreement * mutual_anomaly_evidence",
        "S_adaptive = S_quality + 0.05 * gate",
        "```",
        "",
        "The coefficient `0.05` is intentionally small. "
        "The adaptive term is not intended to be the main performance source. "
        "Its role is to add a consistency bonus only when candidate quality, detector evidence, VLM evidence, and detector-VLM agreement are jointly reliable.",
        "",
        "The full variant can be written as:",
        "",
        "```text",
        "Quality-Calibrated QCR with adaptive consistency refinement",
        "```",
        "",
        "but the core contribution remains candidate quality calibration.",
        "",
        "## 8. Algorithm",
        "",
        "```text",
        "Algorithm 1: Quality-Calibrated QCR",
        "",
        "Input:",
        "    image x",
        "    anomaly detector",
        "    VLM scoring function",
        "",
        "Output:",
        "    image-level anomaly score S",
        "",
        "1. Run detector on x to obtain localization evidence A and detector score D.",
        "2. Generate candidate crop set C = {c_i} from A using the fixed candidate protocol.",
        "3. Score each crop c_i with the VLM and aggregate crop scores into M.",
        "4. Estimate candidate quality Q from localization-derived candidate evidence.",
        "5. Compute the quality-calibrated score:",
        "       S_quality = 0.5D + 0.5 * M * (0.5 + 0.5Q)",
        "6. Optionally compute adaptive consistency refinement:",
        "       agreement = 1 - |D - M|",
        "       mutual_anomaly_evidence = min(D, M)",
        "       gate = Q * K * agreement * mutual_anomaly_evidence",
        "       S_adaptive = S_quality + 0.05 * gate",
        "7. Use S_quality as the main method score, or S_adaptive when reporting the adaptive-refinement variant.",
        "```",
        "",
        "## 9. Method Components",
        "",
        "| ID | Component | Role | Formula / Operation | Paper Status |",
        "|---|---|---|---|---|",
    ]

    for _, r in components.iterrows():
        lines.append(
            f"| {r['component_id']} | {r['component']} | {r['role']} | `{r['main_formula_or_operation']}` | {r['paper_status']} |"
        )

    lines += [
        "",
        "## 10. Claim Boundaries",
        "",
        "| ID | Topic | Safe Statement | Forbidden Statement |",
        "|---|---|---|---|",
    ]

    for _, r in boundaries.iterrows():
        lines.append(
            f"| {r['boundary_id']} | {r['topic']} | {r['safe_statement']} | {r['forbidden_statement']} |"
        )

    lines += [
        "",
        "## 11. Method Section Writing Notes",
        "",
        "The method section should emphasize the following:",
        "",
        "- The paper is not a detector replacement paper.",
        "- The paper is not a generic VLM anomaly understanding paper.",
        "- The main contribution is reliability calibration of localization-guided VLM evidence.",
        "- Candidate quality is the main effective component.",
        "- Adaptive consistency is a conservative refinement.",
        "- Fixed Q+C is a diagnostic ablation, not the final method.",
        "",
        "## 12. Next Step",
        "",
        "Next stage:",
        "",
        "```text",
        "Paper Stage P6: assemble first full paper draft skeleton",
        "```",
        "",
        "P6 should combine P2 Introduction/Contributions, P4 Related Work, P5 Method, and P3 Experiments into a single paper skeleton.",
        "",
        "## 13. Outputs",
        "",
        f"- `{OUT_DOC.relative_to(ROOT)}`",
        f"- `{OUT_NOTATION.relative_to(ROOT)}`",
        f"- `{OUT_COMPONENTS.relative_to(ROOT)}`",
        f"- `{OUT_ALGORITHM.relative_to(ROOT)}`",
        f"- `{OUT_BOUNDARIES.relative_to(ROOT)}`",
        "",
    ]

    OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("[DONE]", OUT_DOC)
    print("[DONE]", OUT_NOTATION)
    print("[DONE]", OUT_COMPONENTS)
    print("[DONE]", OUT_ALGORITHM)
    print("[DONE]", OUT_BOUNDARIES)


if __name__ == "__main__":
    main()
