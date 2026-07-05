from pathlib import Path
import pandas as pd

ROOT = Path(".").resolve()

OUT_DIR = ROOT / "results/stage16_qcru_ablation"
DOC_DIR = ROOT / "docs/stage16_qcru_ablation"
OUT_CSV = OUT_DIR / "stage16_c_final_method_claims.csv"
OUT_DOC = DOC_DIR / "stage16_c_final_method_claims_report.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "claim_id": "C1",
        "claim_type": "final_method_name",
        "claim": "Use Quality-Calibrated QCR as the main method name.",
        "evidence": "Stage 16-B shows adaptive QCR-U improves over quality-only by only about +0.0004 AUROC in the primary protocol.",
        "paper_status": "use",
    },
    {
        "claim_id": "C2",
        "claim_type": "main_effective_component",
        "claim": "Candidate quality calibration is the main effective component.",
        "evidence": "Quality-weighted crop improves over naive fusion by about +0.0096 AUROC in the primary protocol.",
        "paper_status": "use",
    },
    {
        "claim_id": "C3",
        "claim_type": "auxiliary_component",
        "claim": "Adaptive consistency is a conservative refinement, not the main source of improvement.",
        "evidence": "Adaptive QCR-U beats quality-only in the primary protocol, but the gain is only about +0.0004 AUROC.",
        "paper_status": "use_with_caution",
    },
    {
        "claim_id": "C4",
        "claim_type": "rejected_claim",
        "claim": "Do not claim fixed quality-consistency fusion as the final method.",
        "evidence": "Fixed consistency was not robust across all protocols in Stage 16-A2, despite high primary-protocol AUROC.",
        "paper_status": "reject",
    },
    {
        "claim_id": "C5",
        "claim_type": "rejected_claim",
        "claim": "Do not claim consistency is universally beneficial.",
        "evidence": "Stage 16-A2 showed fixed consistency beats quality-only in only half of protocols.",
        "paper_status": "reject",
    },
    {
        "claim_id": "C6",
        "claim_type": "safe_paper_claim",
        "claim": "Localization-guided VLM reasoning becomes substantially stronger when crop evidence is calibrated by candidate quality.",
        "evidence": "Stage 16-B primary protocol shows quality-weighted crop consistently improves over naive fusion for both FastFlow and PatchCore.",
        "paper_status": "use",
    },
    {
        "claim_id": "C7",
        "claim_type": "safe_paper_claim",
        "claim": "The proposed method should be positioned as reliability calibration for localization-guided VLM anomaly recognition.",
        "evidence": "Stage 15 and Stage 16 together show strong baselines plus ablations support reliability calibration better than a pure consistency-fusion story.",
        "paper_status": "use",
    },
]

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, lineterminator="\n")

lines = [
    "# Stage 16-C Final Method Claims",
    "",
    "## 1. Decision",
    "",
    "The final method should not be written as fixed QCR-U or as a consistency-driven method.",
    "",
    "The paper-facing method name should be:",
    "",
    "```text",
    "Quality-Calibrated QCR",
    "```",
    "",
    "A longer descriptive name can be:",
    "",
    "```text",
    "Quality-Calibrated Localization-Guided VLM Reasoning",
    "```",
    "",
    "Adaptive consistency can be retained only as a small refinement:",
    "",
    "```text",
    "Quality-Calibrated QCR with Adaptive Consistency Refinement",
    "```",
    "",
    "## 2. Why this downgrade is necessary",
    "",
    "Stage 16-B shows that adaptive QCR-U is consistently better than naive fusion, but almost all useful gain over naive fusion comes from quality calibration.",
    "",
    "The adaptive consistency term improves over quality-only by only a very small margin. Therefore, consistency cannot be claimed as the main innovation.",
    "",
    "## 3. Final Claims Table",
    "",
    "| Claim ID | Type | Claim | Paper Status |",
    "|---|---|---|---|",
]

for _, r in df.iterrows():
    lines.append(
        f"| {r['claim_id']} | {r['claim_type']} | {r['claim']} | {r['paper_status']} |"
    )

lines += [
    "",
    "## 4. Safe contribution wording",
    "",
    "The safest contribution wording is:",
    "",
    "```text",
    "We propose a quality-calibrated localization-guided VLM reasoning framework for industrial anomaly recognition. Instead of directly fusing detector and VLM scores, the method calibrates crop-level VLM evidence using candidate quality derived from anomaly localization. We further study detector-VLM consistency and find that fixed consistency is not robust; therefore, consistency is used only as a conservative adaptive refinement.",
    "```",
    "",
    "## 5. Claims to avoid",
    "",
    "- Do not claim consistency is universally beneficial.",
    "- Do not claim fixed Q+C fusion is the final method.",
    "- Do not claim adaptive consistency is the main source of improvement.",
    "- Do not claim this solves full industrial anomaly understanding.",
    "- Do not claim manufacturing-cause reasoning.",
    "",
    "## 6. Next step",
    "",
    "Next stage should generate the final paper-facing main table using this method naming:",
    "",
    "```text",
    "Stage 16-D: paper-facing final comparison table",
    "```",
    "",
    "That table should compare:",
    "",
    "- WinCLIP fixed protocol",
    "- EfficientAD-30 fixed-budget",
    "- PatchCore",
    "- detector-only",
    "- crop VLM",
    "- naive fusion",
    "- quality-calibrated QCR",
    "- quality-calibrated QCR + adaptive consistency refinement",
    "",
]

OUT_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")

print("[DONE]", OUT_CSV)
print("[DONE]", OUT_DOC)
print(df.to_string(index=False))
