from pathlib import Path
import pandas as pd

DOC_DIR = Path("docs/stage14_strong_vlm_baselines")
RES_DIR = Path("results/stage14_strong_vlm_baselines")

DOC_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

csv_path = RES_DIR / "stage14_a_strong_baseline_selection_table.csv"
doc_path = DOC_DIR / "stage14_a_strong_baseline_landscape.md"

rows = [
    {
        "method": "PatchCore",
        "group": "Classical anomaly detector",
        "priority": "keep",
        "role_in_our_project": "Existing detector reference and localization source",
        "why_compare": "Classic strong anomaly localization baseline; already used in Stage 11 and Stage 13.",
        "expected_effort": "done",
        "next_action": "Keep as traditional detector baseline, but not as the only strong baseline.",
    },
    {
        "method": "PatchCore + context VLM fusion",
        "group": "Our current fusion method",
        "priority": "main_current",
        "role_in_our_project": "Current strongest internal result",
        "why_compare": "Stage 13-A shows that context VLM can provide complementary signal to PatchCore.",
        "expected_effort": "done",
        "next_action": "Use as current method to compare against WinCLIP and later VLM baselines.",
    },
    {
        "method": "WinCLIP",
        "group": "Vision-language anomaly detection",
        "priority": "first_external_vlm_baseline",
        "role_in_our_project": "First strong VLM anomaly detection baseline",
        "why_compare": "CLIP-based zero-/few-shot anomaly classification and segmentation; directly related to our VLM branch.",
        "expected_effort": "medium",
        "next_action": "Check Anomalib WinCLIP API, then run on AD2 primary categories if compatible.",
    },
    {
        "method": "AnomalyCLIP",
        "group": "Vision-language anomaly detection",
        "priority": "second_external_vlm_baseline",
        "role_in_our_project": "Stronger CLIP-adaptation baseline",
        "why_compare": "Learns object-agnostic normal/abnormal prompts and targets the weakness of vanilla CLIP in anomaly detection.",
        "expected_effort": "high",
        "next_action": "Add literature discussion first; reproduce if WinCLIP stage succeeds.",
    },
    {
        "method": "EfficientAD",
        "group": "Modern classical anomaly detector",
        "priority": "modern_detector_baseline",
        "role_in_our_project": "Modern non-VLM detector baseline",
        "why_compare": "Recent efficient detector with strong detection/localization and practical latency claims.",
        "expected_effort": "medium",
        "next_action": "Check Anomalib EfficientAD support after WinCLIP baseline.",
    },
    {
        "method": "FastFlow",
        "group": "Flow-based anomaly detector",
        "priority": "optional_detector_baseline",
        "role_in_our_project": "Alternative front-end detector",
        "why_compare": "Useful to test whether the context VLM branch is only tied to PatchCore.",
        "expected_effort": "medium",
        "next_action": "Use after WinCLIP or EfficientAD depending on time.",
    },
    {
        "method": "VCP-CLIP / FADE / newer VLM-AD methods",
        "group": "Recent VLM anomaly detection",
        "priority": "related_work_or_later_reproduction",
        "role_in_our_project": "Stronger recent related work",
        "why_compare": "Represents newer VLM anomaly detection direction; useful for literature framing and later extension.",
        "expected_effort": "high",
        "next_action": "Add to related work; reproduce only if time and environment allow.",
    },
]

df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)

lines = []
lines += [
    "# Stage 14-A Strong Baseline Landscape",
    "",
    "## 1. Purpose",
    "",
    "Stage 14 is introduced because comparing only with full-image VLM or only with PatchCore is not sufficient.",
    "",
    "The new experimental question is:",
    "",
    "```text",
    "Does our context-aware VLM branch remain useful when compared with successful anomaly detection directions, especially WinCLIP-style VLM anomaly detection and modern strong detectors?",
    "```",
    "",
    "## 2. Why Stage 14 is necessary",
    "",
    "Stage 13-A showed that PatchCore + context-aware VLM fusion improves over PatchCore alone on ALL_PRIMARY.",
    "",
    "However, PatchCore is a classic detector and cannot represent the whole current anomaly detection landscape. A stronger paper must compare against:",
    "",
    "1. classical strong anomaly detectors;",
    "2. modern efficient anomaly detectors;",
    "3. dedicated vision-language anomaly detection methods;",
    "4. our own PatchCore + context VLM fusion result.",
    "",
    "## 3. Baseline Selection Table",
    "",
    "| Method | Group | Priority | Role | Next Action |",
    "|---|---|---|---|---|",
]

for row in rows:
    lines.append(
        f"| {row['method']} | {row['group']} | {row['priority']} | "
        f"{row['role_in_our_project']} | {row['next_action']} |"
    )

lines += [
    "",
    "## 4. Immediate Decision",
    "",
    "The first external VLM anomaly detection baseline should be WinCLIP.",
    "",
    "Reasons:",
    "",
    "1. It is directly related to CLIP-based anomaly classification and segmentation.",
    "2. It is implemented in Anomalib, so the reproduction cost is lower.",
    "3. It answers the reviewer concern: why compare only against weak full-image CLIP?",
    "4. If our PatchCore + context VLM fusion is competitive against WinCLIP, the paper becomes much stronger.",
    "",
    "## 5. Next Step",
    "",
    "Stage 14-B should check whether the current environment can import and instantiate WinCLIP from Anomalib.",
    "",
    "If WinCLIP is available, Stage 14-C should run a small pilot on one AD2 primary category, preferably fruit_jelly or vial.",
    "",
    "If WinCLIP is not available in the installed Anomalib version, Stage 14-B should document the missing dependency/API issue and decide whether to upgrade Anomalib or use an external WinCLIP implementation.",
]

doc_path.write_text("\n".join(lines), encoding="utf-8")

print("[DONE]", csv_path)
print("[DONE]", doc_path)
print(df.to_string(index=False))
