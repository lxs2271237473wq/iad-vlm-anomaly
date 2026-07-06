from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


ROOT = Path(".").resolve()

PAPER_DIR = ROOT / "paper/quality_calibrated_qcr"
IN_MAIN = PAPER_DIR / "main.tex"
OUT_POLISHED = PAPER_DIR / "main_p14_polished.tex"

OUT_DIR = ROOT / "results/paper_p14"
DOC_DIR = ROOT / "docs/paper_p14"

OUT_PATCH_INV = OUT_DIR / "paper_p14_polish_patch_inventory.csv"
OUT_SAFETY = OUT_DIR / "paper_p14_claim_safety_scan.csv"
OUT_REPORT = DOC_DIR / "paper_p14_language_polish_report.md"


FORBIDDEN = [
    "state-of-the-art segmentation",
    "SOTA segmentation",
    "manufacturing cause",
    "manufacturing-cause reasoning",
    "full anomaly understanding",
    "universally beneficial",
    "defeat EfficientAD",
    "beats EfficientAD",
    "outperforms AnomalyCLIP",
    "SOTA",
]


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def count_words_latex(text: str) -> int:
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[^A-Za-z0-9\-]+", " ", text)
    return len([w for w in text.split() if w.strip()])


def section_word_counts(tex: str) -> pd.DataFrame:
    rows = []
    parts = re.split(r"(\\section\{[^}]+\})", tex)

    preamble = parts[0]
    rows.append(
        {
            "section": "preamble_and_abstract",
            "word_count": count_words_latex(preamble),
        }
    )

    for i in range(1, len(parts), 2):
        heading = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        name = heading.replace("\\section{", "").replace("}", "")
        rows.append(
            {
                "section": name,
                "word_count": count_words_latex(body),
            }
        )

    return pd.DataFrame(rows)


def apply_polish(tex: str) -> tuple[str, pd.DataFrame]:
    patches = [
        {
            "patch_id": "P14-01",
            "type": "tone_softening",
            "old": "remains unreliable when defects are small, localized, and visually subtle",
            "new": "can be unreliable when defects are small, localized, and visually subtle",
            "reason": "Avoid overgeneralizing VLM unreliability.",
        },
        {
            "patch_id": "P14-02",
            "type": "tone_softening",
            "old": "This makes direct full-image VLM reasoning unreliable for industrial inspection.",
            "new": "This makes direct full-image VLM reasoning difficult for industrial inspection.",
            "reason": "Sharper and less absolute.",
        },
        {
            "patch_id": "P14-03",
            "type": "claim_precision",
            "old": "The main method is $S_{\\mathrm{quality}}$.",
            "new": "We use $S_{\\mathrm{quality}}$ as the main method score.",
            "reason": "Clarifies what is reported as the main method.",
        },
        {
            "patch_id": "P14-04",
            "type": "claim_precision",
            "old": "The adaptive variant is reported as a refinement, not as the main performance source.",
            "new": "The adaptive variant is reported only as a refinement.",
            "reason": "More compact and conservative.",
        },
        {
            "patch_id": "P14-05",
            "type": "compression",
            "old": "The evidence supports a cautious but coherent claim: localization-guided VLM anomaly recognition becomes more reliable when crop-level evidence is calibrated by candidate quality.",
            "new": "Overall, candidate-quality calibration makes localization-guided VLM evidence more reliable for image-level anomaly recognition.",
            "reason": "Tighter conclusion.",
        },
        {
            "patch_id": "P14-06",
            "type": "claim_safety",
            "old": "It does not claim pixel-level segmentation SOTA, general defect-cause interpretation, or general defect-cause interpretation.",
            "new": "It does not claim pixel-level segmentation SOTA or general defect-cause interpretation.",
            "reason": "Remove possible duplicated limitation wording.",
        },
        {
            "patch_id": "P14-07",
            "type": "claim_safety",
            "old": "It does not claim pixel-level segmentation SOTA, manufacturing-cause reasoning, or general defect-cause interpretation.",
            "new": "It does not claim pixel-level segmentation SOTA or general defect-cause interpretation.",
            "reason": "Remove forbidden wording while preserving limitation.",
        },
        {
            "patch_id": "P14-08",
            "type": "baseline_precision",
            "old": "EfficientAD is fixed-budget",
            "new": "EfficientAD is evaluated under a fixed 30-epoch budget",
            "reason": "Make baseline budget explicit.",
        },
    ]

    out = tex
    rows = []

    for p in patches:
        before = out
        out = out.replace(p["old"], p["new"])
        applied = before != out

        rows.append(
            {
                "patch_id": p["patch_id"],
                "type": p["type"],
                "applied": applied,
                "reason": p["reason"],
                "old": p["old"],
                "new": p["new"],
            }
        )

    # Normalize multiple spaces introduced by edits, but avoid touching LaTeX commands too aggressively.
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{4,}", "\n\n\n", out)

    return out, pd.DataFrame(rows)


def safety_scan(tex: str) -> pd.DataFrame:
    rows = []

    lower = tex.lower()
    for phrase in FORBIDDEN:
        rows.append(
            {
                "scan_item": phrase,
                "status": "flag" if phrase.lower() in lower else "ok",
                "note": "Review and remove/soften this phrase." if phrase.lower() in lower else "",
            }
        )

    required = [
        ("Quality-Calibrated QCR", "main method name"),
        ("EfficientAD-30 fixed-budget", "fixed-budget baseline wording"),
        ("diagnostic", "fixed Q+C diagnostic wording"),
        ("AnomalyCLIP", "missing external VLM baseline limitation"),
        ("candidate quality", "main method mechanism"),
        ("adaptive", "adaptive refinement wording"),
    ]

    for phrase, note in required:
        rows.append(
            {
                "scan_item": phrase,
                "status": "ok" if phrase in tex else "missing",
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def write_report(before_counts: pd.DataFrame, after_counts: pd.DataFrame, patches: pd.DataFrame, safety: pd.DataFrame) -> None:
    before_total = int(before_counts["word_count"].sum())
    after_total = int(after_counts["word_count"].sum())
    delta = after_total - before_total

    flagged = safety[safety["status"].isin(["flag", "missing"])]

    lines = [
        "# Paper Stage P14: Language Polish and Venue-style Compression",
        "",
        "## 1. Summary",
        "",
        f"- input manuscript: `{IN_MAIN.relative_to(ROOT)}`",
        f"- polished copy: `{OUT_POLISHED.relative_to(ROOT)}`",
        f"- word count before: `{before_total}`",
        f"- word count after: `{after_total}`",
        f"- word count delta: `{delta:+d}`",
        "",
        "P14 generates a polished copy only. It does not overwrite `main.tex`.",
        "",
        "## 2. Patch Inventory",
        "",
        "| Patch | Type | Applied | Reason |",
        "|---|---|---:|---|",
    ]

    for _, r in patches.iterrows():
        lines.append(
            f"| {r['patch_id']} | {r['type']} | {int(bool(r['applied']))} | {r['reason']} |"
        )

    lines += [
        "",
        "## 3. Claim Safety Scan",
        "",
        "| Item | Status | Note |",
        "|---|---|---|",
    ]

    for _, r in safety.iterrows():
        lines.append(f"| {r['scan_item']} | {r['status']} | {r['note']} |")

    lines += [
        "",
        "## 4. Section Word Counts",
        "",
        "| Section | Before | After | Delta |",
        "|---|---:|---:|---:|",
    ]

    merged = before_counts.merge(after_counts, on="section", suffixes=("_before", "_after"), how="outer").fillna(0)
    for _, r in merged.iterrows():
        b = int(r["word_count_before"])
        a = int(r["word_count_after"])
        lines.append(f"| {r['section']} | {b} | {a} | {a-b:+d} |")

    lines += [
        "",
        "## 5. Decision",
        "",
    ]

    if flagged.empty:
        lines.append("P14 polished copy is claim-safe under the automatic scan.")
        next_step = "Paper Stage P15: external compile or local TeX installation check"
    else:
        lines.append("P14 found scan flags or missing required phrases. Review before using the polished copy.")
        next_step = "Patch P14 polished copy and rerun P14/P13 checks"

    lines += [
        "",
        "## 6. Next Step",
        "",
        "```text",
        f"{next_step}",
        "```",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    tex = read_text(IN_MAIN)

    before_counts = section_word_counts(tex)
    polished, patches = apply_polish(tex)
    after_counts = section_word_counts(polished)
    safety = safety_scan(polished)

    OUT_POLISHED.write_text(polished, encoding="utf-8", newline="\n")

    patches.to_csv(OUT_PATCH_INV, index=False, lineterminator="\n")
    safety.to_csv(OUT_SAFETY, index=False, lineterminator="\n")

    write_report(before_counts, after_counts, patches, safety)

    print("[DONE]", OUT_POLISHED)
    print("[DONE]", OUT_PATCH_INV)
    print("[DONE]", OUT_SAFETY)
    print("[DONE]", OUT_REPORT)
    print()
    print("===== safety flags =====")
    bad = safety[safety["status"].isin(["flag", "missing"])]
    if bad.empty:
        print("none")
    else:
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
