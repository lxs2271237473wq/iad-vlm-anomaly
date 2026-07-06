from __future__ import annotations

from pathlib import Path
import math
import shutil
import pandas as pd


ROOT = Path(".").resolve()

IN_P11_REVIEW = ROOT / "results/paper_p11/paper_p11_boundary_case_manual_review_sheet.csv"
IN_P11B_DECISIONS = ROOT / "results/paper_p11b/paper_p11b_manual_replacement_decisions.csv"

OUT_DIR = ROOT / "results/paper_p12"
DOC_DIR = ROOT / "docs/paper_p12"
FIG_DIR = DOC_DIR / "figures"
ASSET_DIR = DOC_DIR / "figure2_final_assets"

PAPER_DIR = ROOT / "paper/quality_calibrated_qcr"
PAPER_FIG_DIR = PAPER_DIR / "figures"
MAIN_TEX = PAPER_DIR / "main.tex"

OUT_MANIFEST = OUT_DIR / "paper_p12_final_figure2_panel_manifest.csv"
OUT_REPORT = DOC_DIR / "paper_p12_final_figure2_report.md"
OUT_LATEX_SNIPPET = DOC_DIR / "figure2_latex_snippet.tex"

OUT_FIG = FIG_DIR / "figure2_boundary_cases_montage.png"
OUT_FIG_PAPER = PAPER_FIG_DIR / "figure2_boundary_cases_montage.png"


FINAL_ORDER = ["A", "B", "C", "E", "F"]

PANEL_TITLES = {
    "A": "Quality boosts anomaly",
    "B": "Quality suppresses normal",
    "C": "Quality boundary",
    "E": "Fixed Q+C risk",
    "F": "Detector-VLM disagreement",
}

PANEL_SHORT_PURPOSE = {
    "A": "True anomaly with reliable candidate quality.",
    "B": "Normal image with high VLM evidence suppressed by low quality.",
    "C": "True anomaly suppressed when quality is misleading.",
    "E": "Fixed consistency boosts a normal case.",
    "F": "Detector high, VLM low: evidence conflict.",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def as_float(x, default=None):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def clean(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def resolve_path(value: str) -> Path:
    value = clean(value)
    if not value:
        raise FileNotFoundError("empty copied_image_path")

    p = Path(value)
    if p.is_absolute() and p.exists():
        return p

    p2 = ROOT / value
    if p2.exists():
        return p2.resolve()

    raise FileNotFoundError(value)


def collect_final_panels() -> pd.DataFrame:
    review = read_csv(IN_P11_REVIEW)
    decisions = read_csv(IN_P11B_DECISIONS)

    rows = []

    for panel in ["A", "B", "C", "F"]:
        sub = review[
            (review["figure_panel"].astype(str) == panel)
            & (review["manual_decision"].astype(str) == "keep")
            & (review["paper_use_allowed"].astype(str) == "yes")
        ].copy()

        if sub.empty:
            raise RuntimeError(f"Missing approved P11 panel {panel}")

        r = sub.iloc[0].to_dict()
        rows.append(
            {
                "final_panel": panel,
                "source_stage": "P11",
                "case_type": r.get("case_type", ""),
                "category": r.get("category", ""),
                "image_key": r.get("image_key", ""),
                "is_anomaly_final": r.get("is_anomaly_final", ""),
                "D": r.get("D", ""),
                "M": r.get("M", ""),
                "Q": r.get("Q", ""),
                "K": r.get("K", ""),
                "delta_quality_minus_naive": r.get("delta_quality_minus_naive", ""),
                "delta_fixed_minus_quality": r.get("delta_fixed_minus_quality", ""),
                "delta_adaptive_minus_quality": r.get("delta_adaptive_minus_quality", ""),
                "detector_vlm_disagreement": r.get("detector_vlm_disagreement", ""),
                "paper_purpose": r.get("paper_purpose", PANEL_SHORT_PURPOSE[panel]),
                "copied_image_path": r.get("copied_image_path", ""),
                "manual_notes": r.get("manual_notes", ""),
            }
        )

    sub_e = decisions[
        (decisions["manual_decision"].astype(str) == "keep")
        & (decisions["final_panel"].astype(str) == "E")
        & (decisions["paper_use_allowed"].astype(str) == "yes")
    ].copy()

    if sub_e.empty:
        raise RuntimeError("Missing approved P11-B replacement panel E")

    r = sub_e.iloc[0].to_dict()
    rows.append(
        {
            "final_panel": "E",
            "source_stage": "P11-B",
            "case_type": r.get("case_type", ""),
            "category": r.get("category", ""),
            "image_key": r.get("image_key", ""),
            "is_anomaly_final": r.get("is_anomaly_final", ""),
            "D": r.get("D", ""),
            "M": r.get("M", ""),
            "Q": r.get("Q", ""),
            "K": r.get("K", ""),
            "delta_quality_minus_naive": r.get("delta_quality_minus_naive", ""),
            "delta_fixed_minus_quality": r.get("delta_fixed_minus_quality", ""),
            "delta_adaptive_minus_quality": r.get("delta_adaptive_minus_quality", ""),
            "detector_vlm_disagreement": r.get("detector_vlm_disagreement", ""),
            "paper_purpose": r.get("paper_purpose", PANEL_SHORT_PURPOSE["E"]),
            "copied_image_path": r.get("copied_image_path", ""),
            "manual_notes": r.get("manual_notes", ""),
        }
    )

    df = pd.DataFrame(rows)
    df["panel_order"] = df["final_panel"].map({p: i for i, p in enumerate(FINAL_ORDER)})
    df = df.sort_values("panel_order").drop(columns=["panel_order"])

    return df


def copy_final_assets(df: pd.DataFrame) -> pd.DataFrame:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    final_paths = []
    source_abs_paths = []

    for _, r in out.iterrows():
        src = resolve_path(r["copied_image_path"])
        panel = r["final_panel"]
        dst = ASSET_DIR / f"panel_{panel}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)

        final_paths.append(str(dst.relative_to(ROOT)))
        source_abs_paths.append(str(src))

    out["source_abs_path"] = source_abs_paths
    out["final_asset_path"] = final_paths
    return out


def draw_multiline(draw, xy, text, font, fill, max_chars=42, line_gap=4):
    x, y = xy
    words = str(text).split()
    lines = []
    cur = ""

    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w

    if cur:
        lines.append(cur)

    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap

    return y


def make_montage(df: pd.DataFrame) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception as e:
        raise RuntimeError("Pillow is required to create Figure 2 montage.") from e

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 3 panels top row, 2 panels bottom row. Good compromise for 5 panels.
    cols = 3
    panel_w = 520
    panel_h = 430
    rows = 2
    margin = 24

    canvas_w = cols * panel_w
    canvas_h = rows * panel_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.load_default()
    font_bold = ImageFont.load_default()

    positions = {
        "A": (0, 0),
        "B": (1, 0),
        "C": (2, 0),
        "E": (0, 1),
        "F": (1, 1),
    }

    for _, r in df.iterrows():
        panel = r["final_panel"]
        col, row = positions[panel]
        x0 = col * panel_w
        y0 = row * panel_h

        # Border
        draw.rectangle(
            [x0 + 8, y0 + 8, x0 + panel_w - 8, y0 + panel_h - 8],
            outline="black",
            width=2,
        )

        title = f"({panel}) {PANEL_TITLES.get(panel, '')}"
        subtitle = f"{r.get('category','')} | GT={r.get('is_anomaly_final','')} | {r.get('case_type','')}"

        draw.text((x0 + margin, y0 + 18), title, fill="black", font=font_bold)
        draw.text((x0 + margin, y0 + 38), subtitle[:72], fill="black", font=font)

        # Image area
        asset = resolve_path(r["final_asset_path"])
        img = Image.open(asset).convert("RGB")

        target_w = panel_w - 2 * margin
        target_h = 250
        img_box = ImageOps.contain(img, (target_w, target_h))
        ix = x0 + (panel_w - img_box.width) // 2
        iy = y0 + 70
        canvas.paste(img_box, (ix, iy))

        # Score line
        D = as_float(r.get("D"))
        M = as_float(r.get("M"))
        Q = as_float(r.get("Q"))
        K = as_float(r.get("K"))
        dq = as_float(r.get("delta_quality_minus_naive"))
        dfixed = as_float(r.get("delta_fixed_minus_quality"))
        disagreement = as_float(r.get("detector_vlm_disagreement"))

        score_bits = []
        if D is not None:
            score_bits.append(f"D={D:.2f}")
        if M is not None:
            score_bits.append(f"M={M:.2f}")
        if Q is not None:
            score_bits.append(f"Q={Q:.2f}")
        if K is not None:
            score_bits.append(f"K={K:.2f}")

        y_text = y0 + 332
        draw.text((x0 + margin, y_text), " | ".join(score_bits), fill="black", font=font)
        y_text += 20

        delta_bits = []
        if dq is not None:
            delta_bits.append(f"Δquality={dq:+.2f}")
        if dfixed is not None:
            delta_bits.append(f"Δfixed={dfixed:+.2f}")
        if disagreement is not None:
            delta_bits.append(f"disagree={disagreement:.2f}")

        draw.text((x0 + margin, y_text), " | ".join(delta_bits)[:75], fill="black", font=font)
        y_text += 22

        purpose = PANEL_SHORT_PURPOSE.get(panel, r.get("paper_purpose", ""))
        draw_multiline(draw, (x0 + margin, y_text), purpose, font, "black", max_chars=58)

    # Empty sixth cell.
    x0 = 2 * panel_w
    y0 = 1 * panel_h
    draw.rectangle(
        [x0 + 8, y0 + 8, x0 + panel_w - 8, y0 + panel_h - 8],
        outline="#bbbbbb",
        width=1,
    )
    draw.text(
        (x0 + margin, y0 + 30),
        "Panel D dropped",
        fill="black",
        font=font_bold,
    )
    draw_multiline(
        draw,
        (x0 + margin, y0 + 58),
        "No replacement was used because all D candidates were weak and would not support a reliable boundary example.",
        font,
        "black",
        max_chars=55,
    )

    canvas.save(OUT_FIG)
    shutil.copy2(OUT_FIG, OUT_FIG_PAPER)


def write_latex_snippet() -> None:
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/figure2_boundary_cases_montage.png}
\caption{Representative boundary cases for quality-calibrated localization-guided VLM reasoning. Quality calibration can boost true anomaly evidence and suppress normal false positives, but it can also fail when candidate quality is misleading. Fixed Q+C can over-boost normal evidence, and detector-VLM disagreement remains a boundary case. These examples illustrate method boundaries rather than universal behavior.}
\label{fig:boundary_cases}
\end{figure}
"""
    OUT_LATEX_SNIPPET.write_text(text, encoding="utf-8", newline="\n")


def update_main_tex() -> None:
    if not MAIN_TEX.exists():
        return

    text = MAIN_TEX.read_text(encoding="utf-8")

    if r"\label{fig:boundary_cases}" in text:
        return

    snippet = OUT_LATEX_SNIPPET.read_text(encoding="utf-8")

    marker = r"\input{tables/table3_boundary_summary}"
    if marker not in text:
        # Do not risk corrupting main.tex if marker changed.
        return

    text = text.replace(marker, snippet + "\n" + marker, 1)
    MAIN_TEX.write_text(text, encoding="utf-8", newline="\n")


def write_report(df: pd.DataFrame) -> None:
    keep_count = len(df)

    lines = [
        "# Paper Stage P12: Final Figure 2 Montage",
        "",
        "## 1. Summary",
        "",
        f"- final panels: `{keep_count}`",
        "- panel order: `A, B, C, E, F`",
        "- dropped panel: `D`",
        f"- montage: `{OUT_FIG.relative_to(ROOT)}`",
        f"- paper copy: `{OUT_FIG_PAPER.relative_to(ROOT)}`",
        f"- LaTeX snippet: `{OUT_LATEX_SNIPPET.relative_to(ROOT)}`",
        "",
        "## 2. Final panel manifest",
        "",
        "| Panel | Source | Case type | Category | GT | Purpose | Asset |",
        "|---|---|---|---|---:|---|---|",
    ]

    for _, r in df.iterrows():
        lines.append(
            f"| {r['final_panel']} | {r['source_stage']} | {r['case_type']} | "
            f"{r['category']} | {r['is_anomaly_final']} | "
            f"{PANEL_SHORT_PURPOSE.get(r['final_panel'], r.get('paper_purpose',''))} | "
            f"`{r['final_asset_path']}` |"
        )

    lines += [
        "",
        "## 3. Caption",
        "",
        "```text",
        "Representative boundary cases for quality-calibrated localization-guided VLM reasoning. Quality calibration can boost true anomaly evidence and suppress normal false positives, but it can also fail when candidate quality is misleading. Fixed Q+C can over-boost normal evidence, and detector-VLM disagreement remains a boundary case. These examples illustrate method boundaries rather than universal behavior.",
        "```",
        "",
        "## 4. Notes",
        "",
        "- D is intentionally dropped because replacement candidates were weak.",
        "- E uses the P11-B selected replacement: fixed-consistency normal boost, pcb3, FastFlow.",
        "- The figure is boundary analysis, not proof of universal behavior.",
        "- The figure must not be described as segmentation output or manufacturing-cause explanation.",
        "",
        "## 5. Next step",
        "",
        "```text",
        "Paper Stage P13: compile-check LaTeX manuscript and patch figure/table issues",
        "```",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    panels = collect_final_panels()
    panels = copy_final_assets(panels)

    panels.to_csv(OUT_MANIFEST, index=False, lineterminator="\n")

    make_montage(panels)
    write_latex_snippet()
    update_main_tex()
    write_report(panels)

    print("[DONE]", OUT_MANIFEST)
    print("[DONE]", OUT_FIG)
    print("[DONE]", OUT_FIG_PAPER)
    print("[DONE]", OUT_LATEX_SNIPPET)
    print("[DONE]", OUT_REPORT)
    if MAIN_TEX.exists():
        print("[DONE] checked/updated", MAIN_TEX)

    print()
    print("===== final panels =====")
    print(panels[[
        "final_panel", "source_stage", "case_type", "category",
        "is_anomaly_final", "final_asset_path"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
