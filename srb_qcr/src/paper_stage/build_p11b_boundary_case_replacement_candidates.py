from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


ROOT = Path(".").resolve()

IN_CASES = ROOT / "results/stage16_qcru_ablation/stage16_e_failure_boundary_case_inventory.csv"
IN_REVIEW = ROOT / "results/paper_p11/paper_p11_boundary_case_manual_review_sheet.csv"

OUT_DIR = ROOT / "results/paper_p11b"
DOC_DIR = ROOT / "docs/paper_p11b"
FIG_DIR = DOC_DIR / "figures"
ASSET_DIR = DOC_DIR / "replacement_candidate_assets"

OUT_CANDIDATES = OUT_DIR / "paper_p11b_replacement_candidates.csv"
OUT_REPORT = DOC_DIR / "paper_p11b_replacement_candidate_report.md"
OUT_CONTACT = FIG_DIR / "paper_p11b_replacement_candidates_contact_sheet.png"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clean(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def build_image_index() -> list[Path]:
    roots = [
        ROOT / "datasets",
        ROOT / "data",
        ROOT / "dataset",
        ROOT / "raw_data",
        ROOT / "external",
        ROOT / "results",
    ]

    files = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p.resolve())
            if len(files) > 200000:
                break
    return files


def resolve_image(row: dict, index: list[Path]) -> Path | None:
    for c in ["image_path", "original_image_path", "img_path", "path", "crop_path", "candidate_crop_path", "image_key"]:
        v = clean(row.get(c, ""))
        if not v:
            continue
        p = Path(v)
        if p.is_absolute() and p.exists():
            return p
        p2 = ROOT / v
        if p2.exists():
            return p2.resolve()

    image_key = clean(row.get("image_key", ""))
    category = clean(row.get("category", "")).lower()

    ids = []
    if image_key:
        ids += [Path(image_key).name, Path(image_key).stem]

    ids = [x.lower() for x in ids if len(x) >= 4]
    if not ids:
        return None

    scored = []
    for p in index:
        ps = str(p).replace("\\", "/").lower()
        score = 0
        for ident in ids:
            if ident in ps:
                score += 10
        if category and category in ps:
            score += 2
        if score > 0:
            scored.append((score, p))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return scored[0][1]


def copy_asset(src: Path | None, tag: str) -> str:
    if src is None or not src.exists():
        return ""

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    dst = ASSET_DIR / f"{safe}_{src.name}"
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    return str(dst.relative_to(ROOT))


def select_candidates(cases: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "D", "M", "Q", "K",
        "adaptive_gate",
        "detector_vlm_disagreement",
        "delta_quality_minus_naive",
        "delta_fixed_minus_quality",
        "delta_adaptive_minus_quality",
        "is_anomaly_final",
    ]

    cases = to_num(cases, numeric_cols)

    used_image_keys = set(review["image_key"].astype(str))
    kept_image_keys = set(review.loc[review["manual_decision"] == "keep", "image_key"].astype(str))

    rows = []

    specs = [
        {
            "target_panel": "D_replacement",
            "case_type": "quality_boundary_normal_boost",
            "score_col": "delta_quality_minus_naive",
            "ascending": False,
            "want_anomaly": 0,
            "purpose": "Replacement for D: stronger normal boundary case where quality calibration boosts normal evidence.",
            "min_strong_score": 0.03,
        },
        {
            "target_panel": "E_replacement",
            "case_type": "fixed_consistency_boundary_normal_boost",
            "score_col": "delta_fixed_minus_quality",
            "ascending": False,
            "want_anomaly": 0,
            "purpose": "Replacement for E: distinct fixed-consistency normal boost case, avoiding duplicate with B.",
            "min_strong_score": 0.05,
        },
        {
            "target_panel": "E_alt_replacement",
            "case_type": "fixed_consistency_boundary_anomaly_suppression",
            "score_col": "delta_fixed_minus_quality",
            "ascending": True,
            "want_anomaly": 1,
            "purpose": "Alternative E: fixed consistency suppresses true anomaly evidence.",
            "min_strong_score": None,
        },
    ]

    for spec in specs:
        sub = cases[cases["case_type"] == spec["case_type"]].copy()

        if "is_anomaly_final" in sub.columns:
            sub = sub[sub["is_anomaly_final"] == spec["want_anomaly"]]

        if "image_key" in sub.columns:
            # Avoid all existing selected panels first.
            sub = sub[~sub["image_key"].astype(str).isin(used_image_keys)]

        if sub.empty:
            # Relax once: only avoid kept A/B/C/F, allow replacing D/E source if needed.
            sub = cases[cases["case_type"] == spec["case_type"]].copy()
            if "is_anomaly_final" in sub.columns:
                sub = sub[sub["is_anomaly_final"] == spec["want_anomaly"]]
            if "image_key" in sub.columns:
                sub = sub[~sub["image_key"].astype(str).isin(kept_image_keys)]

        if spec["score_col"] in sub.columns:
            sub = sub.sort_values(spec["score_col"], ascending=spec["ascending"])

        sub = sub.head(5).copy()

        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            score = r.get(spec["score_col"], None)

            if spec["min_strong_score"] is None or pd.isna(score):
                strength = "inspect"
            else:
                if spec["ascending"]:
                    strength = "strong" if score <= -spec["min_strong_score"] else "weak"
                else:
                    strength = "strong" if score >= spec["min_strong_score"] else "weak"

            out = {
                "target_panel": spec["target_panel"],
                "candidate_rank": rank,
                "case_type": spec["case_type"],
                "selection_score_col": spec["score_col"],
                "selection_score": score,
                "candidate_strength": strength,
                "paper_purpose": spec["purpose"],
            }

            for c in [
                "backbone", "dataset", "strategy", "eval_mode", "category",
                "image_key", "is_anomaly_final", "D", "M", "Q", "K",
                "delta_quality_minus_naive", "delta_fixed_minus_quality",
                "delta_adaptive_minus_quality", "detector_vlm_disagreement",
            ]:
                if c in r:
                    out[c] = r[c]

            rows.append(out)

    return pd.DataFrame(rows)


def create_contact_sheet(candidates: pd.DataFrame) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    panel_w = 460
    panel_h = 310
    cols = 2
    rows = math.ceil(len(candidates) / cols)
    sheet = Image.new("RGB", (cols * panel_w, max(1, rows) * panel_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (_, r) in enumerate(candidates.iterrows()):
        col = i % cols
        row = i // cols
        x0 = col * panel_w
        y0 = row * panel_h

        draw.rectangle([x0 + 8, y0 + 8, x0 + panel_w - 8, y0 + panel_h - 8], outline="black", width=2)

        title = f"{r['target_panel']} #{r['candidate_rank']} | {r['case_type']}"
        score = f"{r['selection_score_col']}={r['selection_score']:.4f}" if pd.notna(r.get("selection_score")) else ""
        subtitle = f"{r.get('category','')} | GT={r.get('is_anomaly_final','')} | {score}"

        draw.text((x0 + 16, y0 + 18), title[:70], fill="black", font=font)
        draw.text((x0 + 16, y0 + 36), subtitle[:70], fill="black", font=font)

        img_path = clean(r.get("copied_image_path", ""))
        if img_path:
            p = ROOT / img_path
            if p.exists():
                try:
                    img = Image.open(p).convert("RGB")
                    img.thumbnail((panel_w - 40, panel_h - 105))
                    ix = x0 + (panel_w - img.width) // 2
                    iy = y0 + 62
                    sheet.paste(img, (ix, iy))
                except Exception as e:
                    draw.text((x0 + 16, y0 + 90), f"image open failed: {e}", fill="black", font=font)
            else:
                draw.text((x0 + 16, y0 + 90), "image not found", fill="black", font=font)
        else:
            draw.text((x0 + 16, y0 + 90), "image unresolved", fill="black", font=font)

        draw.text((x0 + 16, y0 + panel_h - 28), f"strength={r.get('candidate_strength','')}", fill="black", font=font)

    sheet.save(OUT_CONTACT)


def write_report(candidates: pd.DataFrame) -> None:
    lines = []
    lines += [
        "# Paper Stage P11-B: Boundary Case Replacement Candidates",
        "",
        "## 1. Purpose",
        "",
        "Panels D and E from P11 were marked as `replace`. This stage proposes replacement candidates.",
        "",
        "Rules:",
        "",
        "- D replacement should be a stronger `quality_boundary_normal_boost` case.",
        "- E replacement should be a distinct fixed-consistency boundary case, not duplicating panel B.",
        "- If D remains weak, it is acceptable to drop D and use a 4-panel or 5-panel Figure 2.",
        "",
        "## 2. Candidate list",
        "",
        "| Target | Rank | Case type | Category | Image key | Score col | Score | Strength |",
        "|---|---:|---|---|---|---|---:|---|",
    ]

    for _, r in candidates.iterrows():
        score = "" if pd.isna(r.get("selection_score")) else f"{float(r['selection_score']):.4f}"
        lines.append(
            f"| {r['target_panel']} | {r['candidate_rank']} | {r['case_type']} | "
            f"{r.get('category','')} | {r.get('image_key','')} | "
            f"{r.get('selection_score_col','')} | {score} | {r.get('candidate_strength','')} |"
        )

    lines += [
        "",
        "## 3. Contact sheet",
        "",
        f"- `{OUT_CONTACT.relative_to(ROOT)}`",
        "",
        "## 4. Manual decision required",
        "",
        "Open the contact sheet and choose:",
        "",
        "```text",
        "D replacement: one candidate, or drop panel D",
        "E replacement: one candidate, preferably not visually/numerically duplicate with B",
        "```",
        "",
        "After choosing, update the final Figure 2 panel list in P12.",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    cases = read_csv(IN_CASES)
    review = read_csv(IN_REVIEW)

    candidates = select_candidates(cases, review)

    index = build_image_index()
    copied_paths = []
    source_paths = []

    for _, r in candidates.iterrows():
        src = resolve_image(r.to_dict(), index)
        tag = f"{r['target_panel']}_{r['candidate_rank']}"
        copied = copy_asset(src, tag)
        source_paths.append(str(src) if src else "")
        copied_paths.append(copied)

    candidates["source_image_path"] = source_paths
    candidates["copied_image_path"] = copied_paths
    candidates["asset_status"] = ["resolved" if x else "unresolved" for x in copied_paths]

    candidates.to_csv(OUT_CANDIDATES, index=False, lineterminator="\n")

    create_contact_sheet(candidates)
    write_report(candidates)

    print("[DONE]", OUT_CANDIDATES)
    print("[DONE]", OUT_REPORT)
    if OUT_CONTACT.exists():
        print("[DONE]", OUT_CONTACT)
    print()
    print(candidates[[
        "target_panel", "candidate_rank", "case_type", "category",
        "selection_score_col", "selection_score", "candidate_strength",
        "asset_status"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
