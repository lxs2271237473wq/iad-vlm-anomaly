from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


ROOT = Path(".").resolve()

IN_SELECTED = ROOT / "results/paper_p8/paper_p8_figure2_selected_boundary_cases.csv"

OUT_DIR = ROOT / "results/paper_p11"
DOC_DIR = ROOT / "docs/paper_p11"
FIG_DIR = DOC_DIR / "figures"
ASSET_DIR = DOC_DIR / "figure2_review_assets"

OUT_MANIFEST = OUT_DIR / "paper_p11_boundary_case_asset_manifest.csv"
OUT_REVIEW = OUT_DIR / "paper_p11_boundary_case_manual_review_sheet.csv"
OUT_REPORT = DOC_DIR / "paper_p11_boundary_case_manual_review_report.md"
OUT_PLAN = DOC_DIR / "figure2_boundary_cases_montage_plan.md"
OUT_CONTACT = FIG_DIR / "figure2_boundary_cases_contact_sheet.png"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df.columns) <= 1:
        raise RuntimeError(f"{path} read as <=1 column. Repair CSV formatting first.")
    return df


def clean_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def existing_path_from_value(value: str) -> Path | None:
    value = clean_str(value)
    if not value:
        return None

    p = Path(value)
    if p.is_absolute() and p.exists():
        return p

    p2 = ROOT / value
    if p2.exists():
        return p2.resolve()

    return None


def build_image_index() -> list[Path]:
    roots = [
        ROOT / "data",
        ROOT / "datasets",
        ROOT / "dataset",
        ROOT / "raw_data",
        ROOT / "external",
        ROOT / "results",
    ]

    files: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p.resolve())
            if len(files) > 200000:
                break

    return files


def search_image(index: list[Path], row: dict) -> Path | None:
    direct_cols = [
        "image_path",
        "original_image_path",
        "img_path",
        "path",
        "crop_path",
        "candidate_crop_path",
    ]

    for c in direct_cols:
        if c in row:
            p = existing_path_from_value(row.get(c))
            if p is not None:
                return p

    identifiers = []

    for c in ["image_path", "original_image_path", "img_path", "image_key"]:
        v = clean_str(row.get(c, ""))
        if v:
            identifiers.append(Path(v).name)
            identifiers.append(Path(v).stem)
            identifiers.append(v.replace("\\", "/").split("/")[-1])

    identifiers = [x for x in identifiers if x and len(x) >= 4]
    identifiers = list(dict.fromkeys(identifiers))

    if not identifiers:
        return None

    category = clean_str(row.get("category", "")).lower()

    scored = []
    for p in index:
        ps = str(p).replace("\\", "/").lower()
        score = 0
        for ident in identifiers:
            ident_l = ident.lower()
            if ident_l and ident_l in ps:
                score += 10
        if category and category in ps:
            score += 2
        if score > 0:
            scored.append((score, p))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return scored[0][1]


def copy_asset(src: Path | None, panel: str) -> str:
    if src is None or not src.exists():
        return ""

    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    safe_panel = clean_str(panel) or "panel"
    dst = ASSET_DIR / f"{safe_panel}_{src.name}"
    if dst.exists():
        return str(dst.relative_to(ROOT))

    dst.write_bytes(src.read_bytes())
    return str(dst.relative_to(ROOT))


def create_contact_sheet(review: pd.DataFrame) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()

    panel_w = 420
    panel_h = 300
    cols = 2
    rows = math.ceil(len(review) / cols)
    sheet = Image.new("RGB", (cols * panel_w, rows * panel_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (_, r) in enumerate(review.iterrows()):
        col = i % cols
        row = i // cols
        x0 = col * panel_w
        y0 = row * panel_h

        draw.rectangle([x0 + 8, y0 + 8, x0 + panel_w - 8, y0 + panel_h - 8], outline="black", width=2)

        title = f"Panel {r.get('figure_panel','')} | {r.get('case_type','')}"
        subtitle = f"{r.get('category','')} | GT={r.get('is_anomaly_final','')}"
        draw.text((x0 + 16, y0 + 18), title[:58], fill="black", font=font)
        draw.text((x0 + 16, y0 + 36), subtitle[:58], fill="black", font=font)

        img_path = clean_str(r.get("copied_image_path", ""))
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
                draw.text((x0 + 16, y0 + 90), "image path not found", fill="black", font=font)
        else:
            draw.text((x0 + 16, y0 + 90), "image unresolved", fill="black", font=font)

        purpose = clean_str(r.get("paper_purpose", ""))
        draw.text((x0 + 16, y0 + panel_h - 46), purpose[:62], fill="black", font=font)
        draw.text((x0 + 16, y0 + panel_h - 28), "manual_decision: pending", fill="black", font=font)

    sheet.save(OUT_CONTACT)


def build_review_pack(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = build_image_index()

    manifest_rows = []
    review_rows = []

    for _, r in selected.iterrows():
        row = r.to_dict()
        panel = clean_str(row.get("figure_panel", ""))
        resolved = search_image(index, row)
        copied = copy_asset(resolved, panel)

        asset_status = "resolved" if resolved is not None else "unresolved"

        manifest_rows.append(
            {
                "figure_panel": panel,
                "case_type": clean_str(row.get("case_type", "")),
                "category": clean_str(row.get("category", "")),
                "image_key": clean_str(row.get("image_key", "")),
                "source_image_path": str(resolved) if resolved else "",
                "copied_image_path": copied,
                "asset_status": asset_status,
                "paper_purpose": clean_str(row.get("paper_purpose", "")),
            }
        )

        review_rows.append(
            {
                "figure_panel": panel,
                "case_type": clean_str(row.get("case_type", "")),
                "category": clean_str(row.get("category", "")),
                "image_key": clean_str(row.get("image_key", "")),
                "is_anomaly_final": clean_str(row.get("is_anomaly_final", "")),
                "D": clean_str(row.get("D", "")),
                "M": clean_str(row.get("M", "")),
                "Q": clean_str(row.get("Q", "")),
                "K": clean_str(row.get("K", "")),
                "delta_quality_minus_naive": clean_str(row.get("delta_quality_minus_naive", "")),
                "delta_fixed_minus_quality": clean_str(row.get("delta_fixed_minus_quality", "")),
                "delta_adaptive_minus_quality": clean_str(row.get("delta_adaptive_minus_quality", "")),
                "paper_purpose": clean_str(row.get("paper_purpose", "")),
                "source_image_path": str(resolved) if resolved else "",
                "copied_image_path": copied,
                "asset_status": asset_status,
                "manual_decision": "pending",
                "manual_notes": "",
                "paper_use_allowed": "pending",
            }
        )

    return pd.DataFrame(manifest_rows), pd.DataFrame(review_rows)


def write_plan(review: pd.DataFrame) -> None:
    lines = [
        "# Figure 2 Boundary Case Montage Plan",
        "",
        "This is not the final figure. It is the manual inspection plan for selecting valid boundary cases.",
        "",
        "## Required panels",
        "",
        "| Panel | Case type | Category | Asset status | Paper purpose | Manual decision |",
        "|---|---|---|---|---|---|",
    ]

    for _, r in review.iterrows():
        lines.append(
            f"| {r['figure_panel']} | {r['case_type']} | {r['category']} | "
            f"{r['asset_status']} | {r['paper_purpose']} | {r['manual_decision']} |"
        )

    lines += [
        "",
        "## Manual acceptance criteria",
        "",
        "Keep a panel only if:",
        "",
        "1. the original image is visually interpretable;",
        "2. the visible defect or normal region matches the intended case type;",
        "3. the example illustrates a method boundary without implying universal behavior;",
        "4. the caption can describe it as a representative case, not proof of a general rule;",
        "5. the image does not require unsupported manufacturing-cause explanation.",
        "",
        "Reject a panel if:",
        "",
        "- the defect is not visible;",
        "- the image/crop does not match the selected case type;",
        "- the example would force an unsupported segmentation or causal claim;",
        "- the asset path cannot be resolved.",
        "",
        "## Safe caption wording",
        "",
        "```text",
        "Representative boundary cases for quality-calibrated localization-guided VLM reasoning. "
        "Quality calibration can boost true anomaly evidence and suppress normal false positives, "
        "but it can also fail when candidate quality is misleading or detector and VLM evidence disagree. "
        "These examples illustrate method boundaries rather than universal behavior.",
        "```",
        "",
    ]

    OUT_PLAN.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_report(manifest: pd.DataFrame, review: pd.DataFrame) -> None:
    unresolved = int((manifest["asset_status"] != "resolved").sum())
    total = int(len(manifest))

    lines = [
        "# Paper Stage P11: Figure 2 Manual Image Montage Preparation",
        "",
        "## 1. Summary",
        "",
        f"- selected panels: `{total}`",
        f"- unresolved image assets: `{unresolved}`",
        f"- contact sheet: `{OUT_CONTACT.relative_to(ROOT)}`" if OUT_CONTACT.exists() else "- contact sheet: not generated because Pillow is unavailable or no image assets were resolved",
        "",
        "This stage prepares manual inspection assets only. It does not finalize Figure 2.",
        "",
        "## 2. Outputs",
        "",
        f"- asset manifest: `{OUT_MANIFEST.relative_to(ROOT)}`",
        f"- manual review sheet: `{OUT_REVIEW.relative_to(ROOT)}`",
        f"- montage plan: `{OUT_PLAN.relative_to(ROOT)}`",
        f"- contact sheet: `{OUT_CONTACT.relative_to(ROOT)}`",
        "",
        "## 3. Panel review table",
        "",
        "| Panel | Case type | Category | Asset status | Manual decision |",
        "|---|---|---|---|---|",
    ]

    for _, r in review.iterrows():
        lines.append(
            f"| {r['figure_panel']} | {r['case_type']} | {r['category']} | "
            f"{r['asset_status']} | {r['manual_decision']} |"
        )

    lines += [
        "",
        "## 4. Next manual action",
        "",
        "Open the copied assets or contact sheet and edit:",
        "",
        f"`{OUT_REVIEW.relative_to(ROOT)}`",
        "",
        "Set each `manual_decision` to one of:",
        "",
        "```text",
        "keep",
        "reject",
        "replace",
        "```",
        "",
        "Only panels marked `keep` should be used for the final Figure 2 montage.",
        "",
        "## 5. Next stage",
        "",
        "```text",
        "Paper Stage P12: build final Figure 2 montage from manually approved cases",
        "```",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    selected = read_csv_strict(IN_SELECTED)

    manifest, review = build_review_pack(selected)

    manifest.to_csv(OUT_MANIFEST, index=False, lineterminator="\n")
    review.to_csv(OUT_REVIEW, index=False, lineterminator="\n")

    create_contact_sheet(review)
    write_plan(review)
    write_report(manifest, review)

    print("[DONE]", OUT_MANIFEST)
    print("[DONE]", OUT_REVIEW)
    print("[DONE]", OUT_PLAN)
    print("[DONE]", OUT_REPORT)
    if OUT_CONTACT.exists():
        print("[DONE]", OUT_CONTACT)

    print()
    print("===== asset manifest =====")
    print(manifest.to_string(index=False))
    print()
    print("===== manual review sheet =====")
    print(review[["figure_panel", "case_type", "category", "asset_status", "manual_decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
