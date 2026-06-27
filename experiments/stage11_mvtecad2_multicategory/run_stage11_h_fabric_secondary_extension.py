from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path
from typing import Any, Dict

import pandas as pd


ROOT = Path(".").resolve()

STAGE11C_SCRIPT = ROOT / "experiments/stage11_mvtecad2_multicategory/run_stage11_c_candidate_regions.py"
STAGE11D_SCRIPT = ROOT / "experiments/stage11_mvtecad2_multicategory/run_stage11_d_vlm_full_vs_context.py"

QUALITY_CSV = ROOT / "results/stage11_mvtecad2_multicategory/stage11_b1_patchcore_detector_quality_analysis.csv"

RUN_ROOT = ROOT / "runs/stage11_mvtecad2_multicategory/fabric_secondary_candidate_patchcore"
CROP_ROOT = ROOT / "results/stage11_mvtecad2_multicategory/stage11_h_fabric_candidate_crops"

OUT_DIR = ROOT / "results/stage11_mvtecad2_multicategory"
DOC_DIR = ROOT / "docs/stage11_mvtecad2_multicategory"

OUT_CAND_REGIONS = OUT_DIR / "stage11_h_fabric_candidate_regions.csv"
OUT_CAND_SUMMARY = OUT_DIR / "stage11_h_fabric_candidate_summary.csv"
OUT_CAND_STATUS = OUT_DIR / "stage11_h_fabric_candidate_status.csv"

OUT_VLM_CANDIDATES = OUT_DIR / "stage11_h_fabric_vlm_candidate_scores.csv"
OUT_VLM_IMAGES = OUT_DIR / "stage11_h_fabric_vlm_image_predictions.csv"
OUT_VLM_SUMMARY = OUT_DIR / "stage11_h_fabric_vlm_summary.csv"

OUT_EVIDENCE = OUT_DIR / "stage11_h_fabric_secondary_evidence_table.csv"
OUT_REPORT = DOC_DIR / "stage11_h_fabric_secondary_extension_report.md"


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def f4(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def get_quality_row() -> Dict[str, Any]:
    if not QUALITY_CSV.exists():
        return {}

    q = pd.read_csv(QUALITY_CSV)
    part = q[q["category"] == "fabric"]

    if part.empty:
        return {}

    return part.iloc[0].to_dict()


def run_candidate_stage() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage11c = load_module("stage11c_candidate", STAGE11C_SCRIPT)

    # Patch output roots so this secondary experiment does not overwrite Stage 11-C.
    stage11c.RUN_ROOT = RUN_ROOT
    stage11c.CROP_ROOT = CROP_ROOT

    try:
        regions_df, status = stage11c.run_one_category(
            category="fabric",
            top_k=3,
            min_area_ratio=0.0005,
        )
    except Exception as e:
        traceback.print_exc()
        status = {
            "dataset": "MVTec AD 2",
            "category": "fabric",
            "success": False,
            "fit_success": False,
            "predict_success": False,
            "num_images": 0,
            "num_candidate_rows": 0,
            "elapsed_sec": 0.0,
            "error": repr(e),
        }
        regions_df = pd.DataFrame()

    status_df = pd.DataFrame([status])
    summary_df = stage11c.build_summary(regions_df, status_df) if not regions_df.empty else pd.DataFrame()

    regions_df.to_csv(OUT_CAND_REGIONS, index=False)
    summary_df.to_csv(OUT_CAND_SUMMARY, index=False)
    status_df.to_csv(OUT_CAND_STATUS, index=False)

    if status_df.iloc[0]["success"] != True:
        raise SystemExit(f"[ERROR] fabric candidate generation failed: {status_df.iloc[0].get('error', '')}")

    return regions_df, summary_df, status_df


def run_vlm_stage() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage11d = load_module("stage11d_vlm", STAGE11D_SCRIPT)

    # Patch input so Stage 11-D scoring uses fabric secondary candidates.
    stage11d.IN_REGIONS = OUT_CAND_REGIONS

    regions = stage11d.load_regions(["fabric"])

    scorer = stage11d.ClipScorer()

    candidate_scores = stage11d.score_regions(regions, scorer)
    image_predictions = stage11d.build_image_predictions(candidate_scores, scorer)
    summary = stage11d.summarize(image_predictions)

    candidate_scores.to_csv(OUT_VLM_CANDIDATES, index=False)
    image_predictions.to_csv(OUT_VLM_IMAGES, index=False)
    summary.to_csv(OUT_VLM_SUMMARY, index=False)

    return candidate_scores, image_predictions, summary


def build_evidence(summary: pd.DataFrame, candidate_summary: pd.DataFrame) -> pd.DataFrame:
    fabric = summary[summary["category"] == "fabric"].copy()

    full = fabric[fabric["method"] == "full_image"].iloc[0]
    patchcore = fabric[fabric["method"] == "patchcore_score"].iloc[0]

    vlm = fabric[fabric["method"] != "patchcore_score"].copy()
    best_vlm = vlm.sort_values("auroc", ascending=False).iloc[0]

    context = fabric[fabric["method"].str.startswith("context")].copy()
    best_context = context.sort_values("auroc", ascending=False).iloc[0]

    tight = fabric[fabric["method"].str.startswith("tight")].copy()
    best_tight = tight.sort_values("auroc", ascending=False).iloc[0]

    q = get_quality_row()
    c = candidate_summary.iloc[0].to_dict() if not candidate_summary.empty else {}

    evidence = pd.DataFrame([{
        "dataset": "MVTec AD 2",
        "category": "fabric",
        "experiment_role": "secondary_category_extension",
        "detector_priority_group": q.get("stage11_c_priority_group", ""),
        "stage11b_image_AUROC": q.get("image_AUROC", ""),
        "stage11b_pixel_AUROC": q.get("pixel_AUROC", ""),
        "stage11b_pixel_F1": q.get("pixel_F1Score", ""),
        "candidate_coverage": c.get("candidate_coverage", ""),
        "mean_candidates_per_image": c.get("mean_candidates_per_image", ""),
        "top1_tight_gt_coverage_anomaly": c.get("top1_tight_mean_gt_coverage_anomaly", ""),
        "top1_context_gt_coverage_anomaly": c.get("top1_context_mean_gt_coverage_anomaly", ""),
        "full_image_auroc": float(full["auroc"]),
        "best_vlm_method": best_vlm["method"],
        "best_vlm_auroc": float(best_vlm["auroc"]),
        "best_vlm_delta_vs_full": float(best_vlm["auroc"]) - float(full["auroc"]),
        "best_context_method": best_context["method"],
        "best_context_auroc": float(best_context["auroc"]),
        "best_context_delta_vs_full": float(best_context["auroc"]) - float(full["auroc"]),
        "best_tight_method": best_tight["method"],
        "best_tight_auroc": float(best_tight["auroc"]),
        "best_tight_delta_vs_full": float(best_tight["auroc"]) - float(full["auroc"]),
        "patchcore_score_auroc": float(patchcore["auroc"]),
    }])

    evidence.to_csv(OUT_EVIDENCE, index=False)

    return evidence


def write_report(
    candidate_summary: pd.DataFrame,
    candidate_status: pd.DataFrame,
    vlm_summary: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    q = get_quality_row()
    e = evidence.iloc[0]

    lines = []

    lines.append("# Stage 11-H Fabric Secondary Category Extension")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage evaluates `fabric` as a secondary MVTec AD 2 category under the same Stage 11 candidate-region and VLM reasoning pipeline.")
    lines.append("")
    lines.append("It does not modify Stage 11-C/D/E primary results. All outputs are stored as `stage11_h_*` files.")
    lines.append("")
    lines.append("## 2. Why fabric is secondary")
    lines.append("")
    lines.append("Stage 11-B1 categorized `fabric` as secondary because image-level PatchCore detection is acceptable, but pixel-level localization quality is weak.")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| image AUROC | {f4(q.get('image_AUROC', ''))} |")
    lines.append(f"| image F1 | {f4(q.get('image_F1Score', ''))} |")
    lines.append(f"| pixel AUROC | {f4(q.get('pixel_AUROC', ''))} |")
    lines.append(f"| pixel F1 | {f4(q.get('pixel_F1Score', ''))} |")
    lines.append(f"| priority group | {q.get('stage11_c_priority_group', '')} |")
    lines.append("")
    lines.append("## 3. Candidate Generation Status")
    lines.append("")
    lines.append("| Category | Success | Images | Candidate rows | Time sec | Error |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for _, r in candidate_status.iterrows():
        err = "" if pd.isna(r.get("error", "")) else str(r.get("error", "")).replace("|", "/")
        lines.append(
            f"| {r['category']} | {r['success']} | {int(r['num_images'])} | "
            f"{int(r['num_candidate_rows'])} | {float(r['elapsed_sec']):.1f} | `{err}` |"
        )

    lines.append("")
    lines.append("## 4. Candidate Quality")
    lines.append("")

    if candidate_summary.empty:
        lines.append("No candidate summary was produced.")
    else:
        lines.append("| Images | Candidate rows | Coverage | Mean cand/img | Top1 tight GT coverage | Top1 context GT coverage |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        r = candidate_summary.iloc[0]
        lines.append(
            f"| {int(r['num_images'])} | {int(r['num_candidate_rows'])} | "
            f"{f4(r['candidate_coverage'])} | {f4(r['mean_candidates_per_image'])} | "
            f"{f4(r['top1_tight_mean_gt_coverage_anomaly'])} | "
            f"{f4(r['top1_context_mean_gt_coverage_anomaly'])} |"
        )

    lines.append("")
    lines.append("## 5. VLM Reasoning Results")
    lines.append("")
    lines.append("| Method | AUROC | AP | Best F1 | Best Acc | ΔAUROC vs full |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for _, r in vlm_summary[vlm_summary["category"] == "fabric"].sort_values("auroc", ascending=False).iterrows():
        lines.append(
            f"| {r['method']} | {f4(r['auroc'])} | {f4(r['ap'])} | "
            f"{f4(r['best_f1'])} | {f4(r['best_accuracy'])} | "
            f"{f4(r['delta_auroc_vs_full'])} |"
        )

    lines.append("")
    lines.append("## 6. Secondary Evidence Decision")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    lines.append(f"| full-image AUROC | {f4(e['full_image_auroc'])} |")
    lines.append(f"| best VLM method | {e['best_vlm_method']} |")
    lines.append(f"| best VLM AUROC | {f4(e['best_vlm_auroc'])} |")
    lines.append(f"| best VLM ΔAUROC vs full | {f4(e['best_vlm_delta_vs_full'])} |")
    lines.append(f"| best context method | {e['best_context_method']} |")
    lines.append(f"| best context ΔAUROC vs full | {f4(e['best_context_delta_vs_full'])} |")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    if float(e["best_context_delta_vs_full"]) > 0:
        lines.append("`fabric` provides supportive secondary evidence for context-aware crop reasoning.")
    else:
        lines.append("`fabric` should be used as a boundary/limitation case rather than main evidence.")
    lines.append("")
    lines.append("## 7. Output Files")
    lines.append("")
    lines.append(f"- Candidate regions: `{OUT_CAND_REGIONS.relative_to(ROOT)}`")
    lines.append(f"- Candidate summary: `{OUT_CAND_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Candidate status: `{OUT_CAND_STATUS.relative_to(ROOT)}`")
    lines.append(f"- VLM candidate scores: `{OUT_VLM_CANDIDATES.relative_to(ROOT)}`")
    lines.append(f"- VLM image predictions: `{OUT_VLM_IMAGES.relative_to(ROOT)}`")
    lines.append(f"- VLM summary: `{OUT_VLM_SUMMARY.relative_to(ROOT)}`")
    lines.append(f"- Evidence table: `{OUT_EVIDENCE.relative_to(ROOT)}`")
    lines.append(f"- Report: `{OUT_REPORT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 8. Next Step")
    lines.append("")
    lines.append("After this secondary extension is committed, Stage 11-I should build the final paper-ready Stage 11 evidence table.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    CROP_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    print("[INFO] Stage 11-H candidate generation for fabric")
    _, candidate_summary, candidate_status = run_candidate_stage()

    print("[INFO] Stage 11-H VLM scoring for fabric")
    _, _, vlm_summary = run_vlm_stage()

    print("[INFO] Stage 11-H evidence consolidation")
    evidence = build_evidence(vlm_summary, candidate_summary)

    write_report(candidate_summary, candidate_status, vlm_summary, evidence)

    print("[DONE]", OUT_CAND_REGIONS)
    print("[DONE]", OUT_CAND_SUMMARY)
    print("[DONE]", OUT_CAND_STATUS)
    print("[DONE]", OUT_VLM_CANDIDATES)
    print("[DONE]", OUT_VLM_IMAGES)
    print("[DONE]", OUT_VLM_SUMMARY)
    print("[DONE]", OUT_EVIDENCE)
    print("[DONE]", OUT_REPORT)

    print("\n===== evidence =====")
    print(evidence.to_string(index=False))

    print("\n===== VLM summary =====")
    print(vlm_summary.to_string(index=False))


if __name__ == "__main__":
    main()
