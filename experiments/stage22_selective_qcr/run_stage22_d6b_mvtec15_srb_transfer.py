from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()
STAGE = ROOT / "results/stage22_selective_qcr"
OUT = STAGE / "mvtec15_srb_qcr_transfer"
DOC = ROOT / "docs/stage22_selective_qcr"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
EXPECTED_ROWS = 1725
FROZEN = {"w_max": 0.35, "q_quantile": 0.25, "tau_delta": 0.75}
METHODS = {
    "D0": ("Detector only", "score_D0"),
    "M0": ("Crop CLIP only", "score_M0"),
    "V3": ("Naive detector-crop fusion", "score_V3"),
    "V4": ("Old Quality-Calibrated QCR", "score_V4"),
    "V6": ("Old Adaptive QCR diagnostic reconstruction", "score_V6"),
    "S1": ("SRB-QCR frozen transfer", "score_S1"),
}


def load_b1():
    path = ROOT / "experiments/stage22_selective_qcr/run_stage22_b1_visa_patchcore_loco_selection.py"
    spec = importlib.util.spec_from_file_location("stage22_b1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canon(value) -> str:
    s = str(value).replace("\\", "/").strip()
    for marker in ("/datasets/MVTecAD/", "datasets/MVTecAD/"):
        if marker in s:
            return s.split(marker, 1)[1]
    return s.removeprefix("./")


def choose(df, exact, contains=(), required=True):
    low = {str(c).lower(): str(c) for c in df.columns}
    for name in exact:
        if name.lower() in low:
            return low[name.lower()]
    for token in contains:
        hits = [str(c) for c in df.columns if token.lower() in str(c).lower()]
        if len(hits) == 1:
            return hits[0]
    if required:
        raise RuntimeError(f"Cannot choose column; exact={exact}, contains={contains}, columns={list(df.columns)}")
    return None


def binary(series):
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().all():
        return (num > 0).astype(int)
    text = series.astype(str).str.lower().str.strip()
    pos = {"1", "true", "yes", "anomaly", "abnormal", "bad", "defect", "defective"}
    neg = {"0", "false", "no", "normal", "good"}
    unknown = sorted(set(text.unique()) - pos - neg)
    if unknown:
        raise RuntimeError(f"Unknown labels: {unknown}")
    return text.isin(pos).astype(int)


def minmax_by_category(df, col):
    x = pd.to_numeric(df[col], errors="coerce")
    lo = x.groupby(df["category"]).transform("min")
    hi = x.groupby(df["category"]).transform("max")
    span = hi - lo
    return ((x - lo) / span.where(span > 1e-12)).fillna(0.0).clip(0.0, 1.0)


def files_for(category, names):
    marker = f"/mvtecad/{category}/"
    paths = []
    for name in names:
        for path in STAGE.rglob(name):
            norm = str(path).replace("\\", "/").lower()
            if marker in norm and "mvtec15" in norm and "mvtec15_srb_qcr_transfer" not in norm:
                paths.append(path)
    return sorted(set(paths))


def best_csv(category, names, role):
    choices = []
    for path in files_for(category, names):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        path_col = choose(df, ["canonical_image_path", "image_path", "path", "filename"], ["image_path"], False)
        if path_col is None:
            continue
        n = df[path_col].astype(str).map(canon).nunique()
        bonus = 10000 if role == "clip" and "mvtec15_clip_crop_reasoning" in str(path) else 0
        penalty = 10000 if role != "clip" and "clip_crop_reasoning" in str(path) else 0
        choices.append((n + bonus - penalty, len(df), path, df))
    if not choices:
        raise FileNotFoundError(f"No {role} CSV for {category}: {names}")
    choices.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return choices[0][2], choices[0][3]


def patch_table(category):
    path, df = best_csv(category, ["patchcore_image_predictions.csv", "image_predictions.csv"], "patch")
    p = choose(df, ["canonical_image_path", "image_path", "path", "filename"], ["image_path"])
    y = choose(df, ["gt_binary", "is_anomaly", "gt_label", "label", "target"], ["gt_binary", "is_anomaly"])
    d = choose(df, ["patchcore_score", "pred_score", "image_score", "anomaly_score", "score"], ["patchcore", "pred_score"])
    out = pd.DataFrame({
        "category": category,
        "image_path": df[p].astype(str),
        "path_key": df[p].astype(str).map(canon),
        "Y": binary(df[y]),
        "D_raw": pd.to_numeric(df[d], errors="coerce"),
    }).drop_duplicates("path_key")
    if out["D_raw"].isna().any():
        raise RuntimeError(f"Missing D in {path}")
    return out, path


def candidate_table(category):
    path, df = best_csv(category, ["candidate_regions.csv"], "candidate")
    p = choose(df, ["canonical_image_path", "image_path", "path", "filename"], ["image_path"])
    q = choose(df, ["candidate_quality_norm", "candidate_quality", "quality_score", "q"], ["candidate_quality"], False)
    if q is None:
        q = choose(df, ["candidate_score_mean", "mean_score", "candidate_score_max"], ["candidate_score_mean"])
        rule = f"fallback max({q}) per image"
    else:
        rule = f"direct max({q}) per image"
    r = choose(df, ["component_rank", "candidate_rank", "rank"], ["rank"], False)
    work = pd.DataFrame({
        "path_key": df[p].astype(str).map(canon),
        "Q_raw": pd.to_numeric(df[q], errors="coerce"),
    })
    valid = pd.to_numeric(df[r], errors="coerce").fillna(0).gt(0) if r else work["Q_raw"].notna()
    work["valid"] = valid
    work.loc[~valid, "Q_raw"] = np.nan
    out = work.groupby("path_key", as_index=False).agg(Q_raw=("Q_raw", "max"), num_candidates=("valid", "sum"))
    out["has_candidate_bool"] = out["num_candidates"].gt(0)
    return out, path, rule


def clip_score_column(df):
    return choose(
        df,
        [
            "crop_topk_ensemble_score", "crop_topk_mean_score", "crop_topk_score",
            "crop_vlm_score", "crop_vlm_margin", "clip_crop_score",
            "abnormal_probability", "anomaly_probability", "anomaly_score",
            "vlm_score_norm", "score",
        ],
        ["crop_topk", "crop_vlm", "clip_crop", "anomaly_score"],
    )


def clip_table(category):
    path, df = best_csv(category, ["clip_crop_predictions.csv"], "clip")
    p = choose(df, ["canonical_image_path", "image_path", "path", "filename"], ["image_path"])
    m = clip_score_column(df)
    work = pd.DataFrame({"path_key": df[p].astype(str).map(canon), "M_raw": pd.to_numeric(df[m], errors="coerce")})
    return work.groupby("path_key", as_index=False).agg(M_raw=("M_raw", "mean")), path, m


def assemble():
    frames, inventory, q_rules = [], [], set()
    for category in CATEGORIES:
        patch, p_path = patch_table(category)
        cand, c_path, q_rule = candidate_table(category)
        clip, m_path, m_col = clip_table(category)
        q_rules.add(q_rule)
        df = patch.merge(cand, on="path_key", how="left", validate="one_to_one")
        df = df.merge(clip, on="path_key", how="left", validate="one_to_one")
        df["num_candidates"] = pd.to_numeric(df["num_candidates"], errors="coerce").fillna(0).astype(int)
        df["has_candidate_bool"] = df["num_candidates"].gt(0)
        df["M_available"] = df["M_raw"].notna()
        df["fallback_bool"] = ~df["has_candidate_bool"] | ~df["M_available"]
        frames.append(df)
        inventory.append({
            "category": category, "rows": len(df),
            "patch_file": str(p_path.relative_to(ROOT)),
            "candidate_file": str(c_path.relative_to(ROOT)),
            "clip_file": str(m_path.relative_to(ROOT)),
            "clip_score_column": m_col, "q_rule": q_rule,
        })
    base = pd.concat(frames, ignore_index=True)
    if len(base) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, got {len(base)}; counts={base.groupby('category').size().to_dict()}")
    if base["path_key"].duplicated().any():
        raise RuntimeError("Duplicate path_key values.")
    if base[["Y", "D_raw", "M_raw"]].isna().any().any():
        print(base[base[["Y", "D_raw", "M_raw"]].isna().any(axis=1)][["category", "path_key", "Y", "D_raw", "M_raw"]].head(20))
        raise RuntimeError("Missing required Y/D/M values.")
    base["Q_raw"] = pd.to_numeric(base["Q_raw"], errors="coerce").fillna(0.0)
    return base, inventory, " | ".join(sorted(q_rules))


def add_scores(base, b1):
    df = base.copy()
    df["D"] = minmax_by_category(df, "D_raw")
    df["M"] = minmax_by_category(df, "M_raw")
    df["Q"] = minmax_by_category(df, "Q_raw")
    # Stage-22 input calls K a high-high detector/VLM consistency signal.
    # MVTec has no cached K, so this diagnostic reconstruction is fixed here.
    df["K"] = ((df["D"] >= 0.5) & (df["M"] >= 0.5)).astype(float)
    df = b1.add_old_scores(df)
    df["score_M0"] = df["M"]
    eligible = df["has_candidate_bool"] & df["M_available"] & ~df["fallback_bool"]
    q_threshold = float(df.loc[eligible, "Q"].quantile(FROZEN["q_quantile"]))
    df = b1.apply_srb(df, FROZEN["w_max"], q_threshold, FROZEN["tau_delta"])
    return df, q_threshold


def best_f1(y, score):
    p, r, t = precision_recall_curve(y, score)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    i = int(np.nanargmax(f))
    threshold = float(t[i]) if i < len(t) else float(np.max(score))
    return float(f[i]), threshold


def evaluate(y, score):
    f1, threshold = best_f1(y, score)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "best_f1": f1,
        "best_threshold": threshold,
    }


def metrics(pred):
    rows = []
    for category, group in pred.groupby("category", sort=True):
        y = group["Y"].to_numpy(int)
        for vid, (method, col) in METHODS.items():
            row = {
                "category": category, "variant_id": vid, "method": method,
                "num_images": len(group), "num_normal": int((y == 0).sum()),
                "num_anomaly": int((y == 1).sum()),
                **evaluate(y, group[col].to_numpy(float)),
            }
            row["potential_call_rate"] = float(group["srb_pre_gate"].mean()) if vid == "S1" else np.nan
            row["active_weight_rate"] = float(group["srb_active"].mean()) if vid == "S1" else np.nan
            rows.append(row)
    per = pd.DataFrame(rows)
    summary = []
    y = pred["Y"].to_numpy(int)
    for vid, (method, col) in METHODS.items():
        part = per[per["variant_id"] == vid]
        pooled = evaluate(y, pred[col].to_numpy(float))
        summary.append({
            "variant_id": vid, "method": method,
            "macro_image_auroc": float(part["auroc"].mean()),
            "macro_image_ap": float(part["ap"].mean()),
            "macro_best_f1": float(part["best_f1"].mean()),
            "pooled_image_auroc": pooled["auroc"], "pooled_image_ap": pooled["ap"],
            "pooled_best_f1": pooled["best_f1"], "num_categories": len(part),
            "potential_call_rate": float(pred["srb_pre_gate"].mean()) if vid == "S1" else np.nan,
            "active_weight_rate": float(pred["srb_active"].mean()) if vid == "S1" else np.nan,
        })
    return per, pd.DataFrame(summary)


def macro_auc(df, col):
    return float(np.mean([
        roc_auc_score(g["Y"], g[col])
        for _, g in df.groupby("category", sort=False)
    ]))


def bootstrap(pred, repeats=2500, seed=20260725):
    pairs = [
        ("SRB-QCR vs detector", "score_S1", "score_D0"),
        ("SRB-QCR vs naive fusion", "score_S1", "score_V3"),
        ("Quality QCR vs naive fusion", "score_V4", "score_V3"),
        ("SRB-QCR vs old Quality QCR", "score_S1", "score_V4"),
    ]
    rng = np.random.default_rng(seed)
    strata = [g.index.to_numpy(int) for _, g in pred.groupby(["category", "Y"], sort=False)]
    samples = {name: [] for name, _, _ in pairs}
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        sample = pred.loc[idx]
        for name, left, right in pairs:
            samples[name].append(macro_auc(sample, left) - macro_auc(sample, right))
    rows = []
    for name, left, right in pairs:
        values = np.asarray(samples[name])
        pgt = float((values > 0).mean())
        p2 = min(1.0, 2 * min(float((values <= 0).mean()), float((values >= 0).mean())))
        rows.append({
            "comparison": name,
            "point_delta": macro_auc(pred, left) - macro_auc(pred, right),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_delta_gt_zero": pgt,
            "bootstrap_two_sided_p": p2,
            "bootstrap_repeats": repeats,
            "seed": seed,
        })
    return pd.DataFrame(rows)


def report(pred, summary, boot, inventory, q_rule, q_threshold):
    scores = {r["variant_id"]: r for _, r in summary.iterrows()}
    per = pd.read_csv(OUT / "stage22_d6b_mvtec15_per_category.csv")
    piv = per.pivot(index="category", columns="variant_id", values="auroc")
    lines = [
        "# Stage 22-D6b: MVTec AD 15-Class Frozen SRB-QCR Transfer", "",
        "## Protocol", "",
        "- parameters selected on: `VisA PatchCore category-LOCO`",
        "- target: `MVTec AD 15 categories`",
        "- target labels used for parameters: `none`",
        f"- rows: `{len(pred)}`",
        f"- `w_max = {FROZEN['w_max']}`",
        f"- `q_quantile = {FROZEN['q_quantile']}`",
        f"- target unlabeled Q threshold: `{q_threshold:.6f}`",
        f"- `tau_delta = {FROZEN['tau_delta']}`",
        f"- resolved Q adapter: `{q_rule}`", "",
        "The Q adapter is frozen in the script before metric computation. A direct candidate-quality "
        "column is preferred; otherwise the maximum candidate mean score per image is used and "
        "normalized within category. Old Adaptive QCR is diagnostic because MVTec has no cached K; "
        "K is reconstructed by the fixed high-high rule D>=0.5 and M>=0.5.", "",
        "## Summary", "",
        "| ID | Method | Macro AUROC | Macro AP | Macro F1 | Pooled AUROC | Potential call rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        call = f"{r['potential_call_rate']:.4f}" if pd.notna(r["potential_call_rate"]) else "-"
        lines.append(
            f"| {r['variant_id']} | {r['method']} | {r['macro_image_auroc']:.4f} | "
            f"{r['macro_image_ap']:.4f} | {r['macro_best_f1']:.4f} | "
            f"{r['pooled_image_auroc']:.4f} | {call} |"
        )
    lines += [
        "", "## Main deltas", "",
        f"- SRB minus detector: `{scores['S1']['macro_image_auroc'] - scores['D0']['macro_image_auroc']:+.4f}`",
        f"- SRB minus naive: `{scores['S1']['macro_image_auroc'] - scores['V3']['macro_image_auroc']:+.4f}`",
        f"- SRB minus old Quality QCR: `{scores['S1']['macro_image_auroc'] - scores['V4']['macro_image_auroc']:+.4f}`",
        f"- categories SRB > detector: `{int((piv['S1'] > piv['D0']).sum())}/15`",
        f"- worst category delta vs detector: `{float((piv['S1'] - piv['D0']).min()):+.4f}`",
        f"- potential VLM call rate: `{float(pred['srb_pre_gate'].mean()):.4f}`",
        f"- potential calls saved: `{1 - float(pred['srb_pre_gate'].mean()):.4f}`", "",
        "## Bootstrap", "",
        "| Comparison | Delta | CI low | CI high | P(delta>0) | Two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in boot.iterrows():
        lines.append(
            f"| {r['comparison']} | {r['point_delta']:+.4f} | {r['ci95_low']:+.4f} | "
            f"{r['ci95_high']:+.4f} | {r['probability_delta_gt_zero']:.4f} | "
            f"{r['bootstrap_two_sided_p']:.6f} |"
        )
    lines += ["", "## Source inventory", ""]
    for item in inventory:
        lines += [
            f"### {item['category']}",
            f"- PatchCore: `{item['patch_file']}`",
            f"- candidates: `{item['candidate_file']}`",
            f"- CLIP: `{item['clip_file']}`",
            f"- CLIP score column: `{item['clip_score_column']}`",
            f"- Q rule: `{item['q_rule']}`", "",
        ]
    lines += [
        "Potential call saving is an offline pre-gate estimate on MVTec, not a measured wall-clock speedup."
    ]
    (DOC / "stage22_d6b_mvtec15_srb_qcr_transfer.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_repeats", type=int, default=2500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)

    base, inventory, q_rule = assemble()
    protocol = {
        "protocol_id": "stage22_d6b_mvtec15_srb_transfer_v1",
        "status": "frozen_before_mvtec15_metric_computation",
        "target": "MVTec AD 15 categories",
        "expected_rows": EXPECTED_ROWS,
        "configuration": FROZEN,
        "normalization": "within-category min-max for D, M, Q",
        "q_adapter": {
            "preference": "direct candidate-quality column",
            "fallback": "max candidate_score_mean per image",
            "uses_target_labels": False,
            "resolved_rule": q_rule,
        },
        "target_quality_threshold_rule": "25th percentile of unlabeled eligible target Q values",
        "old_adaptive_note": "diagnostic K reconstruction: I(D>=0.5 and M>=0.5)",
        "source_inventory": inventory,
    }
    protocol_path = OUT / "stage22_d6b_mvtec15_transfer_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    pred, q_threshold = add_scores(base, load_b1())
    per, summary = metrics(pred)
    boot = bootstrap(pred, repeats=args.bootstrap_repeats)

    pred.to_csv(OUT / "stage22_d6b_mvtec15_unified_predictions.csv", index=False)
    per.to_csv(OUT / "stage22_d6b_mvtec15_per_category.csv", index=False)
    summary.to_csv(OUT / "stage22_d6b_mvtec15_summary.csv", index=False)
    boot.to_csv(OUT / "stage22_d6b_mvtec15_bootstrap.csv", index=False)

    protocol.update({
        "rows": len(pred),
        "target_q_threshold": q_threshold,
        "candidate_coverage": float(pred["has_candidate_bool"].mean()),
        "clip_coverage": float(pred["M_available"].mean()),
    })
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    report(pred, summary, boot, inventory, q_rule, q_threshold)

    print("===== STAGE 22-D6b COMPLETE =====")
    print("rows:", len(pred))
    print("categories:", pred["category"].nunique())
    print("candidate coverage:", f"{pred['has_candidate_bool'].mean():.6f}")
    print("CLIP coverage:", f"{pred['M_available'].mean():.6f}")
    print("target q threshold:", f"{q_threshold:.6f}")
    print()
    print(summary[[
        "variant_id", "method", "macro_image_auroc", "macro_image_ap",
        "macro_best_f1", "potential_call_rate",
    ]].to_string(index=False))
    print()
    print(boot.to_string(index=False))
    print()
    for path in [
        protocol_path,
        OUT / "stage22_d6b_mvtec15_unified_predictions.csv",
        OUT / "stage22_d6b_mvtec15_per_category.csv",
        OUT / "stage22_d6b_mvtec15_summary.csv",
        OUT / "stage22_d6b_mvtec15_bootstrap.csv",
        DOC / "stage22_d6b_mvtec15_srb_qcr_transfer.md",
    ]:
        print("[DONE]", path)


if __name__ == "__main__":
    main()
