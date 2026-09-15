from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()
CATS = ["fruit_jelly", "sheet_metal", "vial", "walnuts"]
RUNNER = ROOT / "experiments/stage22_selective_qcr/run_stage22_b2b_ad2_frozen_transfer.py"
SRC_ROOT = ROOT / "results/stage22_selective_qcr"
OUT = ROOT / "results/stage23_ad2_mirror/ad2_frozen_mirror"
DOC = ROOT / "docs/stage23_ad2_mirror"
FROZEN = {"w_max": 0.35, "q_quantile": 0.25, "tau_delta": 0.75}
MARGIN = 0.002
VARIANTS = {
    "D0": ("Detector only", "score_D0"),
    "M0": ("Crop VLM only", "score_M0"),
    "V3": ("Naive detector-crop fusion", "score_V3"),
    "V4": ("Old Quality-Calibrated QCR", "score_V4"),
    "V6": ("Old Adaptive QCR", "score_V6"),
    "S1": ("SRB-QCR frozen transfer", "score_S1"),
}

def locate_csv() -> Path:
    ranked = []
    for p in SRC_ROOT.rglob("*.csv"):
        try:
            h = pd.read_csv(p, nrows=0)
        except Exception:
            continue
        cols = {str(c).lower() for c in h.columns}
        score = 20 * len(cols & {"score_d0","score_v3","score_v4","score_v6","score_s1"})
        score += 20 * int("srb_pre_gate" in cols)
        score += 20 * int("category" in cols)
        score += 20 * int("ad2" in str(p).lower())
        score += 20 * int("b2b" in str(p).lower())
        if score:
            ranked.append((score, p.stat().st_size, p))
    for _, _, p in sorted(ranked, key=lambda x: (-x[0], -x[1], str(x[2]))):
        try:
            d = pd.read_csv(p, usecols=lambda c: str(c).lower() == "category")
            if set(CATS).issubset(set(d.iloc[:,0].dropna().astype(str))):
                return p
        except Exception:
            pass
    raise RuntimeError("No Stage 22-B2b AD2 prediction CSV containing all four categories was found.")

def col(df, names, role):
    m = {str(c).lower(): str(c) for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    raise RuntimeError(f"Cannot resolve {role}; columns={list(df.columns)}")

def bool_series(s):
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().all():
        return n.gt(0)
    t = s.astype(str).str.strip().str.lower()
    pos = {"1","true","yes","anomaly","abnormal","bad"}
    neg = {"0","false","no","normal","good"}
    unknown = sorted(set(t.unique()) - pos - neg)
    if unknown:
        raise RuntimeError(f"Unknown binary labels: {unknown}")
    return t.isin(pos)

def path_key(v):
    s = str(v).replace("\\","/").strip()
    for marker in ["/datasets/MVTec_AD_2_anomalib_all/","datasets/MVTec_AD_2_anomalib_all/","/datasets/","datasets/"]:
        if marker in s:
            return s.split(marker,1)[1]
    return s.removeprefix("./")

def prepare(p):
    df = pd.read_csv(p)
    cc = col(df, ["category","object","objects"], "category")
    yc = col(df, ["Y","gt_binary","gt_label","is_anomaly","is_anomaly_final"], "label")
    pc = col(df, ["image_key","path_key","canonical_image_path","image_path"], "path")
    df = df[df[cc].astype(str).isin(CATS)].copy()
    df["category"] = df[cc].astype(str)
    df["Y"] = bool_series(df[yc]).astype(int)
    df["path_key"] = df[pc].astype(str).map(path_key)
    req = ["score_D0","score_V3","score_V4","score_V6","score_S1","srb_pre_gate"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    if "score_M0" not in df.columns:
        if "M" in df.columns:
            df["score_M0"] = df["M"]
        elif "vlm_score_norm" in df.columns:
            df["score_M0"] = df["vlm_score_norm"]
        else:
            raise RuntimeError("Cannot derive M0 from score_M0, M, or vlm_score_norm.")
    nums = ["Y","score_D0","score_M0","score_V3","score_V4","score_V6","score_S1","srb_pre_gate"]
    for c in nums:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[nums].isna().any().any():
        raise RuntimeError("Missing numeric values in unified AD2 predictions.")
    if df["path_key"].duplicated().any():
        raise RuntimeError("Duplicate AD2 path_key values found.")
    if set(df["category"].unique()) != set(CATS):
        raise RuntimeError("AD2 category set is incomplete.")
    for cat, g in df.groupby("category"):
        if set(g["Y"].unique()) != {0,1}:
            raise RuntimeError(f"{cat} does not contain both normal and anomaly labels.")
    optional = [c for c in ["D","M","Q","srb_agreement","srb_weight","srb_active","has_candidate_bool","fallback_bool"] if c in df.columns]
    return df[["category","path_key","Y","score_D0","score_M0","score_V3","score_V4","score_V6","score_S1","srb_pre_gate"] + optional].sort_values(["category","path_key"]).reset_index(drop=True)

def best_f1(y, s):
    best = 0.0
    for t in np.unique(s):
        best = max(best, float(f1_score(y, (s >= t).astype(int), zero_division=0)))
    return best

def eval_scores(y, s):
    return {
        "auroc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "best_f1": best_f1(y, s),
    }

def metric_tables(df):
    per = []
    for cat, g in df.groupby("category", sort=True):
        y = g["Y"].to_numpy(int)
        for vid, (method, sc) in VARIANTS.items():
            per.append({"category":cat,"variant_id":vid,"method":method,"num_images":len(g),**eval_scores(y,g[sc].to_numpy(float))})
    per = pd.DataFrame(per)
    summary = []
    y = df["Y"].to_numpy(int)
    for vid, (method, sc) in VARIANTS.items():
        p = per[per.variant_id == vid]
        pooled = eval_scores(y, df[sc].to_numpy(float))
        summary.append({
            "variant_id":vid,"method":method,
            "macro_image_auroc":float(p.auroc.mean()),
            "macro_image_ap":float(p.ap.mean()),
            "macro_best_f1":float(p.best_f1.mean()),
            "pooled_image_auroc":pooled["auroc"],
            "pooled_image_ap":pooled["ap"],
            "pooled_best_f1":pooled["best_f1"],
            "potential_call_rate":float(df.srb_pre_gate.mean()) if vid == "S1" else np.nan,
        })
    return per, pd.DataFrame(summary)

def macro_auc(df, c):
    return float(np.mean([roc_auc_score(g.Y, g[c]) for _, g in df.groupby("category", sort=False)]))

def bootstrap(df, repeats, seed):
    pairs = [
        ("SRB-QCR vs detector","score_S1","score_D0"),
        ("SRB-QCR vs crop VLM","score_S1","score_M0"),
        ("SRB-QCR vs naive fusion","score_S1","score_V3"),
        ("Quality QCR vs naive fusion","score_V4","score_V3"),
        ("SRB-QCR vs old Quality QCR","score_S1","score_V4"),
        ("SRB-QCR vs old Adaptive QCR","score_S1","score_V6"),
    ]
    rng = np.random.default_rng(seed)
    strata = [g.index.to_numpy(int) for _, g in df.groupby(["category","Y"], sort=False)]
    vals = {n: [] for n,_,_ in pairs}
    for i in range(repeats):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        sample = df.loc[idx]
        for n,l,r in pairs:
            vals[n].append(macro_auc(sample,l)-macro_auc(sample,r))
        if i == 0 or (i+1) % 1000 == 0 or i+1 == repeats:
            print(f"[BOOTSTRAP] {i+1}/{repeats}")
    rows = []
    for n,l,r in pairs:
        v = np.asarray(vals[n], float)
        low, high = np.quantile(v,[0.025,0.975])
        pgt = float((v>0).mean())
        p2 = min(1.0, 2*min(float((v<=0).mean()),float((v>=0).mean())))
        rows.append({
            "comparison":n,
            "point_delta":macro_auc(df,l)-macro_auc(df,r),
            "ci95_low":float(low),"ci95_high":float(high),
            "probability_delta_gt_zero":pgt,
            "bootstrap_two_sided_p":p2,
            "noninferiority_margin":MARGIN,
            "probability_noninferior":float((v > -MARGIN).mean()),
            "ci_supports_noninferiority":bool(low > -MARGIN),
            "bootstrap_repeats":repeats,"seed":seed,
        })
    return pd.DataFrame(rows)

def report(df, per, summary, boot, source):
    scores = {r.variant_id:r for _,r in summary.iterrows()}
    piv = per.pivot(index="category",columns="variant_id",values="auroc")
    lines = [
        "# Stage 23-B1: AD2 Four-Category Frozen Mirror Evaluation","",
        "## Protocol","",
        "- parameters selected on: `VisA PatchCore category-LOCO`",
        "- target labels used for parameter selection: `none`",
        f"- rows: `{len(df)}`",
        f"- frozen parameters: `{FROZEN}`",
        f"- non-inferiority margin: `-{MARGIN}`","",
        "## Summary","",
        "| ID | Method | Macro AUROC | Macro AP | Macro F1 | Pooled AUROC | Call rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _,r in summary.iterrows():
        call = f"{r.potential_call_rate:.4f}" if pd.notna(r.potential_call_rate) else "-"
        lines.append(f"| {r.variant_id} | {r.method} | {r.macro_image_auroc:.4f} | {r.macro_image_ap:.4f} | {r.macro_best_f1:.4f} | {r.pooled_image_auroc:.4f} | {call} |")
    lines += ["","## Main deltas","",
        f"- SRB minus detector: `{scores['S1'].macro_image_auroc-scores['D0'].macro_image_auroc:+.4f}`",
        f"- SRB minus crop VLM: `{scores['S1'].macro_image_auroc-scores['M0'].macro_image_auroc:+.4f}`",
        f"- SRB minus naive: `{scores['S1'].macro_image_auroc-scores['V3'].macro_image_auroc:+.4f}`",
        f"- SRB minus old Quality QCR: `{scores['S1'].macro_image_auroc-scores['V4'].macro_image_auroc:+.4f}`",
        f"- SRB minus old Adaptive QCR: `{scores['S1'].macro_image_auroc-scores['V6'].macro_image_auroc:+.4f}`",
        f"- categories SRB > detector: `{int((piv.S1>piv.D0).sum())}/4`",
        f"- worst category delta vs detector: `{float((piv.S1-piv.D0).min()):+.4f}`",
        f"- potential calls saved: `{1-float(df.srb_pre_gate.mean()):.4f}`","",
        "## Bootstrap","",
        "| Comparison | Delta | CI low | CI high | P(delta>0) | p | P(non-inferior) | CI non-inferior |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _,r in boot.iterrows():
        lines.append(f"| {r.comparison} | {r.point_delta:+.4f} | {r.ci95_low:+.4f} | {r.ci95_high:+.4f} | {r.probability_delta_gt_zero:.4f} | {r.bootstrap_two_sided_p:.6f} | {r.probability_noninferior:.4f} | {bool(r.ci_supports_noninferiority)} |")
    lines += ["","## Restrictions","",
        "- AD2 has only four categories; treat statistical conclusions as supplementary.",
        "- Potential call saving is offline until Stage 23-C measured runtime.",
        f"- source: `{source.relative_to(ROOT)}`",
    ]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap_repeats",type=int,default=5000)
    ap.add_argument("--seed",type=int,default=20260726)
    ap.add_argument("--skip_stage22_rebuild",action="store_true")
    args = ap.parse_args()
    if not args.skip_stage22_rebuild:
        subprocess.run([sys.executable,str(RUNNER)],cwd=str(ROOT),check=True)
    source = locate_csv()
    print("[INPUT]", source)
    df = prepare(source)
    per, summary = metric_tables(df)
    boot = bootstrap(df,args.bootstrap_repeats,args.seed)
    OUT.mkdir(parents=True,exist_ok=True)
    DOC.mkdir(parents=True,exist_ok=True)
    protocol = {
        "protocol_id":"stage23_b1_ad2_four_category_frozen_mirror_v1",
        "target":"AD2 four categories","categories":CATS,
        "configuration":FROZEN,"uses_target_labels_for_parameters":False,
        "noninferiority_margin":MARGIN,
        "bootstrap":{"paired":True,"stratified_by":["category","Y"],"repeats":args.bootstrap_repeats,"seed":args.seed},
        "source_csv":str(source.relative_to(ROOT)),
    }
    paths = {
        "protocol":OUT/"stage23_b1_ad2_protocol.json",
        "pred":OUT/"stage23_b1_ad2_unified_predictions.csv",
        "per":OUT/"stage23_b1_ad2_per_category.csv",
        "summary":OUT/"stage23_b1_ad2_summary.csv",
        "boot":OUT/"stage23_b1_ad2_bootstrap.csv",
        "report":DOC/"stage23_b1_ad2_frozen_mirror.md",
    }
    paths["protocol"].write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding="utf-8")
    df.to_csv(paths["pred"],index=False,lineterminator="\n")
    per.to_csv(paths["per"],index=False,lineterminator="\n")
    summary.to_csv(paths["summary"],index=False,lineterminator="\n")
    boot.to_csv(paths["boot"],index=False,lineterminator="\n")
    paths["report"].write_text(report(df,per,summary,boot,source),encoding="utf-8")
    piv = per.pivot(index="category",columns="variant_id",values="auroc")
    piv["S1-D0"] = piv.S1-piv.D0
    print("\n===== SUMMARY =====\n",summary.to_string(index=False))
    print("\n===== PER-CATEGORY AUROC =====\n",piv[["D0","M0","V3","V4","V6","S1","S1-D0"]].to_string())
    print("\n===== BOOTSTRAP =====\n",boot.to_string(index=False))
    print("\n===== MAIN CLAIMS =====")
    s = {r.variant_id:r for _,r in summary.iterrows()}
    for name,right in [("detector","D0"),("crop VLM","M0"),("naive","V3"),("old quality","V4"),("old adaptive","V6")]:
        print(f"SRB - {name}: {s['S1'].macro_image_auroc-s[right].macro_image_auroc:+.6f}")
    print(f"category wins vs detector: {int((piv.S1>piv.D0).sum())}/4")
    print(f"worst category delta vs detector: {float((piv.S1-piv.D0).min()):+.6f}")
    print(f"potential call rate: {float(df.srb_pre_gate.mean()):.6f}")
    print(f"potential calls saved: {1-float(df.srb_pre_gate.mean()):.6f}")
    for p in paths.values():
        print("[DONE]",p)

if __name__ == "__main__":
    main()
