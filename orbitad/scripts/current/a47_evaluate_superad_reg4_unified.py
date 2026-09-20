"""Evaluate A45 maps with the exact native category-condition AU-PRO protocol."""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile


ROOT = Path("/root/private_data/iad-vlm-anomaly")
META = ROOT / "orbitad/results/a10_multilayer_v1/public_meta.csv"
METHOD = os.environ.get("UNIFIED_METHOD", "superad_reg4")
LAYOUT = os.environ.get("UNIFIED_LAYOUT", "superad")
MAP_ROOT = Path(os.environ.get("UNIFIED_MAP_ROOT", ROOT / "ad2_model_zoo/results/a45_superad_reg4_public_v1"))
FINAL_DETAIL = ROOT / "orbitad/results/a34_l8_normal_tail_depth_fusion_v1/native_category_condition_size.csv"
OUT = Path(os.environ.get("UNIFIED_OUT", ROOT / "orbitad/results/a47_superad_reg4_unified_eval_v1"))
MAX_FPR, BINS = .05, 65536
BANDS = ("all", "tiny_le_0.1pct", "small_0.1_to_1pct", "large_gt_1pct")
CATEGORIES = tuple(x.strip() for x in os.environ.get("UNIFIED_CATEGORIES", "").split(",") if x.strip())


def map_path(row):
    rel = Path(row["image_path"])
    anomaly_type, stem = rel.parent.name, rel.stem
    if LAYOUT == "flat_category_label":
        return MAP_ROOT / row["category"] / anomaly_type / f"{stem}.tiff"
    return MAP_ROOT / row["category"] / "anomaly_maps/seed=0" / row["category"] / "test" / anomaly_type / f"{stem}.tiff"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if LAYOUT == "superad":
        marker = os.environ.get("UNIFIED_COMPLETION_MARKER", "A45_INFERENCE_COMPLETE.json")
        if marker:
            assert (MAP_ROOT / marker).exists(), MAP_ROOT / marker
    else:
        assert (MAP_ROOT / "MAPS_COMPLETE.json").exists()
    rows = list(csv.DictReader(open(META)))
    if CATEGORIES:
        rows = [r for r in rows if r["category"] in CATEGORIES]
    missing = [str(map_path(r)) for r in rows if not map_path(r).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} maps; first={missing[:3]}")
    maximum = max(float(np.nanmax(tifffile.imread(map_path(r)))) for r in rows)
    edges = np.linspace(0, max(maximum*1.001, maximum+1e-6), BINS+1)

    class Stats:
        def __init__(self):
            self.neg=np.zeros(BINS,np.int64); self.region=np.zeros(BINS,np.float64); self.regions=0
    stats={}
    def get(cat, condition, band):
        key=cat,condition,band
        if key not in stats: stats[key]=Stats()
        return stats[key]
    for n,row in enumerate(rows,1):
        score=np.asarray(tifffile.imread(map_path(row)),np.float32)
        if int(row["label"])==0:
            mask=np.zeros(score.shape,np.uint8)
        else:
            mask=(cv2.imread(str(ROOT/"datasets/MVTec_AD_2"/row["mask_path"]),cv2.IMREAD_GRAYSCALE)>0).astype(np.uint8)
        if score.shape != mask.shape:
            score=cv2.resize(score,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_LINEAR)
        neg_hist=np.histogram(score[mask==0],bins=edges)[0]
        for band in BANDS: get(row["category"],row["condition"],band).neg += neg_hist
        count, labels=cv2.connectedComponents(mask,8)
        for rid in range(1,count):
            vals=score[labels==rid]; ratio=len(vals)/mask.size
            band="tiny_le_0.1pct" if ratio<=.001 else ("small_0.1_to_1pct" if ratio<=.01 else "large_gt_1pct")
            contribution=np.histogram(vals,bins=edges)[0]/len(vals)
            for target in ("all",band):
                s=get(row["category"],row["condition"],target); s.region += contribution; s.regions += 1
        if n%50==0: print("evaluated",n,len(rows),flush=True)

    def aupro(s):
        if s.neg.sum()==0 or s.regions==0: return float("nan")
        fp=np.r_[0.,np.cumsum(s.neg[::-1],dtype=np.float64)]/s.neg.sum()
        pro=np.r_[0.,np.cumsum(s.region[::-1],dtype=np.float64)/s.regions]
        keep=fp<=MAX_FPR; x,y=fp[keep],pro[keep]
        if x[-1]<MAX_FPR:
            i=np.searchsorted(fp,MAX_FPR,side="right")
            if i<len(fp):
                w=(MAX_FPR-fp[i-1])/max(fp[i]-fp[i-1],1e-12)
                x=np.r_[x,MAX_FPR]; y=np.r_[y,pro[i-1]+w*(pro[i]-pro[i-1])]
        return float(np.trapezoid(y,x)/MAX_FPR)

    detail=[]
    for (cat,condition,band),s in sorted(stats.items()):
        detail.append({"method":METHOD,"category":cat,"condition":condition,"size_band":band,"regions":s.regions,"aupro_0_05":aupro(s)})
    with (OUT/"native_category_condition_size.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0])); w.writeheader(); w.writerows(detail)
    summary={}
    for band in BANDS:
        selected=[r for r in detail if r["size_band"]==band and np.isfinite(r["aupro_0_05"])]
        grouped=defaultdict(list)
        for r in selected: grouped[r["category"]].append(r["aupro_0_05"])
        summary[band]={"category_macro":float(np.mean([np.mean(v) for v in grouped.values()])),"categories":len(grouped),"regions":sum(r["regions"] for r in selected)}

    final_rows=list(csv.DictReader(open(FINAL_DETAIL)))
    final_by=defaultdict(list); ext_by=defaultdict(list)
    for r in final_rows:
        if r["method"]=="a16_l8_equal" and r["size_band"]=="all": final_by[r["category"]].append(float(r["aupro_0_05"]))
    for r in detail:
        if r["size_band"]=="all": ext_by[r["category"]].append(r["aupro_0_05"])
    common_categories = sorted(set(final_by) & set(ext_by))
    deltas={cat:float(np.mean(final_by[cat])-np.mean(ext_by[cat])) for cat in common_categories}
    rng=np.random.default_rng(20260917); values=np.array(list(deltas.values()))
    boot=np.mean(rng.choice(values,(100000,len(values)),replace=True),axis=1)
    comparison={f"final_minus_{METHOD}_by_category":deltas,"mean_delta":float(values.mean()),"wins":int((values>0).sum()),"bootstrap_95ci":[float(x) for x in np.quantile(boot,[.025,.975])]}
    payload={"summary":summary,"comparison":comparison,"protocol":"same evaluator as A28/A34; native category-condition macro AU-PRO@FPR<=0.05"}
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2))
    (OUT/"A47_EVALUATION_COMPLETE.json").write_text(json.dumps({"status":"complete"},indent=2))
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__": main()
