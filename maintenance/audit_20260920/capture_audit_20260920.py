from pathlib import Path
import json, csv, hashlib, subprocess, datetime
r=Path("/root/private_data/iad-vlm-anomaly")
out=r/"orbitad/results/audit_20260920"
out.mkdir(parents=True,exist_ok=True)
names=["a34_l8_normal_tail_depth_fusion_v1","a47_superad_reg4_unified_eval_v1","a54_raw_preserving_residual_fusion_v1","a66_a64_paired_audit_v1","a68_normal_only_resolution_router_v1","a69_normal_calibrated_multires_fusion_v1","a70_a69_paired_audit_v1","a72_frozen_q99_mean_cable_v1"]
for name in names:
 p=r/"orbitad/results"/name
 for f in p.glob("*"):
  if f.is_file() and f.suffix in [".csv",".json",".md"] and f.stat().st_size<3000000:
   dest=out/name/f.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
meta=list(csv.DictReader((r/"orbitad/results/a10_multilayer_v1/public_meta.csv").open()))
p=r/"ad2_model_zoo/results/a73_mvtec14_dual_resolution_v1"
data={"time":datetime.datetime.now().isoformat(),"meta_columns":list(meta[0]),"n_public":len(meta),"by_category":{},"a73_markers":[f.name for f in p.glob("*COMPLETE.json")],"hashes":{}}
for c in sorted({x["category"] for x in meta}):
 rows=[x for x in meta if x["category"]==c]
 data["by_category"][c]={"n":len(rows),"labels":{k:sum(x["label"]==k for x in rows) for k in ["0","1"]},"conditions":sorted({x["condition"] for x in rows}),"normal_validation":len(list((r/"datasets/MVTec_AD_2"/c/"validation/good").glob("*.png")))}
for f in ["work/stage25/a69_normal_calibrated_multires_fusion.py","work/stage25/a70_a69_paired_audit.py","work/stage25/a72_evaluate_frozen_q99_mean_cable.py","work/stage25/run_a73_mvtec14_dual_resolution.sh","ad2_model_zoo/repos/SuperAD/src/detection.py","ad2_model_zoo/repos/SuperAD/src/sampler.py"]:
 b=(r/f).read_bytes();data["hashes"][f]=hashlib.sha256(b).hexdigest()
(out/"snapshot.json").write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
print("RECENT", [p.name for p in (r/"orbitad/results").glob("a5*")])
