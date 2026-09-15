from pathlib import Path
import json,hashlib,pandas as pd
R=Path('/root/private_data/iad-vlm-anomaly');O=R/'model_comparison/efficientad_export'
p=pd.read_csv(O/'predictions.csv');p['legacy_folder']=p.legacy_test;p['legacy_test']=False
audit=[]
for c in json.loads((O/'checkpoints.json').read_text()):
 cat=c['category'];ck=Path(c['checkpoint']);visuals=list((ck.parents[2]/'images').glob('*/*.png'))
 assert visuals,cat
 source=R/'datasets/MVTec_AD_2_anomalib_all'/f'{cat}_folder'/'test';ids=[]
 for v in visuals:
  original=source/v.parent.name/v.name;assert original.exists(),str(original)
  sha=hashlib.sha256(original.read_bytes()).hexdigest();hit=(p.category==cat)&(p.sha256==sha)
  assert hit.sum()==1,(cat,str(v));p.loc[hit,'legacy_test']=True;ids.append(sha)
 assert len(ids)==len(set(ids))
 audit.append(dict(category=cat,n=len(ids),source='Historical test visualization filenames -> original Folder test files -> SHA256 -> canonical AD2 public',visual_root=str(ck.parents[2]/'images')))
p.to_csv(O/'predictions.csv',index=False);(O/'subset_audit.json').write_text(json.dumps(audit,indent=2));print(audit)
