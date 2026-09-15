"""Posthoc paired check versus already-existing strong source controls."""
from pathlib import Path
import pandas as pd
from evaluate_exchange import intervals
root=Path('/root/private_data/iad-vlm-anomaly/results/stage24_evidence')
new=pd.read_csv(root/'20260911_i/oof_predictions.csv')
old=pd.read_csv(root/'20260911_d/evaluation/oof_predictions.csv')
f=new.merge(old[['image_id','geometry_unconditional','geometry512']],on='image_id',validate='one_to_one');assert len(f)==2162
ci=pd.DataFrame(intervals(f,[('linear_contrast','geometry_unconditional'),('linear_contrast','geometry512')],replicates=2000))
ci.to_csv(root/'20260911_i/strong_control_intervals.csv',index=False);print(ci.to_string(index=False))
