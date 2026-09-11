"""Boundary and anomaly-response tests for the exploratory residual probe."""
import json
import numpy as np
from crossscale_residual import fit_relation,residual,aggregate

rng=np.random.default_rng(10)
x=rng.normal(size=(100,100));y=2*x+rng.normal(scale=.1,size=x.shape)
model=fit_relation([x.ravel()],[y.ravel()])
query_x=np.zeros((100,100));normal=rng.normal(scale=.1,size=query_x.shape)
anomaly=normal.copy();anomaly[:10,:10]+=5
assert aggregate(residual(query_x,anomaly,model))>aggregate(residual(query_x,normal,model))+1
constant=fit_relation([np.zeros(100)],[np.ones(100)])
assert np.isfinite(residual(np.zeros((2,2)),np.ones((2,2)),constant)).all()
# A high score consistent with the normal scale relation should be discounted.
consistent=residual(np.ones((2,2)),np.full((2,2),2.),model)
unexpected=residual(np.zeros((2,2)),np.full((2,2),2.),model)
assert float(consistent.mean())<float(unexpected.mean())
print(json.dumps(dict(status='passed',checks=['constant calibration finite','local positive anomaly response','normal scale relation discounts expected response'],
                     note='Synthetic functional tests, not measured anomaly detection performance.')))
