import json
import numpy as np
from audit_normal_shift import fit_reference,transform

rng=np.random.default_rng(42);fit=rng.normal(size=(2000,7)).astype(np.float32)
reference=fit_reference(fit);query=rng.normal(size=(1000,7)).astype(np.float32)
a,b=transform(query,reference)
assert np.isfinite(a).all() and np.isfinite(b).all() and ((b>=0)&(b<=1)).all()
for j in range(7):
    order=np.argsort(query[:,j]);assert (np.diff(b[order,j])>=-1e-6).all()
c,d=transform(np.ones((10,7),np.float32),fit_reference(np.ones((20,7),np.float32)))
assert np.isfinite(c).all() and np.isfinite(d).all()
print(json.dumps(dict(status='passed',checks=['finite transformed features','percentile bounds','monotone mapping','constant feature handling'],detection_performance_claim=False)))
