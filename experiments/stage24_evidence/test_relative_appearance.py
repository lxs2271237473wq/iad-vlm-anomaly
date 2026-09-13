import json
import numpy as np
from audit_normal_shift import fit_reference,transform
from relative_appearance_controls import relative
rng=np.random.default_rng(42)
x=rng.normal(size=(1000,7)).astype(np.float32);ref=fit_reference(rng.normal(size=(2000,7)).astype(np.float32))
a,b=transform(x,ref)
for kind,expected in [('robust',a),('percentile',b)]:
    result=relative(x,ref,kind)
    np.testing.assert_array_equal(result[:,:4],x[:,:4])
    np.testing.assert_allclose(result[:,4:],expected[:,4:],rtol=1e-6,atol=1e-7)
print(json.dumps(dict(status='passed',checks=['anomaly channels unchanged','appearance-only transformation matches full calculation'],performance_claim=False)))
