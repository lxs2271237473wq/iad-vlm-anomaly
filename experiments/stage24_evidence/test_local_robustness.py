import json
import numpy as np
import torch
from local_learning_robustness import convex_linear,dense_scores

rng=np.random.default_rng(123)
x=rng.normal(size=(2000,7)).astype(np.float32)
y=(rng.random(2000)<1/(1+np.exp(-x[:,0]))).astype(np.float32)
model,report=convex_linear(x,y,np.ones(2000))
assert report['success'] and report['max_gradient']<1e-5
mask=torch.tensor([1,0,0,0,0,0,0],device='cuda',dtype=torch.float32)
models={'test':(model,mask)}
mean=np.zeros(7,np.float32);scale=np.ones(7,np.float32)
a=dense_scores(models,x,mean,scale)
changed=x.copy();changed[:,1:]+=100
b=dense_scores(models,changed,mean,scale)
assert a==b
print(json.dumps(dict(status='passed',linear_optimizer=report,masked_inputs_do_not_affect_predictions=True,performance_claim=False)))
