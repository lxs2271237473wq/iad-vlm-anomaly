"""Sample exclusion and dense-inference regression checks."""
import json
import cv2
import numpy as np
import torch
from local_learning import sample_indices,score

rng=np.random.default_rng(42)
x=rng.normal(size=(512*512,7)).astype(np.float32)
mask=np.zeros((512,512),dtype=bool);mask[100:110,100:110]=True
idx,y=sample_indices(x,mask,rng)
excluded=cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8)).ravel()>0
assert mask.ravel()[idx[y==1]].all()
assert not excluded[idx[y==0]].any()
assert len(idx)==len(np.unique(idx))
normal_idx,normal_y=sample_indices(x,np.zeros_like(mask),rng)
assert not normal_y.any() and len(normal_idx)>0
linear=torch.nn.Linear(7,1,bias=False).cuda()
with torch.no_grad():linear.weight.zero_();linear.weight[0,0]=1
actual=score(linear,x,np.zeros(7,dtype=np.float32),np.ones(7,dtype=np.float32))
assert np.isclose(actual,x[:,0].max(),atol=1e-6)
print(json.dumps(dict(status='passed',checks=['positive samples inside GT','negatives outside dilated GT','unique samples','normal-only negatives','chunked max equals analytic linear score'],performance_claim=False)))
