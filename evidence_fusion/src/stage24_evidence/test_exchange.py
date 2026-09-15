import numpy as np
from PIL import Image
from extract_exchange import search_template,swap_pair,match_references
import cv2

rng=np.random.default_rng(42)
array=rng.integers(0,256,(256,384,3),dtype=np.uint8)
box=(120,80,180,140)
altered=array.copy(); altered[80:140,120:180]=255
t1,m1,b1=search_template(Image.fromarray(array),box)
t2,m2,b2=search_template(Image.fromarray(altered),box)
assert np.array_equal(m1,m2) and b1==b2
assert np.array_equal(t1[m1.astype(bool)],t2[m2.astype(bool)]),'Candidate pixels leaked into matching context'
q=Image.fromarray(array); r=Image.fromarray(255-array)
removed,inserted=swap_pair(q,r,box)
outside=np.ones(array.shape[:2],dtype=bool); outside[80:140,120:180]=False
assert np.array_equal(np.asarray(removed)[outside],array[outside])
assert np.array_equal(np.asarray(inserted)[outside],(255-array)[outside])
assert np.array_equal(np.asarray(removed)[80:140,120:180],(255-array)[80:140,120:180])
small=cv2.resize(array,(128,128),interpolation=cv2.INTER_AREA).astype(np.float32)/255
refs=[dict(image_id='a',image=q,small=small),dict(image_id='b',image=q,small=small)]
matches,reason=match_references(q,box,(90,50,210,170),refs)
assert reason=='ok' and len(matches)==2 and matches[0]['image'].size==(120,120)
assert matches[0]['error']<1e-5
print('PASS: candidate-content exclusion; exact two-way pixel swap; outside pixels unchanged; masked reference search')
