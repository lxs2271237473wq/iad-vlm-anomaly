"""Meaningful regression checks for historical crop offsets and label isolation."""
import numpy as np
from PIL import Image
from common import map_box_to_original,context_box,candidates,robust_fit,robust_apply

def main():
    assert map_box_to_original((0,0,256,256),(256,256),(1234,901))==(0,0,1234,901)
    assert map_box_to_original((240,240,256,256),(256,256),(1024,512))==(960,480,1024,512)
    a=np.zeros((256,256),dtype=np.float32); a[245:256,245:256]=1
    rows=candidates(a,(1024,512))
    assert len(rows)==1
    box=rows[0]['box']; assert box[2:]==[1024,512]
    synthetic=np.zeros((512,1024,3),dtype=np.uint8); synthetic[490:,980:]=[255,0,0]
    im=Image.fromarray(synthetic)
    crop=np.asarray(im.crop(box)); assert (crop[:,:,0]==255).any()
    assert np.array_equal(crop,synthetic[box[1]:box[3],box[0]:box[2]])
    assert rows[0]['context_box'][2:]==[1024,512]
    assert candidates(np.zeros((256,256)),(1024,512))==[]
    fit=robust_fit([1,2,3,4]); original=robust_apply([5,7],fit)
    # A later extreme test image must not change earlier predictions.
    assert np.array_equal(original,robust_apply([5,7,100000],fit)[:2])
    print('PASS: full-field geometry, border ROI, pixel equivalence, empty-map fallback, test-composition-independent calibration')

if __name__=='__main__': main()
