"""Small synthetic CPU tests; fixtures stay inside evaluation/ and are removed."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cv2
import tifffile
from evaluation_common_v2 import aupro, histograms, edge_distance, stratum_for, load_sample, map_path, run, write_csv

class EvaluationTests(unittest.TestCase):
    def test_normal_denominator(self):
        s = np.array([[2.,0.]])
        neg, regs = histograms(s,np.array([[1,0]],np.uint8))
        normal_neg, normal_regs = histograms(np.zeros((2,3)),np.zeros((2,3),np.uint8))
        self.assertEqual(neg.sum(),1)
        self.assertEqual((neg+normal_neg).sum(),7)
        self.assertFalse(normal_regs)
        self.assertEqual(aupro(neg,regs[0][-1],1),1.)

    def test_integration_cutoff_ties(self):
        # All scores tied: PRO=FPR, normalized area to .05 is .025.
        self.assertAlmostEqual(aupro(np.array([100,0]),np.array([1.,0]),1),.025)
        # PRO jumps to one exactly at the cutoff: triangle area / .05 = .5.
        self.assertAlmostEqual(aupro(np.array([95,5]),np.array([0.,1.]),1),.5)
        # Zero-FPR vertical segment must be retained.
        self.assertAlmostEqual(aupro(np.array([100,0]),np.array([0.,1.]),1),1.)
        self.assertIsNone(aupro(np.zeros(2),np.ones(2),1))
        self.assertIsNone(aupro(np.ones(2),np.zeros(2),0))

    def test_exact_border(self):
        bd = edge_distance((101,120))
        self.assertEqual(bd[5,50],5)
        self.assertEqual(bd[100,50],0)
        self.assertEqual(stratum_for(0,100),'touching')
        self.assertEqual(stratum_for(1,100),'near')
        self.assertEqual(stratum_for(5,100),'near')
        self.assertEqual(stratum_for(6,100),'far')

    def test_end_to_end_and_errors(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            base=Path(tmp); data=base/'data'; data.mkdir(); root=base/'maps'
            rows=[]
            for name,label,value in [('normal',0,.2),('defect',1,1.8)]:
                ip=f'cat/test/{name}.png'; mp=f'cat/mask/{name}.png'
                (data/ip).parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(data/ip),np.zeros((20,20),np.uint8))
                r=dict(category='cat',condition='regular',image_path=ip,mask_path=mp,label=label)
                rows.append(r)
                p=map_path(root,r); p.parent.mkdir(parents=True,exist_ok=True)
                tifffile.imwrite(p,np.full((20,20),value,np.float32))
                if label:
                    (data/mp).parent.mkdir(parents=True,exist_ok=True)
                    m=np.zeros((20,20),np.uint8); m[0,5]=255
                    cv2.imwrite(str(data/mp),m)
            meta=base/'meta.csv'; write_csv(meta,list(rows[0]),rows)
            out=base/'out'; run({'v':root},data,meta,out,['cat'],border=True)
            with (out/'metrics.csv').open() as f: metrics=list(csv.DictReader(f))
            target=next(m for m in metrics if m['condition']=='__pooled__' and m['stratum']=='touching' and m['band']=='all')
            self.assertEqual(int(target['negative_pixels']),799)
            with (out/'border_normal_scores.csv').open() as f: normals=list(csv.DictReader(f))
            self.assertTrue(all(int(n['normal_images'])==1 for n in normals))
            self.assertAlmostEqual(float(normals[0]['mean_score_border_band']),.2,places=6)
            with self.assertRaises(FileExistsError): run({'v':root},data,meta,out,['cat'])
            # Normal mask is zero despite its nonexistent mask_path.
            self.assertEqual(load_sample(root,data,rows[0])[1].sum(),0)
            p=map_path(root,rows[0]); tifffile.imwrite(p,np.full((20,20),np.nan,np.float32))
            with self.assertRaisesRegex(ValueError,'nonfinite'): load_sample(root,data,rows[0])
            p.unlink()
            with self.assertRaises(FileNotFoundError): load_sample(root,data,rows[0])
            with self.assertRaisesRegex(RuntimeError,'Partial coverage'): run({'v':root},data,meta,base/'partial',['cat'])
            report=json.loads((base/'partial/complete.json').read_text())
            self.assertEqual(report['status'],'partial')
            self.assertEqual(report['evaluated_variants'],[])
            self.assertEqual(len((base/'partial/metrics.csv').read_text().splitlines()),1)
            (data/rows[1]['mask_path']).unlink()
            with self.assertRaises(FileNotFoundError): load_sample(root,data,rows[1])

if __name__=='__main__': unittest.main(verbosity=2)
