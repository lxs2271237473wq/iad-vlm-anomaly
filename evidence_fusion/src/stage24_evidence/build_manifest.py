"""Build immutable image identities; inference and evaluation metadata are separate."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

AD2 = {
    'can': (412,46,72,90), 'fabric': (387,43,66,90),
    'fruit_jelly': (263,37,20,60), 'rice': (313,35,42,90),
    'sheet_metal': (137,19,24,90), 'vial': (291,41,35,105),
    'wallplugs': (293,33,60,90), 'walnuts': (432,48,60,90),
}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def inspect_row(row):
    p = Path(row['path'])
    with Image.open(p) as im:
        row.update(width=im.width, height=im.height)
    row.update(sha256=digest(p), bytes=p.stat().st_size)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, labels = [], []
    def add(dataset, category, split, p, y, scene):
        relative = p.relative_to(args.root).as_posix()
        image_id = hashlib.sha256(relative.encode()).hexdigest()[:24]
        rows.append(dict(image_id=image_id, dataset=dataset, category=category,
                         split=split, path=str(p)))
        labels.append(dict(image_id=image_id, label=y, scene_group=scene))
    for category, expected in AD2.items():
        base = args.root/'datasets/MVTec_AD_2'/category
        for (folder,split,y), count in zip([
            ('train/good','train',0), ('validation/good','calibration',0),
            ('test_public/good','evaluation',0), ('test_public/bad','evaluation',1)
        ], expected):
            files = sorted(base.joinpath(folder).glob('*.png'))
            assert len(files)==count, (category,folder,len(files),count)
            for p in files:
                # Scene grouping is an evaluation-only filename convention.
                scene = f'AD2/{category}/{folder}/{p.stem.split("_")[0]}'
                add('AD2',category,split,p,y,scene)
    source = args.root/'datasets/VisA'
    visa = list(csv.DictReader(open(source/'split_csv/1cls.csv')))
    assert len({r['object'] for r in visa})==12
    heldout = set()
    for category in sorted({r['object'] for r in visa}):
        training = [r for r in visa if r['object']==category and r['split']=='train']
        assert all(r['label']=='normal' for r in training)
        ordered = sorted(training,key=lambda r: hashlib.sha256(('stage24-cal-42/'+r['image']).encode()).hexdigest())
        heldout.update(r['image'] for r in ordered[:max(1,round(len(ordered)*.1))])
    for r in visa:
        split = 'calibration' if r['image'] in heldout else ('train' if r['split']=='train' else 'evaluation')
        add('VisA',r['object'],split,source/r['image'],int(r['label']!='normal'),'VisA/'+r['image'])
    assert len({r['image_id'] for r in rows})==len(rows)
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked=[]
        for i,r in enumerate(pool.map(inspect_row, rows),1):
            checked.append(r)
            if i%500==0: print('hashed',i,'/',len(rows),flush=True)
    hashes=defaultdict(list)
    for r in checked: hashes[r['sha256']].append(r)
    duplicates=[[dict(image_id=r['image_id'],dataset=r['dataset'],split=r['split']) for r in group]
                for group in hashes.values() if len(group)>1]
    cross_split=[g for g in duplicates if len({r['split'] for r in g})>1]
    write_csv(args.out/'images.csv',checked)
    write_csv(args.out/'evaluation_labels.csv',labels)
    counts=Counter((r['dataset'],r['category'],r['split']) for r in checked)
    summary=dict(total=len(rows),counts={'/'.join(k):v for k,v in sorted(counts.items())},
                 duplicate_hash_groups=duplicates,cross_split_duplicates=cross_split,
                 scene_group_status='AD2 filename prefix convention; not independently verified physical-object metadata',
                 visa_calibration='10% of official normal train, SHA256 ordering stage24-cal-42',
                 inference_manifest_sha256=digest(args.out/'images.csv'))
    (args.out/'manifest_report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    assert not cross_split, 'Cross-split identical images found; resolve before fitting.'

if __name__=='__main__': main()
