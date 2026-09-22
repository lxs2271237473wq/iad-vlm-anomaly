"""CPU-only reassessment. No inference or remote execution.

Pooled observations include repeated physical instances across conditions; they
are not independent samples. No instance-level de-duplication or uncertainty CI.
"""
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import tifffile

BINS, MAX_FPR = 65536, .05
BANDS = ('all', 'tiny', 'small', 'large')
STRATA = ('touching', 'near', 'far')
cv2.setNumThreads(2)


def aupro(neg, reg, n):
    if not n or not neg.sum():
        return None
    x = np.r_[0., np.cumsum(neg[::-1], dtype=np.float64)] / neg.sum()
    y = np.r_[0., np.cumsum(reg[::-1], dtype=np.float64)] / n
    k = np.searchsorted(x, MAX_FPR, side='right')
    xx, yy = x[:k], y[:k]
    if xx[-1] < MAX_FPR:
        yy = np.r_[yy, yy[-1] + (y[k]-yy[-1]) * (MAX_FPR-xx[-1]) / (x[k]-xx[-1])]
        xx = np.r_[xx, MAX_FPR]
    return float(np.sum(np.diff(xx) * (yy[1:] + yy[:-1]) / 2) / MAX_FPR)


def edge_distance(shape):
    h, w = shape
    y, x = np.indices(shape)
    return np.minimum(np.minimum(y, h-1-y), np.minimum(x, w-1-x))


def stratum_for(d, short):
    return 'touching' if d == 0 else 'near' if d <= .05 * short else 'far'


def map_path(root, r):
    ip = Path(r['image_path'])
    return root / r['category'] / 'component_maps/seed=0' / r['category'] / ip.parent.name / (ip.stem + '_tiled.tiff')


def load_sample(root, data, r):
    p = map_path(root, r)
    if not p.is_file():
        raise FileNotFoundError(p)
    s = np.asarray(tifffile.imread(p), np.float32)
    if s.ndim != 2 or not s.size or not np.isfinite(s).all():
        raise ValueError(f'Invalid/nonfinite score map: {p}')
    ip = data / r['image_path']
    if not ip.is_file():
        raise FileNotFoundError(ip)
    image = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Unreadable image: {ip}')
    label = int(r['label'])
    if label == 0:
        mask = np.zeros(image.shape, np.uint8)
    elif label == 1:
        mp = data / r['mask_path']
        if not mp.is_file():
            raise FileNotFoundError(mp)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape:
            raise ValueError(f'Unreadable/wrong-shape mask: {mp}')
        mask = (mask > 0).astype(np.uint8)
    else:
        raise ValueError(f'Invalid label: {label}')
    if s.shape != mask.shape:
        s = cv2.resize(s, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(s).all():
        raise ValueError(f'Nonfinite resized map: {p}')
    return s, mask


def histograms(s, mask):
    q = np.rint(np.clip(s, 0, 2) * ((BINS-1)/2)).astype(np.int32)
    neg = np.bincount(q[mask == 0], minlength=BINS)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for j in range(1, n):
        x, y, w, h, area = map(int, stats[j])
        selected = labels[y:y+h, x:x+w] == j
        rh = np.bincount(q[y:y+h, x:x+w][selected], minlength=BINS) / area
        ratio = area / mask.size
        band = 'tiny' if ratio <= .001 else 'small' if ratio <= .01 else 'large'
        # Bounding-box extrema are attained by region pixels: exact minimum.
        d = min(x, y, mask.shape[1]-x-w, mask.shape[0]-y-h)
        regions.append((j, area, band, d, rh))
    return neg, regions


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def run(roots, data, meta, out, cats, border=False):
    if not data.is_dir():
        raise FileNotFoundError(data)
    if not meta.is_file():
        raise FileNotFoundError(meta)
    with meta.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Metadata CSV is empty')
    for cat in cats:
        if not any(r['category'] == cat for r in rows):
            raise ValueError(f'No metadata rows for category {cat}')
    out.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a previous reassessment as well as legacy results.
    outputs = ['coverage.csv', 'metrics.csv', 'normal_drift.csv', 'border_normal_scores.csv', 'region_border.csv', 'complete.json']
    if any((out / name).exists() for name in outputs):
        raise FileExistsError(f'Reassessment outputs already exist: {out}')
    coverage, metrics, drift, normals, rr = [], [], [], [], []
    for variant, root in roots.items():
        for cat in cats:
            cr = [r for r in rows if r['category'] == cat]
            missing = [str(map_path(root, r)) for r in cr if not map_path(root, r).is_file()]
            coverage.append(dict(variant=variant, category=cat, expected=len(cr), have=len(cr)-len(missing), status='partial' if missing else 'complete', missing_maps=json.dumps(missing)))
    write_csv(out/'coverage.csv', ['variant','category','expected','have','status','missing_maps'], coverage)
    partial_variants = {c['variant'] for c in coverage if c['status'] == 'partial'}
    for c in coverage:
        if c['status'] == 'partial':
            print('ERROR incomplete map coverage: ' + json.dumps(c), file=sys.stderr)
    for variant, root in roots.items():
        # Complete means every requested category is covered, not just one pair.
        if variant in partial_variants:
            continue
        for cat in cats:
            st, nq, nb = {}, {}, {}
            def get(cond, stratum, band):
                return st.setdefault((cond,stratum,band), dict(neg=np.zeros(BINS,np.int64), reg=np.zeros(BINS), n=0))
            for r in [r for r in rows if r['category'] == cat]:
                s, mask = load_sample(root, data, r)
                conds = (r['condition'], '__pooled__')
                if r['condition'].startswith('__'):
                    raise ValueError('Reserved condition name')
                if int(r['label']) == 0:
                    for cond in conds:
                        nq.setdefault(cond, []).append(float(np.quantile(s,.99)))
                        if border:
                            band_mask = edge_distance(s.shape) <= .02*min(s.shape)
                            if not (~band_mask).any():
                                raise ValueError('Normal image has no interior pixels')
                            nb.setdefault(cond, []).append((float(s[band_mask].mean()),float(s[~band_mask].mean())))
                neg, regions = histograms(s, mask)
                for cond in conds:
                    for stratum in STRATA if border else ('all',):
                        for band in BANDS:
                            get(cond,stratum,band)['neg'] += neg
                for j, area, band, d, rh in regions:
                    stratum = stratum_for(d,min(mask.shape)) if border else 'all'
                    for cond in conds:
                        for target in ('all',band):
                            z = get(cond,stratum,target)
                            z['reg'] += rh
                            z['n'] += 1
                    if border:
                        rr.append(dict(category=cat,condition=r['condition'],image_path=r['image_path'],region_id=j,area_px=area,band=band,border_dist_px=d,border_dist_rel=d/min(mask.shape),stratum=stratum))
                # Normal rows deliberately reach histogram accumulation above.
            cat_metrics = []
            for (cond,stratum,band), z in st.items():
                cat_metrics.append(dict(variant=variant,category=cat,condition=cond,stratum=stratum,band=band,regions=z['n'],negative_pixels=int(z['neg'].sum()),aupro_0_05=aupro(z['neg'],z['reg'],z['n'])))
            metrics.extend(cat_metrics)
            for stratum in STRATA if border else ('all',):
                for band in BANDS:
                    valid = [m['aupro_0_05'] for m in cat_metrics if m['condition'] != '__pooled__' and m['stratum']==stratum and m['band']==band and m['aupro_0_05'] is not None]
                    metrics.append(dict(variant=variant,category=cat,condition='__condition_macro__',stratum=stratum,band=band,regions=None,negative_pixels=None,aupro_0_05=float(np.mean(valid)) if valid else None))
            base = float(np.median(nq['regular'])) if nq.get('regular') else None
            for cond, vals in nq.items():
                median = float(np.median(vals))
                drift.append(dict(variant=variant,category=cat,condition=cond,normal_images=len(vals),median_normal_q99=median,drift_vs_regular=median/base if base is not None and base > 0 else None))
            for cond, vals in nb.items():
                b, i = np.mean(vals,axis=0)
                normals.append(dict(category=cat,condition=cond,normal_images=len(vals),mean_score_border_band=float(b),mean_score_interior=float(i),border_over_interior=float(b/i) if i != 0 else None))
    write_csv(out/'metrics.csv', ['variant','category','condition','stratum','band','regions','negative_pixels','aupro_0_05'], metrics)
    write_csv(out/'normal_drift.csv', ['variant','category','condition','normal_images','median_normal_q99','drift_vs_regular'], drift)
    if border:
        write_csv(out/'border_normal_scores.csv', ['category','condition','normal_images','mean_score_border_band','mean_score_interior','border_over_interior'], normals)
        write_csv(out/'region_border.csv', ['category','condition','image_path','region_id','area_px','band','border_dist_px','border_dist_rel','stratum'], rr)
    report = dict(status='partial' if partial_variants else 'complete', coverage=coverage, evaluated_variants=[v for v in roots if v not in partial_variants], excluded_variants=sorted(partial_variants), normal_mask='all zero at native image resolution; all normal pixels enter FPR', protocol='65536 bins [0,2], 8-connectivity, linear interpolation at FPR=.05; macro over defined condition metrics only', border_definition='exact min(y,x,H-1-y,W-1-x); touching d=0; near 0<d<=.05*short; far d>.05*short; NOT directly comparable with old .005 touching definition', limitation='Pooled repeated physical instances across conditions are not independent; no deduplication or confidence intervals.', empty_csv='Header-only when no rows; undefined metrics are blank, never NaN')
    (out/'complete.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if partial_variants:
        raise RuntimeError('Partial coverage: incomplete variants excluded; inspect coverage.csv and complete.json')
    return report
