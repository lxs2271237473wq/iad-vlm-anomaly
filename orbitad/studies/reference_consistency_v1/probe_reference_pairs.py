#!/usr/bin/env python3
"""Frozen-backbone paired-reference pilot; not native-resolution AU-PRO.
No network/SSH orchestration. GPU execution requires explicit invocation.
Distances are cosine distances (half squared L2 for unit vectors).
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
LAYERS = [5, 11, 17, 23]
NOTICE = ('Pilot global448 token-grid anchors, not native-GT AU-PRO. Local672 '
          'features are sampled onto the coarser global grid; tiny defects may not '
          'be represented even when footprint labels are positive. Local density '
          'and intensity are not preserved. All matching conditions use identical '
          'anchors and bank. A74 reference list was not saved: this is a '
          'reconstruction, not proven identical references or metrics.')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_meta(path, category, schema=None):
    """Explicit schema maps canonical fields to actual column names; fail closed."""
    aliases = {'category': ['category', 'object', 'object_name'],
               'image_path': ['image_path', 'image', 'path', 'img_path'],
               'mask_path': ['mask_path', 'mask', 'gt_path'],
               'label': ['label', 'is_anomaly', 'anomaly'],
               'instance_id': ['instance_id', 'instance', 'sample_id'],
               'condition': ['condition', 'lighting', 'illumination']}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        mapping = {}
        for key, names in aliases.items():
            candidates = [schema[key]] if schema and key in schema else names
            mapping[key] = next((x for x in candidates if x in fields), None)
        for key in ('category', 'image_path', 'label', 'instance_id', 'condition'):
            if not mapping[key]:
                raise ValueError(f'{path}: missing {key}; columns={fields}; supply --schema-json')
        out = []
        for raw in reader:
            row = {k: (raw.get(v) or '').strip() if v else '' for k, v in mapping.items()}
            if row['category'] != category:
                continue
            val = row['label'].lower()
            if val not in ('0', '1', 'good', 'bad', 'normal', 'anomaly', 'false', 'true'):
                raise ValueError(f'Unsupported label {val!r}')
            row['label'] = int(val in ('1', 'bad', 'anomaly', 'true'))
            if any(not row[k] for k in ('image_path', 'instance_id', 'condition')):
                raise ValueError(f'Incomplete metadata: {row}')
            out.append(row)
    if not out:
        raise ValueError(f'No rows for {category} in {path}')
    if len({r['image_path'] for r in out}) != len(out):
        raise ValueError('Duplicate image paths in metadata')
    return out


def select_rows(rows, per_label, seed):
    # Instance IDs may repeat across good/bad; stratify using the pair, not filename.
    groups = {}
    for r in rows:
        groups.setdefault((r['label'], r['instance_id']), []).append(r)
    selected = set()
    for label in (0, 1):
        keys = [k for k in groups if k[0] == label]
        keys.sort(key=lambda k: hashlib.sha256(f'{seed}|{k[0]}|{k[1]}'.encode()).hexdigest())
        selected.update(keys[:per_label])
    chosen = [r for r in rows if (r['label'], r['instance_id']) in selected]
    rest = [r for r in rows if (r['label'], r['instance_id']) not in selected]
    return chosen, rest


def tiles(width, height):
    side = min(width, height)
    long = max(width, height)
    starts = list(range(0, long, side))
    if starts[-1] + side > long:
        starts[-1] = long - side
    return [(s, 0, s + side, side) if width >= height else (0, s, side, s + side)
            for s in sorted(set(starts))]


def geometry(width, height, resized_width, resized_height, gh, gw, origin=(0, 0)):
    # Use actual integer resize dimensions; long-axis rounding is not a stretch.
    sx, sy = resized_width / width, resized_height / height
    yy, xx = np.mgrid[:gh, :gw]
    coords = np.stack([(xx.ravel() + .5) * 14 / sx + origin[0],
                       (yy.ravel() + .5) * 14 / sy + origin[1]], axis=1).astype('float32')
    half = np.array([7 / sx, 7 / sy], dtype='float32')
    boxes = np.concatenate([coords - half, coords + half], axis=1)
    info = dict(native_wh=[width, height], resized_wh=[resized_width, resized_height],
                cropped_wh=[gw * 14, gh * 14], scale_xy=[sx, sy], origin_xy=list(origin),
                valid_native_box=[origin[0], origin[1], origin[0] + gw * 14 / sx,
                                  origin[1] + gh * 14 / sy], grid_hw=[gh, gw])
    return coords, boxes, info


def normalize(x):
    return F.normalize(torch.as_tensor(x, dtype=torch.float32), dim=-1).numpy()


def encode(model, image, resolution, origin=(0, 0)):
    model.set_smaller_edge_size(resolution)
    # Invoke exact wrapper resize to measure integer rounding, then wrapper preparation.
    resized = model.transform.transforms[0](image)
    tensor, (gh, gw) = model.prepare_image(image)
    coords, boxes, geo = geometry(*image.size, *resized.size, gh, gw, origin)
    feats = np.stack(model.extract_features(tensor)).reshape(4, gh * gw, -1).astype('float32')
    if feats.shape[-1] != 1024:
        raise ValueError('Expected DINOv2-L 1024-dimensional features')
    return feats, coords, boxes, geo


def sample_local(feats, geo, coords):
    gh, gw = geo['grid_hw']
    origin = np.array(geo['origin_xy'])
    scale = np.array(geo['scale_xy'])
    p = (coords - origin) * scale
    valid = (p[:, 0] >= 0) & (p[:, 0] < gw * 14) & (p[:, 1] >= 0) & (p[:, 1] < gh * 14)
    grid = torch.tensor(2 * p / np.array([gw * 14, gh * 14]) - 1, dtype=torch.float32)
    grid = grid.reshape(1, 1, -1, 2).expand(4, -1, -1, -1)
    fmap = torch.from_numpy(feats).reshape(4, gh, gw, -1).permute(0, 3, 1, 2)
    # border padding represents nearest token at half-token outer rim, not zero dilution.
    values = F.grid_sample(fmap, grid, mode='bilinear', padding_mode='border',
                           align_corners=False)[:, :, 0].permute(0, 2, 1).numpy()
    return values, valid


def paired(model, image):
    g, coords, boxes, gg = encode(model, image, 448)
    local = np.zeros_like(g)
    count = np.zeros(len(coords), dtype='int32')
    geos = []
    for box in tiles(*image.size):
        f, _, _, geo = encode(model, image.crop(box), 672, box[:2])
        v, valid = sample_local(f, geo, coords)
        local[:, valid] += v[:, valid]
        count[valid] += 1
        geos.append(geo)
    if (count == 0).any():
        raise ValueError('Uncovered global anchors: geometry error')
    local /= count[None, :, None]
    return np.stack([normalize(g), normalize(local)]), coords, boxes, dict(global_view=gg, local_views=geos,
                 overlap_counts_minmax=[int(count.min()), int(count.max())])


def retrieve(query, bank, ref_ids, permutation, device='cpu', qchunk=64, bchunk=2048, topk=8, temperature=.05):
    """Exact full-bank blocked FP32 cosine search. Bank stays CPU/memmap.
    bank/query shape [2,4,N,D]. No full NxM distance matrix retained.
    """
    n = query.shape[2]
    m = bank.shape[2]
    k = min(topk, m)
    nr = int(np.max(ref_ids)) + 1
    output = {}
    def emit(name, tensor):
        output.setdefault(name, []).append(tensor.detach().cpu().numpy())
    for qa in range(0, n, qchunk):
        q = torch.as_tensor(np.array(query[:, :, qa:qa + qchunk]), device=device, dtype=torch.float32)
        nq = q.shape[2]
        legacy = torch.full((2, 4, nq), float('inf'), device=device)
        legacy_ids = torch.full((2, 4, nq), -1, device=device, dtype=torch.long)
        refmin = torch.full((2, nq, nr), float('inf'), device=device)
        best = {name: (torch.full((nq, k), float('inf'), device=device),
                       torch.full((nq, k), -1, device=device, dtype=torch.long))
                for name in ('global', 'local', 'strict', 'shuffled')}
        for ba in range(0, m, bchunk):
            end = min(m, ba + bchunk)
            b = torch.as_tensor(np.array(bank[:, :, ba:end]), device=device, dtype=torch.float32)
            d = (1 - torch.matmul(q, b.transpose(-1, -2))).clamp_min(0)
            ld, li = d.min(-1)
            better = ld < legacy
            legacy_ids = torch.where(better, li + ba, legacy_ids)
            legacy = torch.minimum(legacy, ld)
            dg, dl = d.mean(1).unbind(0)
            bs = torch.as_tensor(np.array(bank[1][:, permutation[ba:end], :]), device=device)
            ds = (1 - torch.matmul(q[1], bs.transpose(-1, -2))).clamp_min(0).mean(0)
            values = dict(global_=dg, local=dl, strict=(dg + dl) / 2, shuffled=(dg + ds) / 2)
            values['global'] = values.pop('global_')
            for name, matrix in values.items():
                old, oi = best[name]
                ids = torch.arange(ba, end, device=device).expand(nq, -1)
                combined = torch.cat([old, matrix], 1)
                ci = torch.cat([oi, ids], 1)
                val, ix = torch.topk(combined, k, largest=False, sorted=True)
                best[name] = val, ci.gather(1, ix)
            for rid in np.unique(ref_ids[ba:end]):
                sel = torch.as_tensor(ref_ids[ba:end] == rid, device=device)
                refmin[0, :, rid] = torch.minimum(refmin[0, :, rid], dg[:, sel].min(1).values)
                refmin[1, :, rid] = torch.minimum(refmin[1, :, rid], dl[:, sel].min(1).values)
        emit('score_legacy_layer_independent', legacy.mean((0, 1)))
        emit('legacy_layer_top1_ids', legacy_ids.permute(2, 0, 1))
        emit('legacy_layer_min_distances', legacy.permute(2, 0, 1))
        emit('score_perview_four_layer_joint_independent', (best['global'][0][:, 0] + best['local'][0][:, 0]) / 2)
        relaxed, rids = refmin.mean(0).min(1)
        emit('score_same_reference_relaxed', relaxed)
        emit('same_reference_top1_id', rids)
        for name, (d, ids) in best.items():
            emit('score_' + name, d[:, 0])
            emit(name + '_topk_ids', ids)
            emit(name + '_topk_distances', d)
            w = torch.softmax(-(d - d[:, :1]) / temperature, dim=1)
            emit(name + '_topk_soft_distance', (w * d).sum(1))
            emit(name + '_topk_entropy', -(w * w.clamp_min(1e-30).log()).sum(1))
        emit('global_local_top1_same_row', best['global'][1][:, 0] == best['local'][1][:, 0])
    return {key: np.concatenate(value) for key, value in output.items()}


def gt_diagnostics(mask, coords, boxes):
    h, w = mask.shape
    labels, fractions, centers = [], [], []
    covered = np.zeros_like(mask, dtype=bool)
    for (x, y), (x0, y0, x1, y1) in zip(coords, boxes):
        a, b = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        c, d = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        patch = mask[b:d, a:c]
        labels.append(bool(patch.any()))
        fractions.append(float(patch.mean()) if patch.size else 0.)
        centers.append(bool(mask[min(h - 1, int(y)), min(w - 1, int(x))]))
        covered[b:d, a:c] = True
    # Component coverage is diagnostic only, after all score computations.
    from scipy.ndimage import label
    components, nc = label(mask, structure=np.ones((3, 3)))
    component_report = []
    for i in range(1, nc + 1):
        region = components == i
        center_hits = sum(region[min(h - 1, int(y)), min(w - 1, int(x))] for x, y in coords)
        component_report.append(dict(pixels=int(region.sum()), covered_pixels=int((region & covered).sum()),
                                     token_center_hits=int(center_hits)))
    return dict(labels_footprint_max=np.array(labels, dtype='uint8'),
                labels_footprint_fraction=np.array(fractions, dtype='float32'),
                labels_center=np.array(centers, dtype='uint8')), dict(
                native_pixels=int(h * w), covered_pixels=int(covered.sum()), gt_pixels=int(mask.sum()),
                covered_gt_pixels=int((mask & covered).sum()), components=component_report,
                footprint_definition='integer pixels intersecting native token footprint (floor/ceil)',
                limitation=NOTICE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--category', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/root/private_data/iad-vlm-anomaly'))
    p.add_argument('--data-root', type=Path, required=True, help='directory containing category/train/good')
    p.add_argument('--source-dir', type=Path, default=HERE / 'source')
    p.add_argument('--public-meta', type=Path, default=HERE / 'source/public_meta.csv')
    p.add_argument('--validation-meta', type=Path, default=HERE / 'source/validation_meta.csv')
    p.add_argument('--schema-json', type=Path, help='optional canonical-column to actual-column map')
    p.add_argument('--selection-json', type=Path, help='frozen categories/category/public+validation rows; overrides hash selection')
    p.add_argument('--max-instances', type=int, default=4, help='per label: 4 good + 4 bad; all conditions bound')
    p.add_argument('--max-val', type=int, default=6, help='maximum validation diagnostic instances total')
    p.add_argument('--smoke', action='store_true', help='first 2 reconstructed refs; NONCOMPARABLE engineering only')
    p.add_argument('--resume', action='store_true', help='strict manifest and content hashes required')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=74)
    p.add_argument('--qchunk', type=int, default=64)
    p.add_argument('--bchunk', type=int, default=2048)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--temperature', type=float, default=.05)
    p.add_argument('--tf32', action='store_true', help='default off; recorded; exact FP32 baseline requires off')
    return p.parse_args()


def main():
    a = parse_args()
    os.environ['DINOV2_SDPA'] = '0'  # frozen attention opt-in must remain disabled
    if min(a.max_instances, a.qchunk, a.bchunk, a.topk) < 1 or a.max_val < 0 or a.temperature <= 0:
        raise ValueError('Invalid limits')
    if a.tf32:
        raise ValueError('This exact FP32 probe forbids TF32; omit --tf32')
    if not a.device.startswith('cuda'):
        raise ValueError('Real pilot requires CUDA; CPU reserved for synthetic tests')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(int(os.environ.get('SUPERAD_TORCH_THREADS', '8')))
    schema = json.loads(a.schema_json.read_text()) if a.schema_json else None
    public = read_meta(a.public_meta, a.category, schema)
    validation = read_meta(a.validation_meta, a.category, schema)
    chosen, _ = select_rows(public, a.max_instances, a.seed)
    # Validation bounded by total instance count; preserve all conditions.
    vgroups = sorted({(r['label'], r['instance_id']) for r in validation},
                     key=lambda k: hashlib.sha256(f'{a.seed}|val|{k}'.encode()).hexdigest())
    vkeys = set(vgroups[:a.max_val])
    val = [r for r in validation if (r['label'], r['instance_id']) in vkeys]
    calibration = [r for r in validation if (r['label'], r['instance_id']) not in vkeys]
    if a.selection_json:
        frozen = json.loads(a.selection_json.read_text(encoding='utf-8'))['categories'][a.category]
        def select_frozen(key, available):
            by_path = {r['image_path']: r for r in available}
            result = []
            for saved in frozen[key]:
                path = saved['image_path']
                if path not in by_path:
                    raise ValueError(f'Frozen selection path absent from metadata: {path}')
                canonical = by_path[path]
                for field in ('category', 'instance_id', 'condition', 'label', 'mask_path'):
                    if field in saved and str(saved[field]) != str(canonical[field]):
                        raise ValueError(f'Frozen selection metadata mismatch: {path}/{field}')
                result.append(canonical)
            if len({r['image_path'] for r in result}) != len(result):
                raise ValueError('Duplicate frozen selections')
            keys = {(r['label'], r['instance_id']) for r in result}
            if {r['image_path'] for r in available if (r['label'], r['instance_id']) in keys} != {r['image_path'] for r in result}:
                raise ValueError('Frozen selection must bind all conditions of each instance')
            return result
        chosen = select_frozen('public', public)
        val = select_frozen('validation', validation)
        chosen_v = {r['image_path'] for r in val}
        calibration = [r for r in validation if r['image_path'] not in chosen_v]
    if a.smoke:
        chosen = [next(r for r in chosen if r['label'] == label) for label in (0, 1)
                  if any(r['label'] == label for r in chosen)]
        val = val[:1]
    rows = [('public', r) for r in chosen] + [('validation', r) for r in val]
    def resolve(s):
        path = Path(s)
        return path if path.is_absolute() else a.data_root / path
    train = sorted((a.data_root / a.category / 'train/good').glob('*'))
    train = [p for p in train if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    if len(train) < (2 if a.smoke else 16):
        raise ValueError('Insufficient normal training images (full=16, smoke=2)')
    training_paths = {p.resolve() for p in train}
    query_paths = [resolve(r['image_path']).resolve() for _, r in rows]
    if training_paths.intersection(query_paths):
        raise ValueError('Training/query overlap')
    if set(resolve(r['image_path']).resolve() for r in public).intersection(resolve(r['image_path']).resolve() for r in validation):
        raise ValueError('Public/validation path overlap')
    if any('train' in resolve(r['image_path']).parts for r in validation):
        raise ValueError('Validation includes training path')
    weights = os.environ.get('DINOV2_LOCAL_WEIGHTS')
    repo = os.environ.get('DINOV2_LOCAL_REPO')
    if not weights or not repo:
        raise ValueError('Set DINOV2_LOCAL_REPO and DINOV2_LOCAL_WEIGHTS: network model downloads forbidden')
    source_files = [a.source_dir / name for name in ('backbones.py', 'sampler.py', 'detection.py', 'test_public.py')]
    source_files += sorted(Path(repo).rglob('*.py'))
    assets = train + [resolve(r['image_path']) for _, r in rows]
    assets += [resolve(r['mask_path']) for _, r in rows if r['mask_path']]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in ('resume', 'out')}
    manifest = dict(version=1, config=config, notice=NOTICE, noncomparable=a.smoke,
                    source_hashes={str(p.resolve()): digest(p) for p in source_files + [Path(__file__)]},
                    weights_sha256=digest(weights), local_repo=str(Path(repo).resolve()),
                    metadata_hashes={str(p): digest(p) for p in (a.public_meta, a.validation_meta)},
                    schema=schema, asset_hashes={str(p.resolve()): digest(p) for p in assets},
                    selected_public=chosen, selected_validation=val, independent_calibration=calibration,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(a.device),
                    tf32_matmul=False, tf32_cudnn=False, sdpa_patch=False,
                    geometry='global448 top-left multiple14 crop; native square A74 windows resized672',
                    metric='mean per-layer cosine distances; no GT influences scores')
    mp = a.out / 'manifest.json'
    if a.out.exists():
        if not a.resume:
            raise FileExistsError('Output exists; use a NEW --out or explicit --resume')
        if not mp.exists() or json.loads(mp.read_text()) != manifest:
            raise ValueError('Resume manifest mismatch/missing: refusing reuse')
    else:
        a.out.mkdir(parents=True)
        save_json(mp, manifest)
    # Cache entries are accepted only with sha256 receipts. Interrupted writes fail closed.
    receipts_path = a.out / 'receipts.json'
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    def verified(name):
        p = a.out / name
        if not p.exists():
            if name in receipts:
                raise ValueError(f'Missing receipted artifact {name}')
            return False
        if name not in receipts or digest(p) != receipts[name]:
            raise ValueError(f'Unverified/partial cache {name}; use a new output directory')
        return True
    def receipt(*names):
        for name in names:
            receipts[name] = digest(a.out / name)
        save_json(receipts_path, receipts)
    timing = {}
    t0 = time.perf_counter()
    backbone = load_module('probe_frozen_backbones', a.source_dir / 'backbones.py')
    sampler_mod = load_module('probe_frozen_sampler', a.source_dir / 'sampler.py')
    model = backbone.get_model('dinov2_vitl14_reg', a.device, smaller_edge_size=448)
    if int(getattr(model.model, 'num_register_tokens', 0)) != 4:
        raise ValueError('Expected reg4 model')
    timing['load_model_seconds'] = time.perf_counter() - t0
    ref_file = 'references.json'
    if verified(ref_file):
        selected = json.loads((a.out / ref_file).read_text())['selected']
    elif a.smoke:
        selected = [str(p) for p in train[:2]]
        save_json(a.out / ref_file, dict(selected=selected, selection='sorted_first_two_engineering_only',
                                       noncomparable=True, reconstruction_not_A74_identity=True))
        receipt(ref_file)
    else:
        t = time.perf_counter()
        cls = []
        for pi, p in enumerate(train):
            print(f'CLS {pi + 1}/{len(train)} {p.name}', flush=True)
            with Image.open(p) as im:
                tensor, _ = model.prepare_image(im.convert('RGB'))
            cls.append(model.extract_cls_features(tensor).cpu())
        features = torch.stack(cls).to(a.device)
        sampler = sampler_mod.GreedyCoresetSampler(.1, torch.device(a.device), dimension_to_project_features_to=1024)
        indices = sampler.run(features).tolist()
        if len(set(indices)) != 16:
            raise ValueError('Original coreset returned duplicate references')
        selected = [str(train[i]) for i in indices[:2 if a.smoke else 16]]
        save_json(a.out / ref_file, dict(selected=selected, full_reconstructed_indices=indices,
                                       candidates=[str(p) for p in train], reconstruction_not_A74_identity=True))
        receipt(ref_file)
        timing['cls_selection_seconds'] = time.perf_counter() - t
        del features, cls
    bank_name = 'bank.float32.npy'
    index_name = 'bank_index.npz'
    if verified(bank_name):
        if not verified(index_name) or not verified('bank_geometry.json'):
            raise ValueError('Incomplete bank cache')
        bank = np.load(a.out / bank_name, mmap_mode='r')
        index = np.load(a.out / index_name)
        ref_ids = index['ref_ids']
    else:
        t = time.perf_counter()
        # Precompute sizes without model inference, then stream each ref into memmap.
        counts = []
        model.set_smaller_edge_size(448)
        for p in selected:
            with Image.open(p) as im:
                resized = model.transform.transforms[0](im)
                counts.append((resized.width // 14) * (resized.height // 14))
        bank = np.lib.format.open_memmap(a.out / bank_name, mode='w+', dtype='float32', shape=(2, 4, sum(counts), 1024))
        ref_ids = np.repeat(np.arange(len(selected), dtype='int32'), counts)
        allcoords, allboxes, geos = [], [], []
        start = 0
        for ri, (p, count) in enumerate(zip(selected, counts)):
            print(f'Paired reference {ri + 1}/{len(selected)} {p}', flush=True)
            with Image.open(p) as im:
                feats, coords, boxes, geo = paired(model, im.convert('RGB'))
            bank[:, :, start:start + count] = feats
            start += count
            allcoords.append(coords)
            allboxes.append(boxes)
            geos.append(dict(image=p, geometry=geo))
        bank.flush()
        np.savez_compressed(a.out / index_name, ref_ids=ref_ids, coords=np.concatenate(allcoords), footprints=np.concatenate(allboxes))
        save_json(a.out / 'bank_geometry.json', geos)
        receipt(bank_name, index_name, 'bank_geometry.json')
        timing['bank_seconds'] = time.perf_counter() - t
    permutation = np.random.default_rng(a.seed).permutation(bank.shape[2])
    if not verified('shuffle.npy'):
        np.save(a.out / 'shuffle.npy', permutation)
        receipt('shuffle.npy')
    elif not np.array_equal(np.load(a.out / 'shuffle.npy'), permutation):
        raise ValueError('Shuffle mismatch')
    coverage = []
    torch.cuda.reset_peak_memory_stats(a.device)
    for qi, (split, row) in enumerate(rows):
        print(f'Query {qi + 1}/{len(rows)} {split} {row["image_path"]}', flush=True)
        key = split + '_' + hashlib.sha256(row['image_path'].encode()).hexdigest()[:20]
        name, report_name = key + '.npz', key + '.json'
        if verified(name):
            if not verified(report_name):
                raise ValueError('Missing query report')
            coverage.append(json.loads((a.out / report_name).read_text()))
            continue
        t = time.perf_counter()
        with Image.open(resolve(row['image_path'])) as im:
            image = im.convert('RGB')
        feats, coords, boxes, geo = paired(model, image)
        torch.cuda.synchronize(a.device)
        te = time.perf_counter()
        scores = retrieve(feats, bank, ref_ids, permutation, a.device, a.qchunk, a.bchunk, a.topk, a.temperature)
        torch.cuda.synchronize(a.device)
        ts = time.perf_counter()
        # GT is first opened AFTER matching; no defect-size selection, model fitting or threshold tuning.
        if row['mask_path']:
            with Image.open(resolve(row['mask_path'])) as im:
                mask = np.asarray(im.convert('L')) > 0
            if mask.shape != (image.height, image.width):
                raise ValueError('GT/native image size mismatch; refusing silent resize')
        elif row['label']:
            raise ValueError('Anomalous row missing mask_path')
        else:
            mask = np.zeros((image.height, image.width), dtype=bool)
        labels, cov = gt_diagnostics(mask, coords, boxes)
        np.savez_compressed(a.out / name, coords=coords, footprints=boxes, **scores, **labels)
        report = dict(row=row, split=split, geometry=geo, coverage=cov,
                      encode_seconds=te - t, search_seconds=ts - te, total_seconds=time.perf_counter() - t,
                      noncomparable=a.smoke, output=name)
        save_json(a.out / report_name, report)
        receipt(name, report_name)
        coverage.append(report)
        del feats, scores
    save_json(a.out / 'coverage_timings.json', dict(images=coverage, timings_this_invocation=timing,
              calibration_reserved_not_scored=calibration, notice=NOTICE, noncomparable=a.smoke,
              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device)))
    print(f'Completed {len(rows)} image variants; {NOTICE}')


if __name__ == '__main__':
    main()
