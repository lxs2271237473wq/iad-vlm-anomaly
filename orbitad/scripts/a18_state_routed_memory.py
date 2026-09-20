"""A18: normal-state prototypes and routed patch memory for AD2."""
from pathlib import Path
from collections import Counter
import ast
import csv
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision.transforms.functional import to_tensor


ROOT = Path("/root/private_data/iad-vlm-anomaly")
DATA = ROOT / "datasets/MVTec_AD_2"
A1 = ROOT / "orbitad/results/a1_anomaly_score_sensitivity"
A10 = ROOT / "orbitad/results/a10_multilayer_v1"
OUT = ROOT / "orbitad/results/a18_state_routed_memory_v1"
LAYERS = [2, 5, 8, 11]
K_STATES = 4
TOP_STATES = 2
SEED = 20260916


def l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def spherical_kmeans(x, k, seed):
    """Deterministic farthest-point initialization followed by cosine k-means."""
    x = l2_normalize(np.asarray(x, dtype=np.float32))
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(len(x)))]]
    for _ in range(1, k):
        similarity = x @ np.stack(centers).T
        centers.append(x[int(np.argmin(similarity.max(axis=1)))])
    centers = np.stack(centers)
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(50):
        new_labels = np.argmax(x @ centers.T, axis=1)
        new_centers = []
        for state in range(k):
            members = x[new_labels == state]
            if len(members) == 0:
                similarity = x @ centers.T
                new_centers.append(x[int(np.argmin(similarity.max(axis=1)))])
            else:
                center = members.mean(axis=0, keepdims=True)
                new_centers.append(l2_normalize(center)[0])
        new_centers = np.stack(new_centers)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    return centers.astype(np.float32), labels


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert (A10 / "MAPS_COMPLETE.json").exists()
    OUT.mkdir(parents=True, exist_ok=False)

    source = ROOT / "orbitad/third_party/SuperADD/tracks/industrial/src/industrial/model.py"
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PatchedExecution"]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), globals())
    patcher = PatchedExecution(640, 128, 16)

    meta = {split: list(csv.DictReader(open(A10 / f"{split}_meta.csv"))) for split in ("validation", "public")}
    for split, rows in meta.items():
        with open(OUT / f"{split}_meta.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    protocol = {
        "model": "vit_base_patch16_dinov3.lvd1689m",
        "layers": LAYERS,
        "global_descriptor": "L2-normalized mean of full512 layer-11 patch tokens",
        "normal_prototype_data": "train/good + validation/good only",
        "states_per_category": K_STATES,
        "clustering": "spherical k-means; deterministic farthest initialization",
        "routing": f"top-{TOP_STATES} nearest cosine prototypes; union their patch memory tokens",
        "memory_source": "A10 patch640 memory and references; no added tokens",
        "memory_budget": "unchanged from A10; routed subset only",
        "score": "mean over 4 layers of Euclidean NN distance / 768",
        "public_role": "development evaluation only; labels unused during map generation",
        "seed": SEED,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))

    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]

    global_model = timm.create_model(protocol["model"], pretrained=True, num_classes=0, img_size=512).eval().cuda()

    @torch.inference_mode()
    def global_descriptor(path):
        with Image.open(path) as im:
            im = im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
            x = to_tensor(im)[None].cuda()
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feature = global_model.forward_intermediates(
                x, indices=[11], norm=False, output_fmt="NLC", intermediates_only=True
            )[0].float()
        descriptor = F.normalize(feature.mean(dim=1), dim=1)
        return descriptor[0].cpu().numpy().astype(np.float32)

    categories = sorted({row["category"] for row in meta["public"]})
    state_data = {}
    progress = []
    for category in categories:
        tick = time.time()
        train_paths = sorted(
            p for p in (DATA / category / "train/good").iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        )
        val_rows = [r for r in meta["validation"] if r["category"] == category]
        public_rows = [r for r in meta["public"] if r["category"] == category]
        train_desc = np.stack([global_descriptor(p) for p in train_paths])
        val_desc = np.stack([global_descriptor(DATA / r["image_path"]) for r in val_rows])
        centers, train_labels = spherical_kmeans(train_desc, K_STATES, SEED)
        val_labels = np.argmax(val_desc @ centers.T, axis=1)
        refined = []
        for state in range(K_STATES):
            members = np.concatenate([train_desc[train_labels == state], val_desc[val_labels == state]])
            refined.append(l2_normalize(members.mean(axis=0, keepdims=True))[0])
        centers = np.stack(refined).astype(np.float32)
        labels = np.concatenate([train_labels, np.argmax(val_desc @ centers.T, axis=1)])
        val_routes = np.argsort(-(val_desc @ centers.T), axis=1)[:, :TOP_STATES]
        public_desc = np.stack([global_descriptor(DATA / r["image_path"]) for r in public_rows])
        public_routes = np.argsort(-(public_desc @ centers.T), axis=1)[:, :TOP_STATES]
        train_rel = [str(p.relative_to(DATA)).replace("\\", "/") for p in train_paths]
        state_data[category] = {
            "centers": centers,
            "train_state": dict(zip(train_rel, map(int, train_labels))),
            "routes": {
                "validation": {int(r["map_index"]): list(map(int, route)) for r, route in zip(val_rows, val_routes)},
                "public": {int(r["map_index"]): list(map(int, route)) for r, route in zip(public_rows, public_routes)},
            },
        }
        dest = OUT / category
        dest.mkdir()
        np.savez(dest / "global_state.npz", centers=centers, train_desc=train_desc, val_desc=val_desc, public_desc=public_desc)
        state_meta = {
            "train_counts": dict(Counter(map(int, train_labels))),
            "normal_counts": dict(Counter(map(int, labels))),
            "validation_routes": state_data[category]["routes"]["validation"],
            "public_routes": state_data[category]["routes"]["public"],
        }
        (dest / "state_meta.json").write_text(json.dumps(state_meta, indent=2))
        record = {"phase": "global_states", "category": category, "seconds": time.time() - tick, "train_counts": state_meta["train_counts"]}
        progress.append(record)
        (OUT / "progress.json").write_text(json.dumps(progress, indent=2))
        print(record, flush=True)

    del global_model
    torch.cuda.empty_cache()
    (OUT / "STATES_COMPLETE.json").write_text(json.dumps({"status": "complete", "categories": len(categories)}))

    patch_model = timm.create_model(protocol["model"], pretrained=True, num_classes=0, img_size=640).eval().cuda()

    @torch.inference_mode()
    def patch_backbone(x):
        result = [[] for _ in LAYERS]
        for part in x.split(1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = patch_model.forward_intermediates(
                    part, indices=LAYERS, norm=False, output_fmt="NLC", intermediates_only=True
                )
            for dest, feature in zip(result, out):
                dest.append(feature.float())
        return [torch.cat(v) for v in result]

    @torch.inference_mode()
    def extract_patch(path):
        with Image.open(path) as im:
            x = to_tensor(im.convert("RGB"))[None].cuda()
        shape = (int(x.shape[-2] * 0.625), int(x.shape[-1] * 0.625))
        assert min(shape) >= 640, (path, shape)
        x = F.interpolate(x, size=shape, mode="bicubic", align_corners=False, antialias=True)
        x = (x - mean) / std
        return patcher(x, patch_backbone)

    @torch.inference_mode()
    def score(features, state_memories, chosen_states):
        maps = []
        for layer_index, feature in enumerate(features):
            h, w = feature.shape[1:3]
            query = torch.from_numpy(feature.reshape(-1, 768)).cuda()
            values = []
            for query_chunk in query.split(256):
                best = torch.full((len(query_chunk),), float("inf"), device="cuda")
                for state in chosen_states:
                    memory = state_memories[int(state)][layer_index]
                    for memory_chunk in memory.split(4096):
                        best = torch.minimum(
                            best,
                            torch.cdist(query_chunk, memory_chunk, compute_mode="use_mm_for_euclid_dist").min(1).values,
                        )
                values.append(best)
            maps.append(torch.cat(values).reshape(h, w) / 768)
        result = torch.stack(maps).mean(0).cpu().numpy()
        assert np.isfinite(result).all()
        return result

    for category in categories:
        tick = time.time()
        source_dir = A10 / "patch640" / category
        references = json.loads((source_dir / "references.json").read_text())
        archive = np.load(source_dir / "memory.npz")
        train_state = state_data[category]["train_state"]
        token_states = np.asarray([train_state[ref["path"]] for ref in references], dtype=np.int64)
        state_memories = []
        state_counts = {}
        for state in range(K_STATES):
            indices = np.flatnonzero(token_states == state)
            assert len(indices) > 0, (category, state)
            state_counts[state] = int(len(indices))
            state_memories.append([
                torch.from_numpy(archive[str(layer)][indices].astype(np.float32)).cuda()
                for layer in LAYERS
            ])
        dest = OUT / category / "state_patch640"
        timings = {}
        for split in ("validation", "public"):
            rows = [r for r in meta[split] if r["category"] == category]
            split_dest = dest / split
            split_dest.mkdir(parents=True)
            start = time.time()
            for number, row in enumerate(rows, 1):
                route = state_data[category]["routes"][split][int(row["map_index"])]
                anomaly_map = score(extract_patch(DATA / row["image_path"]), state_memories, route)
                np.save(split_dest / f"{int(row['map_index']):05d}.npy", anomaly_map)
                if number % 30 == 0:
                    print(category, split, number, len(rows), route, flush=True)
            timings[split] = {"images": len(rows), "seconds": time.time() - start}
        record = {
            "phase": "state_patch_maps",
            "category": category,
            "state_memory_counts": state_counts,
            "timings": timings,
            "seconds": time.time() - tick,
        }
        progress.append(record)
        (OUT / "progress.json").write_text(json.dumps(progress, indent=2))
        print(record, flush=True)
        del state_memories
        torch.cuda.empty_cache()

    (OUT / "MAPS_COMPLETE.json").write_text(json.dumps({"status": "complete", "categories": len(categories)}))
    print("A18_MAPS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
