"""Select honest success/failure cases and render A16 vs final-method panels."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score


ROOT = Path("/root/private_data/iad-vlm-anomaly")
DATA = ROOT / "datasets/MVTec_AD_2"
A10 = ROOT / "orbitad/results/a10_multilayer_v1"
A16 = ROOT / "orbitad/results/a16_normal_only_component_soft_router_v1"
A21 = ROOT / "orbitad/results/a21_layerwise_evidence_v1"
A28 = ROOT / "orbitad/results/a28_depth_resolution_factorial_v1"
OUT = ROOT / "orbitad/results/a46_qualitative_final_method_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(A10 / "public_meta.csv")) if int(r["label"]) == 1]
    a16_scales = json.loads((A16 / "frozen_scales.json").read_text())
    thresholds = json.loads((A16 / "route_thresholds.json").read_text())
    layer_scales = json.loads((A28 / "frozen_layer_scales.json").read_text())

    def load_map(row, root, branch, split="public", depth=None):
        idx, cat = int(row["map_index"]), row["category"]
        if depth is None:
            p = root / branch / cat / split / f"{idx:05d}.npy"
            scale = a16_scales[cat][branch]
        else:
            p = root / branch / f"layer{depth}" / cat / split / f"{idx:05d}.npy"
            scale = layer_scales[cat][branch][str(depth)]
        return np.asarray(np.load(p), np.float32) / scale

    def alpha(value, threshold):
        binary = (value > threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        out = np.zeros_like(value, np.float32)
        for region in range(1, count):
            ratio = stats[region, cv2.CC_STAT_AREA] / binary.size
            out[labels == region] = np.exp(-ratio / .001)
        return out

    def maps(row, size):
        cat = row["category"]
        full = load_map(row, A10, "full512")
        patch = load_map(row, A10, "patch640")
        full_n = cv2.resize(full, size, interpolation=cv2.INTER_LINEAR)
        patch_n = cv2.resize(patch, size, interpolation=cv2.INTER_LINEAR)
        gate = cv2.resize(alpha(patch, thresholds[cat]), size, interpolation=cv2.INTER_LINEAR)
        a16 = .5 * (full_n + patch_n)
        a16 = a16 + gate * (patch_n - a16)
        l8p = load_map(row, A21, "patch640", depth=8)
        l8f = load_map(row, A21, "full512", depth=8)
        l8 = .5 * (l8p + cv2.resize(l8f, (l8p.shape[1], l8p.shape[0]), interpolation=cv2.INTER_LINEAR))
        l8 = cv2.resize(l8, size, interpolation=cv2.INTER_LINEAR)
        return a16, .5 * (a16 + l8)

    ranked = defaultdict(list)
    for n, row in enumerate(rows, 1):
        image_path, mask_path = DATA / row["image_path"], DATA / row["mask_path"]
        with Image.open(image_path) as im:
            size = im.size
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        base, final = maps(row, size)
        target = (256, max(1, round(256 * size[1] / size[0]))) if size[0] >= size[1] else (max(1, round(256 * size[0] / size[1])), 256)
        y = cv2.resize(mask.astype(np.uint8), target, interpolation=cv2.INTER_NEAREST).reshape(-1)
        b = cv2.resize(base, target, interpolation=cv2.INTER_AREA).reshape(-1)
        f = cv2.resize(final, target, interpolation=cv2.INTER_AREA).reshape(-1)
        ap_b, ap_f = average_precision_score(y, b), average_precision_score(y, f)
        item = {**row, "ap_a16": float(ap_b), "ap_final": float(ap_f), "delta_ap": float(ap_f-ap_b)}
        ranked[row["category"]].append(item)
        if n % 50 == 0:
            print("ranked", n, len(rows), flush=True)

    chosen = []
    for cat, items in sorted(ranked.items()):
        ordered = sorted(items, key=lambda x: x["delta_ap"])
        chosen.extend([{**ordered[-1], "case": "best"}, {**ordered[0], "case": "worst"}])
    fields = ["case", "category", "condition", "image_path", "mask_path", "map_index", "ap_a16", "ap_final", "delta_ap"]
    with (OUT / "selected_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(chosen)

    for case in ("best", "worst"):
        selected = [x for x in chosen if x["case"] == case]
        fig, axes = plt.subplots(len(selected), 5, figsize=(15, 3*len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            image = cv2.cvtColor(cv2.imread(str(DATA/item["image_path"])), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(DATA/item["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
            base, final = maps(item, (image.shape[1], image.shape[0]))
            def norm(x):
                lo, hi = np.quantile(x, [.01, .995]); return np.clip((x-lo)/max(hi-lo, 1e-12), 0, 1)
            panels = [image, mask, norm(base), norm(final), norm(final)-norm(base)]
            cmaps = [None, "gray", "turbo", "turbo", "coolwarm"]
            titles = [item["category"], "GT", f"A16 AP={item['ap_a16']:.3f}", f"Final AP={item['ap_final']:.3f}", f"Delta={item['delta_ap']:+.3f}"]
            for ax, data, cmap, title in zip(row_axes, panels, cmaps, titles):
                ax.imshow(data, cmap=cmap, vmin=(-1 if title.startswith("Delta") else None), vmax=(1 if title.startswith("Delta") else None))
                ax.set_title(title); ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"{case}_cases.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    (OUT / "A46_QUALITATIVE_COMPLETE.json").write_text(json.dumps({"status":"complete", "cases":len(chosen)}, indent=2))


if __name__ == "__main__":
    main()
