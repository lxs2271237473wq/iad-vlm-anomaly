#!/usr/bin/env python3
"""Select 448/672 query resolution using normal validation maps only.

The selector treats an excessive increase in the upper anomaly-score tail at
672 as evidence that higher resolution amplifies normal texture.  The cutoff
is a robust across-category upper fence, so no anomalous image or mask is read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile


def paired_maps(root: Path, category: str):
    folder = root / category / "component_maps" / "seed=0" / category / "good"
    for global_path in sorted(folder.glob("*_global.tiff")):
        tiled_path = global_path.with_name(global_path.name.replace("_global.tiff", "_tiled.tiff"))
        if not tiled_path.exists():
            raise FileNotFoundError(tiled_path)
        yield global_path, tiled_path


def image_features(global_map: np.ndarray, tiled_map: np.ndarray) -> dict[str, float]:
    g = np.asarray(global_map, dtype=np.float32).reshape(-1)
    t = np.asarray(tiled_map, dtype=np.float32).reshape(-1)
    eps = 1e-12
    step = max(1, g.size // 20_000)
    gs, ts = g[::step], t[::step]
    corr = float(np.corrcoef(gs, ts)[0, 1]) if np.std(gs) > 0 and np.std(ts) > 0 else 0.0
    result = {"corr": corr}
    for q_name, q in (("q95", .95), ("q99", .99), ("q999", .999)):
        gq, tq = float(np.quantile(g, q)), float(np.quantile(t, q))
        result[f"global_{q_name}"] = gq
        result[f"tiled_{q_name}"] = tq
        result[f"log_{q_name}_ratio"] = float(np.log((tq + eps) / (gq + eps)))
    result["global_mean"] = float(g.mean())
    result["tiled_mean"] = float(t.mean())
    result["positive_excess_over_global_q99"] = float(np.maximum(t - g, 0).mean() / (result["global_q99"] + eps))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted(p.name for p in args.maps_root.iterdir() if p.is_dir())
    image_rows: list[dict] = []
    category_rows: list[dict] = []
    for category in categories:
        rows = []
        for global_path, tiled_path in paired_maps(args.maps_root, category):
            features = image_features(tifffile.imread(global_path), tifffile.imread(tiled_path))
            row = {"category": category, "image": global_path.name.replace("_global.tiff", "")}
            row.update(features)
            rows.append(row)
            image_rows.append(row)
        if not rows:
            raise RuntimeError(f"No paired normal maps for {category}")
        category_rows.append({
            "category": category,
            "n_normal": len(rows),
            **{f"median_{key}": float(np.median([r[key] for r in rows]))
               for key in ("log_q95_ratio", "log_q99_ratio", "log_q999_ratio", "corr", "positive_excess_over_global_q99")},
            "q90_log_q99_ratio": float(np.quantile([r["log_q99_ratio"] for r in rows], .9)),
        })

    # Pre-declared robust unsupervised rule: upper Tukey fence of category-level
    # median q99 tail inflation. It only ranks behavior on normal validation data.
    risks = np.asarray([r["median_log_q99_ratio"] for r in category_rows])
    q1, q3 = np.quantile(risks, [.25, .75])
    cutoff = float(q3 + args.iqr_multiplier * (q3 - q1))
    for row in category_rows:
        row["normal_tail_cutoff"] = cutoff
        row["selected_resolution"] = 448 if row["median_log_q99_ratio"] > cutoff else 672
        row["route_reason"] = "normal-tail outlier" if row["selected_resolution"] == 448 else "normal-tail stable"

    def write_csv(path: Path, rows: list[dict]):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "normal_image_statistics.csv", image_rows)
    write_csv(args.output_dir / "normal_category_statistics_and_route.csv", category_rows)
    summary = {
        "status": "complete",
        "selection_data": "normal validation images only",
        "anomaly_images_or_masks_read": False,
        "criterion": "median log(q99_672/q99_448) <= Q3 + 1.5*IQR across categories",
        "cutoff": cutoff,
        "routes": {r["category"]: r["selected_resolution"] for r in category_rows},
        "category_statistics": category_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "A68_NORMAL_ONLY_ROUTER_COMPLETE.json").write_text(
        json.dumps({"status": "complete", "routes": summary["routes"]}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
