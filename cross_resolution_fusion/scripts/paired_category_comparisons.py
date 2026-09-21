#!/usr/bin/env python3
"""Compute paired category bootstrap comparisons from an existing metric CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--bootstraps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.metrics.open(newline="")))
    metrics = args.metric or ["aupro_0_05", "pixel_auroc", "image_auroc"]
    categories = sorted({row["category"] for row in rows})
    table = {(row["category"], row["method"]): row for row in rows}
    rng = np.random.default_rng(args.seed)
    comparisons = {}
    for baseline in args.baseline:
        key = f"{args.candidate}_vs_{baseline}"
        comparisons[key] = {}
        for metric in metrics:
            delta = np.asarray([
                float(table[category, args.candidate][metric])
                - float(table[category, baseline][metric])
                for category in categories
            ])
            draws = rng.choice(delta, size=(args.bootstraps, len(delta)), replace=True).mean(1)
            comparisons[key][metric] = {
                "mean_delta": float(delta.mean()),
                "paired_category_bootstrap_95ci": [
                    float(np.quantile(draws, 0.025)),
                    float(np.quantile(draws, 0.975)),
                ],
                "wins": int((delta > 1e-12).sum()),
                "losses": int((delta < -1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
                "by_category": dict(zip(categories, map(float, delta))),
            }

    payload = {
        "source": str(args.metrics),
        "candidate": args.candidate,
        "bootstrap_unit": "category",
        "bootstrap_repetitions": args.bootstraps,
        "seed": args.seed,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
