from pathlib import Path
import csv
import json
from collections import defaultdict

import numpy as np


ROOT = Path(
    "orbitad/results/a1_feature_sensitivity"
)

FEATURES_NPY = ROOT / "dinov3b_public_features.npy"
META_CSV = ROOT / "dinov3b_public_features.csv"

PAIR_CSV = ROOT / "orbit_feature_drift.csv"
COND_CSV = ROOT / "orbit_condition_summary.csv"
CAT_CSV = ROOT / "orbit_category_summary.csv"
SUMMARY_JSON = ROOT / "orbit_feature_summary.json"


features = np.load(FEATURES_NPY)

rows = []

with META_CSV.open() as f:
    reader = csv.DictReader(f)

    for row in reader:
        row["feature_index"] = int(
            row["feature_index"]
        )
        rows.append(row)


if len(rows) != len(features):
    raise RuntimeError(
        "Metadata / feature count mismatch."
    )


def cosine_distance(a, b):
    return float(
        1.0 - np.dot(a, b)
    )


# ------------------------------------------------------------
# Build indexes
# ------------------------------------------------------------

groups = defaultdict(list)

regular_index = defaultdict(list)
condition_index = defaultdict(list)


for row in rows:

    key = (
        row["set_type"],
        row["category"],
        row["instance_id"],
    )

    groups[key].append(row)

    condition_index[
        (
            row["set_type"],
            row["category"],
            row["condition"],
        )
    ].append(row)

    if row["condition"] == "regular":

        regular_index[
            (
                row["set_type"],
                row["category"],
            )
        ].append(row)


# ------------------------------------------------------------
# Pair audit
# ------------------------------------------------------------

pair_rows = []


for (
    set_type,
    category,
    instance_id,
), members in sorted(groups.items()):

    regular = [
        x for x in members
        if x["condition"] == "regular"
    ]

    if len(regular) != 1:
        raise RuntimeError(
            f"{set_type}/{category}/{instance_id}: "
            f"expected exactly one regular image, "
            f"found {len(regular)}"
        )

    reg_row = regular[0]

    reg_feat = features[
        reg_row["feature_index"]
    ]

    # --------------------------------------------------------
    # Identity reference:
    # regular instance vs other regular instances
    # in SAME set_type + category.
    # --------------------------------------------------------

    other_regular = [
        x
        for x in regular_index[
            (set_type, category)
        ]
        if x["instance_id"] != instance_id
    ]

    other_regular_distances = [
        cosine_distance(
            reg_feat,
            features[x["feature_index"]]
        )
        for x in other_regular
    ]

    d_id_regular = (
        float(
            np.median(
                other_regular_distances
            )
        )
        if other_regular_distances
        else np.nan
    )

    # --------------------------------------------------------
    # Acquisition variants
    # --------------------------------------------------------

    for current in members:

        condition = current["condition"]

        if condition == "regular":
            continue

        cur_feat = features[
            current["feature_index"]
        ]

        d_acq = cosine_distance(
            reg_feat,
            cur_feat
        )

        # Same-condition different-instance distance
        others_same_condition = [
            x
            for x in condition_index[
                (
                    set_type,
                    category,
                    condition,
                )
            ]
            if x["instance_id"] != instance_id
        ]

        same_cond_distances = [
            cosine_distance(
                cur_feat,
                features[x["feature_index"]]
            )
            for x in others_same_condition
        ]

        d_id_condition = (
            float(
                np.median(
                    same_cond_distances
                )
            )
            if same_cond_distances
            else np.nan
        )

        acr_regular = (
            d_acq / d_id_regular
            if np.isfinite(d_id_regular)
            and d_id_regular > 0
            else np.nan
        )

        acr_condition = (
            d_acq / d_id_condition
            if np.isfinite(d_id_condition)
            and d_id_condition > 0
            else np.nan
        )

        pair_rows.append({
            "set_type": set_type,
            "category": category,
            "instance_id": instance_id,
            "condition": condition,

            "d_acquisition":
                d_acq,

            "d_identity_regular":
                d_id_regular,

            "d_identity_same_condition":
                d_id_condition,

            "acr_regular":
                acr_regular,

            "acr_same_condition":
                acr_condition,

            "identity_margin":
                (
                    d_id_regular - d_acq
                    if np.isfinite(
                        d_id_regular
                    )
                    else np.nan
                ),

            "acquisition_exceeds_identity":
                (
                    acr_regular >= 1.0
                    if np.isfinite(
                        acr_regular
                    )
                    else False
                ),
        })


# ------------------------------------------------------------
# Write pair-level CSV
# ------------------------------------------------------------

with PAIR_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            pair_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(pair_rows)


# ------------------------------------------------------------
# Summary helpers
# ------------------------------------------------------------

def summarize(values):

    x = np.asarray(
        [
            float(v)
            for v in values
            if np.isfinite(float(v))
        ],
        dtype=np.float64
    )

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(
            np.percentile(x, 90)
        ),
        "max": float(np.max(x)),
    }


# ------------------------------------------------------------
# Condition summary
# ------------------------------------------------------------

condition_groups = defaultdict(list)

for r in pair_rows:
    condition_groups[
        (
            r["set_type"],
            r["condition"],
        )
    ].append(r)


condition_rows = []

for (
    set_type,
    condition,
), members in sorted(
    condition_groups.items()
):

    drift = summarize(
        [
            r["d_acquisition"]
            for r in members
        ]
    )

    acr = summarize(
        [
            r["acr_regular"]
            for r in members
        ]
    )

    collision = np.mean(
        [
            r[
                "acquisition_exceeds_identity"
            ]
            for r in members
        ]
    )

    condition_rows.append({
        "set_type": set_type,
        "condition": condition,
        "n": drift["n"],

        "fod_mean":
            drift["mean"],

        "fod_median":
            drift["median"],

        "fod_p90":
            drift["p90"],

        "acr_mean":
            acr["mean"],

        "acr_median":
            acr["median"],

        "acr_p90":
            acr["p90"],

        "fraction_acq_ge_identity":
            float(collision),
    })


with COND_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(condition_rows)


# ------------------------------------------------------------
# Category summary
# ------------------------------------------------------------

category_groups = defaultdict(list)

for r in pair_rows:
    category_groups[
        (
            r["set_type"],
            r["category"],
        )
    ].append(r)


category_rows = []

for (
    set_type,
    category,
), members in sorted(
    category_groups.items()
):

    drift = summarize(
        [
            r["d_acquisition"]
            for r in members
        ]
    )

    acr = summarize(
        [
            r["acr_regular"]
            for r in members
        ]
    )

    category_rows.append({
        "set_type": set_type,
        "category": category,
        "n": drift["n"],
        "fod_mean": drift["mean"],
        "fod_median": drift["median"],
        "fod_p90": drift["p90"],
        "acr_mean": acr["mean"],
        "acr_median": acr["median"],
        "acr_p90": acr["p90"],
    })


with CAT_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            category_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(category_rows)


# ------------------------------------------------------------
# Global summaries
# ------------------------------------------------------------

payload = {}

for set_type in [
    "normal",
    "defect",
]:

    members = [
        r for r in pair_rows
        if r["set_type"] == set_type
    ]

    drift = summarize(
        [
            r["d_acquisition"]
            for r in members
        ]
    )

    acr = summarize(
        [
            r["acr_regular"]
            for r in members
        ]
    )

    collision = float(
        np.mean(
            [
                r[
                    "acquisition_exceeds_identity"
                ]
                for r in members
            ]
        )
    )

    payload[set_type] = {
        "pair_count":
            len(members),

        "feature_orbit_dispersion":
            drift,

        "acquisition_confusion_ratio":
            acr,

        "fraction_acquisition_ge_identity":
            collision,
    }


with SUMMARY_JSON.open("w") as f:
    json.dump(
        payload,
        f,
        indent=2
    )


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("=" * 105)
print("OrbitAD A1.1 — REAL ACQUISITION FEATURE SENSITIVITY")
print("=" * 105)


for set_type in [
    "normal",
    "defect",
]:

    x = payload[set_type]

    print()
    print(
        f"[{set_type.upper()} ORBIT]"
    )

    print(
        "  acquisition pairs        :",
        x["pair_count"]
    )

    fod = x[
        "feature_orbit_dispersion"
    ]

    acr = x[
        "acquisition_confusion_ratio"
    ]

    print(
        f"  FOD mean                 : "
        f"{fod['mean']:.8f}"
    )

    print(
        f"  FOD median               : "
        f"{fod['median']:.8f}"
    )

    print(
        f"  FOD p90                  : "
        f"{fod['p90']:.8f}"
    )

    print(
        f"  ACR mean                 : "
        f"{acr['mean']:.8f}"
    )

    print(
        f"  ACR median               : "
        f"{acr['median']:.8f}"
    )

    print(
        f"  ACR p90                  : "
        f"{acr['p90']:.8f}"
    )

    print(
        f"  P(acq drift >= identity) : "
        f"{x['fraction_acquisition_ge_identity']:.4f}"
    )


print()
print("[CONDITION-WISE]")

print(
    f"{'set':8s} "
    f"{'condition':16s} "
    f"{'N':>5s} "
    f"{'FOD-med':>10s} "
    f"{'ACR-med':>10s} "
    f"{'P>=ID':>9s}"
)

print("-" * 72)

for r in condition_rows:

    print(
        f"{r['set_type']:8s} "
        f"{r['condition']:16s} "
        f"{r['n']:5d} "
        f"{r['fod_median']:10.6f} "
        f"{r['acr_median']:10.6f} "
        f"{r['fraction_acq_ge_identity']:9.4f}"
    )


print()
print("PAIR CSV      :", PAIR_CSV)
print("CONDITION CSV :", COND_CSV)
print("CATEGORY CSV  :", CAT_CSV)
print("SUMMARY JSON  :", SUMMARY_JSON)
print("STATUS        : PASS")
print("=" * 105)
