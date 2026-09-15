from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np


ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

SCORES_CSV = ROOT / "test_scores.csv"

NORMAL_DRIFT_CSV = ROOT / "normal_orbit_score_drift.csv"
DEFECT_DRIFT_CSV = ROOT / "defect_orbit_score_drift.csv"

DRIFT_SUMMARY_CSV = ROOT / "score_drift_condition_summary.csv"

CATEGORY_AUC_CSV = ROOT / "category_condition_auc.csv"
CONDITION_AUC_CSV = ROOT / "condition_auc.csv"

SUMMARY_JSON = ROOT / "a1_3_summary.json"


# ============================================================
# Load
# ============================================================

rows = []

with SCORES_CSV.open() as f:

    for row in csv.DictReader(f):

        row["label"] = int(
            row["label"]
        )

        for key in [
            "image_score_top10",
            "patch_mean",
            "patch_p95",
            "patch_p99",
            "patch_max",
        ]:
            row[key] = float(
                row[key]
            )

        rows.append(row)


if len(rows) != 1084:
    raise RuntimeError(
        f"Expected 1084 rows, got {len(rows)}"
    )


# ============================================================
# Exact small-sample AUROC, including ties
# ============================================================

def binary_auc(pos, neg):

    pos = np.asarray(
        pos,
        dtype=np.float64,
    )

    neg = np.asarray(
        neg,
        dtype=np.float64,
    )

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    comparison = (
        pos[:, None]
        -
        neg[None, :]
    )

    return float(
        (
            np.sum(comparison > 0)
            +
            0.5 * np.sum(comparison == 0)
        )
        /
        comparison.size
    )


def median_or_nan(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if len(x) == 0:
        return np.nan

    return float(
        np.median(x)
    )


def mean_or_nan(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if len(x) == 0:
        return np.nan

    return float(
        np.mean(x)
    )


# ============================================================
# Paired score drift
# ============================================================

groups = defaultdict(list)

for row in rows:

    groups[
        (
            row["set_type"],
            row["category"],
            row["instance_id"],
        )
    ].append(row)


drift_rows = []


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
            f"regular count={len(regular)}"
        )

    ref = regular[0]

    ref_score = ref[
        "image_score_top10"
    ]

    for cur in members:

        if cur["condition"] == "regular":
            continue

        score = cur[
            "image_score_top10"
        ]

        delta = (
            score - ref_score
        )

        ratio = (
            score
            /
            max(
                ref_score,
                1e-8,
            )
        )

        pct = (
            100.0
            * delta
            /
            max(
                abs(ref_score),
                1e-8,
            )
        )

        drift_rows.append({
            "set_type": set_type,
            "category": category,
            "instance_id": instance_id,
            "condition": cur["condition"],
            "regular_score": ref_score,
            "condition_score": score,
            "score_delta": delta,
            "score_ratio": ratio,
            "score_change_pct": pct,
            "score_increased":
                score > ref_score,
            "score_increased_ge_30pct":
                ratio >= 1.30,
        })


normal_drift = [
    x for x in drift_rows
    if x["set_type"] == "normal"
]

defect_drift = [
    x for x in drift_rows
    if x["set_type"] == "defect"
]


for path, data in [
    (NORMAL_DRIFT_CSV, normal_drift),
    (DEFECT_DRIFT_CSV, defect_drift),
]:

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                data[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(data)


# ============================================================
# Drift condition summary
# ============================================================

drift_groups = defaultdict(list)

for row in drift_rows:

    drift_groups[
        (
            row["set_type"],
            row["condition"],
        )
    ].append(row)


drift_summary_rows = []


for (
    set_type,
    condition,
), members in sorted(
    drift_groups.items()
):

    ratios = [
        float(x["score_ratio"])
        for x in members
    ]

    deltas = [
        float(x["score_delta"])
        for x in members
    ]

    pcts = [
        float(x["score_change_pct"])
        for x in members
    ]

    frac_up = np.mean([
        bool(x["score_increased"])
        for x in members
    ])

    frac_30 = np.mean([
        bool(
            x[
                "score_increased_ge_30pct"
            ]
        )
        for x in members
    ])

    drift_summary_rows.append({
        "set_type": set_type,
        "condition": condition,
        "n_pairs": len(members),

        "median_score_ratio":
            median_or_nan(ratios),

        "median_score_change_pct":
            median_or_nan(pcts),

        "median_score_delta":
            median_or_nan(deltas),

        "mean_score_delta":
            mean_or_nan(deltas),

        "fraction_score_increased":
            float(frac_up),

        "fraction_score_increased_ge_30pct":
            float(frac_30),
    })


with DRIFT_SUMMARY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            drift_summary_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        drift_summary_rows
    )


# ============================================================
# Category × condition AUROC
# ============================================================

category_condition_groups = defaultdict(list)

for row in rows:

    category_condition_groups[
        (
            row["category"],
            row["condition"],
        )
    ].append(row)


category_auc_rows = []


for (
    category,
    condition,
), members in sorted(
    category_condition_groups.items()
):

    pos = [
        x["image_score_top10"]
        for x in members
        if x["label"] == 1
    ]

    neg = [
        x["image_score_top10"]
        for x in members
        if x["label"] == 0
    ]

    if not pos or not neg:
        continue

    auc = binary_auc(
        pos,
        neg,
    )

    category_auc_rows.append({
        "category": category,
        "condition": condition,
        "n_normal": len(neg),
        "n_defect": len(pos),
        "auc": auc,
    })


# regular reference per category
regular_auc = {
    row["category"]: row["auc"]
    for row in category_auc_rows
    if row["condition"] == "regular"
}


for row in category_auc_rows:

    ref = regular_auc.get(
        row["category"],
        np.nan,
    )

    row["regular_auc"] = ref

    row["delta_auc_vs_regular"] = (
        row["auc"] - ref
        if np.isfinite(ref)
        else np.nan
    )


with CATEGORY_AUC_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            category_auc_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        category_auc_rows
    )


# ============================================================
# Condition macro AUROC
# ============================================================

condition_to_category_auc = defaultdict(list)

for row in category_auc_rows:

    condition_to_category_auc[
        row["condition"]
    ].append(row)


condition_auc_rows = []


for condition, members in sorted(
    condition_to_category_auc.items()
):

    aucs = [
        x["auc"]
        for x in members
    ]

    deltas = [
        x["delta_auc_vs_regular"]
        for x in members
        if np.isfinite(
            x["delta_auc_vs_regular"]
        )
    ]

    # pooled AUROC for descriptive use only
    member_categories = {
        x["category"]
        for x in members
    }

    pooled_rows = [
        x
        for x in rows
        if (
            x["condition"] == condition
            and
            x["category"] in member_categories
        )
    ]

    pooled_pos = [
        x["image_score_top10"]
        for x in pooled_rows
        if x["label"] == 1
    ]

    pooled_neg = [
        x["image_score_top10"]
        for x in pooled_rows
        if x["label"] == 0
    ]

    pooled_auc = binary_auc(
        pooled_pos,
        pooled_neg,
    )

    condition_auc_rows.append({
        "condition": condition,
        "categories": len(members),

        "n_normal":
            sum(x["n_normal"] for x in members),

        "n_defect":
            sum(x["n_defect"] for x in members),

        "macro_auc":
            mean_or_nan(aucs),

        "median_auc":
            median_or_nan(aucs),

        "pooled_auc":
            pooled_auc,

        "macro_delta_auc_vs_regular":
            mean_or_nan(deltas),
    })


with CONDITION_AUC_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_auc_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        condition_auc_rows
    )


# ============================================================
# Go / No-Go diagnostics
# ============================================================

normal_shift_rows = [
    x for x in drift_summary_rows
    if (
        x["set_type"] == "normal"
        and x["condition"].startswith("shift_")
    )
]

auc_shift_rows = [
    x for x in condition_auc_rows
    if x["condition"].startswith("shift_")
]


max_normal_median_pct = (
    max(
        x["median_score_change_pct"]
        for x in normal_shift_rows
    )
    if normal_shift_rows
    else np.nan
)

worst_auc_drop = (
    min(
        x["macro_delta_auc_vs_regular"]
        for x in auc_shift_rows
    )
    if auc_shift_rows
    else np.nan
)


# Main pre-registered decision signals:
# 1) median normal score increase >= 30% for any spatial shift
# 2) macro AUROC drop >= 0.03 for any spatial shift

score_signal = (
    np.isfinite(
        max_normal_median_pct
    )
    and
    max_normal_median_pct >= 30.0
)

auc_signal = (
    np.isfinite(
        worst_auc_drop
    )
    and
    worst_auc_drop <= -0.03
)

go = bool(
    score_signal
    or auc_signal
)


payload = {
    "num_test_images": len(rows),

    "decision_thresholds": {
        "normal_median_score_increase_pct": 30.0,
        "macro_auc_drop": 0.03,
    },

    "observed": {
        "max_shift_normal_median_score_change_pct":
            float(max_normal_median_pct),

        "worst_shift_macro_delta_auc_vs_regular":
            float(worst_auc_drop),
    },

    "signals": {
        "score_signal": bool(score_signal),
        "auc_signal": bool(auc_signal),
    },

    "decision":
        "GO"
        if go
        else "NO_GO_PENDING_PIXEL_CHECK",
}


with SUMMARY_JSON.open("w") as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


# ============================================================
# Console
# ============================================================

print("=" * 105)
print("OrbitAD A1.3 — ACTUAL ANOMALY-SCORE SENSITIVITY")
print("=" * 105)

print()
print("[NORMAL PAIRED SCORE DRIFT]")

print(
    f"{'condition':16s}"
    f"{'N':>6s}"
    f"{'ratio-med':>12s}"
    f"{'change%':>12s}"
    f"{'P(up)':>10s}"
    f"{'P(+30%)':>10s}"
)

print("-" * 66)

for row in drift_summary_rows:

    if row["set_type"] != "normal":
        continue

    print(
        f"{row['condition']:16s}"
        f"{row['n_pairs']:6d}"
        f"{row['median_score_ratio']:12.4f}"
        f"{row['median_score_change_pct']:12.2f}"
        f"{row['fraction_score_increased']:10.4f}"
        f"{row['fraction_score_increased_ge_30pct']:10.4f}"
    )


print()
print("[DEFECT PAIRED SCORE DRIFT]")

print(
    f"{'condition':16s}"
    f"{'N':>6s}"
    f"{'ratio-med':>12s}"
    f"{'change%':>12s}"
    f"{'P(up)':>10s}"
)

print("-" * 56)

for row in drift_summary_rows:

    if row["set_type"] != "defect":
        continue

    print(
        f"{row['condition']:16s}"
        f"{row['n_pairs']:6d}"
        f"{row['median_score_ratio']:12.4f}"
        f"{row['median_score_change_pct']:12.2f}"
        f"{row['fraction_score_increased']:10.4f}"
    )


print()
print("[CONDITION IMAGE AUROC]")

print(
    f"{'condition':16s}"
    f"{'cats':>6s}"
    f"{'normal':>8s}"
    f"{'defect':>8s}"
    f"{'macro-AUC':>12s}"
    f"{'delta':>10s}"
)

print("-" * 66)

for row in condition_auc_rows:

    print(
        f"{row['condition']:16s}"
        f"{row['categories']:6d}"
        f"{row['n_normal']:8d}"
        f"{row['n_defect']:8d}"
        f"{row['macro_auc']:12.4f}"
        f"{row['macro_delta_auc_vs_regular']:10.4f}"
    )


print()
print("[PRE-REGISTERED GO / NO-GO]")

print(
    "max shift normal median score change : "
    f"{max_normal_median_pct:.2f}%"
)

print(
    "worst shift macro AUROC delta         : "
    f"{worst_auc_drop:.4f}"
)

print(
    "score signal                          :",
    score_signal,
)

print(
    "AUC signal                            :",
    auc_signal,
)

print()
print(
    "DECISION :",
    payload["decision"]
)

print()
print("NORMAL DRIFT :", NORMAL_DRIFT_CSV)
print("DEFECT DRIFT :", DEFECT_DRIFT_CSV)
print("DRIFT SUMMARY:", DRIFT_SUMMARY_CSV)
print("CATEGORY AUC :", CATEGORY_AUC_CSV)
print("CONDITION AUC:", CONDITION_AUC_CSV)
print("SUMMARY      :", SUMMARY_JSON)

print("=" * 105)
