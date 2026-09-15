from pathlib import Path
from collections import defaultdict
import csv
import json
import re

import cv2
import numpy as np
from scipy.stats import rankdata


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A1_ROOT = Path(
    "orbitad/results/a1_anomaly_score_sensitivity"
)

A5_PCA_ROOT = Path(
    "orbitad/results/a5_tangent_specificity"
)

A5_SCORE_ROOT = Path(
    "orbitad/results/a5_orthogonal_scoring"
)

OUT = Path(
    "orbitad/results/a10_dual_evidence"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


VAL_META = (
    A1_ROOT / "validation_patch_maps.csv"
)

PUBLIC_META = (
    A1_ROOT / "public_patch_maps.csv"
)


PATHS = {
    "val_raw":
        A5_SCORE_ROOT
        / "validation_raw_maps.npy",

    "val_pca":
        A5_PCA_ROOT
        / "validation_normalpca16_maps.npy",

    "pub_raw":
        A5_SCORE_ROOT
        / "public_raw_maps.npy",

    "pub_pca":
        A5_PCA_ROOT
        / "public_normalpca16_maps.npy",
}


CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


SHIFTS = [
    "shift_1",
    "shift_2",
    "shift_3",
]


# ============================================================
# Metadata
# ============================================================

def load_meta(path):

    with path.open() as f:

        rows = list(
            csv.DictReader(f)
        )

    for i, r in enumerate(rows):

        if "map_index" in r:
            r["map_index"] = int(
                r["map_index"]
            )
        else:
            r["map_index"] = i

        if "label" in r:
            r["label"] = int(
                r["label"]
            )

    return rows


val_rows = load_meta(
    VAL_META
)

pub_rows = load_meta(
    PUBLIC_META
)


print("=" * 110)
print("A10.0 — DUAL EVIDENCE COMPLEMENTARITY AUDIT")
print("=" * 110)

print(
    "validation rows:",
    len(val_rows)
)

print(
    "public rows    :",
    len(pub_rows)
)


# ============================================================
# Maps
# ============================================================

arrays = {
    k: np.load(
        p,
        mmap_mode="r",
    )
    for k, p in PATHS.items()
}


for k, x in arrays.items():

    print(
        f"{k:10s}",
        x.shape
    )


# ============================================================
# Per-category empirical CDF calibration
# ============================================================

cdf_ref = {
    c: {}
    for c in CATEGORIES
}


for category in CATEGORIES:

    ids = [
        r["map_index"]
        for r in val_rows
        if r["category"] == category
    ]


    raw = np.asarray(
        arrays["val_raw"][ids],
        dtype=np.float32,
    ).reshape(-1)


    pca = np.asarray(
        arrays["val_pca"][ids],
        dtype=np.float32,
    ).reshape(-1)


    cdf_ref[category]["raw"] = np.sort(
        raw.astype(
            np.float64
        )
    )

    cdf_ref[category]["pca"] = np.sort(
        pca.astype(
            np.float64
        )
    )


def cdf(
    category,
    method,
    values,
):

    ref = cdf_ref[
        category
    ][
        method
    ]

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        np.searchsorted(
            ref,
            values,
            side="right",
        )
        /
        len(ref)
    )


# ============================================================
# Helpers
# ============================================================

COND_RE = re.compile(
    r"_(regular|shift_[1-4]|overexposed|underexposed)$"
)


def instance_id(row):

    if row.get(
        "instance_id",
        "",
    ):

        return row[
            "instance_id"
        ]


    stem = Path(
        row[
            "image_path"
        ]
    ).stem

    return COND_RE.sub(
        "",
        stem,
    )


def get_mask_path(row):

    value = row.get(
        "mask_path",
        "",
    )


    if value:

        p = Path(
            value
        )

        if not p.is_absolute():

            p = (
                DATA_ROOT
                /
                p
            )

        if p.exists():

            return p


    image_path = Path(
        row[
            "image_path"
        ]
    )


    candidate = (
        DATA_ROOT
        /
        row[
            "category"
        ]
        /
        "test_public"
        /
        "ground_truth"
        /
        "bad"
        /
        (
            image_path.stem
            +
            "_mask.png"
        )
    )


    if not candidate.exists():

        raise RuntimeError(
            f"GT missing: {candidate}"
        )


    return candidate


def mask32(path):

    m = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )


    if m is None:

        raise RuntimeError(
            f"Cannot read {path}"
        )


    m = (
        m > 0
    ).astype(
        np.float32
    )


    # Fraction of anomaly pixels in each feature-grid cell.
    m = cv2.resize(
        m,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )


    # Any anomalous occupancy => defect patch.
    return (
        m > 0
    ).reshape(-1)


def auc_rank(
    labels,
    scores,
):

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    s = np.asarray(
        scores,
        dtype=np.float64,
    )


    pos = (
        y == 1
    )

    neg = (
        y == 0
    )


    P = int(
        pos.sum()
    )

    N = int(
        neg.sum()
    )


    if (
        P == 0
        or
        N == 0
    ):

        return np.nan


    r = rankdata(
        s,
        method="average",
    )


    return float(
        (
            r[pos].sum()
            -
            P * (P + 1) / 2
        )
        /
        (
            P * N
        )
    )


def top10(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)


    k = min(
        10,
        len(x),
    )


    return float(
        np.partition(
            x,
            -k,
        )[-k:].mean()
    )


# ============================================================
# Defect/background patch discrimination
# ============================================================

patch_records = []


for row in pub_rows:

    if int(
        row.get(
            "label",
            0,
        )
    ) != 1:

        continue


    category = row[
        "category"
    ]

    idx = row[
        "map_index"
    ]


    raw = cdf(
        category,
        "raw",
        arrays[
            "pub_raw"
        ][
            idx
        ],
    ).reshape(-1)


    pca = cdf(
        category,
        "pca",
        arrays[
            "pub_pca"
        ][
            idx
        ],
    ).reshape(-1)


    # Fixed, pre-registered dual evidence.
    dual = np.maximum(
        raw,
        pca,
    )


    labels = mask32(
        get_mask_path(
            row
        )
    ).astype(
        np.int64
    )


    for method, score in [
        ("raw", raw),
        ("pca16", pca),
        ("dual_or", dual),
    ]:

        patch_records.append({
            "category":
                category,

            "condition":
                row[
                    "condition"
                ],

            "image_path":
                row[
                    "image_path"
                ],

            "method":
                method,

            "labels":
                labels,

            "scores":
                score,
        })


# ============================================================
# Per-category AUC
# ============================================================

category_rows = []


for category in CATEGORIES:

    result = {
        "category":
            category,
    }


    for method in [
        "raw",
        "pca16",
        "dual_or",
    ]:

        members = [
            r
            for r in patch_records
            if (
                r["category"]
                ==
                category
                and
                r["method"]
                ==
                method
            )
        ]


        y = np.concatenate([
            r["labels"]
            for r in members
        ])


        s = np.concatenate([
            r["scores"]
            for r in members
        ])


        result[
            f"{method}_auc"
        ] = auc_rank(
            y,
            s,
        )


    result[
        "dual_minus_pca"
    ] = (
        result[
            "dual_or_auc"
        ]
        -
        result[
            "pca16_auc"
        ]
    )


    category_rows.append(
        result
    )


# Macro category AUC, not pooled-pixel AUC.
macro = {
    method:
        float(
            np.mean([
                r[
                    f"{method}_auc"
                ]
                for r in category_rows
            ])
        )
    for method in [
        "raw",
        "pca16",
        "dual_or",
    ]
}


# ============================================================
# Same-instance public-good shift stability
# ============================================================

image_scores = {}


for row in pub_rows:

    if int(
        row.get(
            "label",
            0,
        )
    ) != 0:

        continue


    if row[
        "condition"
    ] not in (
        ["regular"]
        +
        SHIFTS
    ):

        continue


    category = row[
        "category"
    ]

    idx = row[
        "map_index"
    ]

    iid = instance_id(
        row
    )


    raw = cdf(
        category,
        "raw",
        arrays[
            "pub_raw"
        ][
            idx
        ],
    )


    pca = cdf(
        category,
        "pca",
        arrays[
            "pub_pca"
        ][
            idx
        ],
    )


    dual = np.maximum(
        raw,
        pca,
    )


    image_scores[
        (
            category,
            iid,
            row[
                "condition"
            ],
            "raw",
        )
    ] = top10(
        raw
    )


    image_scores[
        (
            category,
            iid,
            row[
                "condition"
            ],
            "pca16",
        )
    ] = top10(
        pca
    )


    image_scores[
        (
            category,
            iid,
            row[
                "condition"
            ],
            "dual_or",
        )
    ] = top10(
        dual
    )


drifts = {
    method: []
    for method in [
        "raw",
        "pca16",
        "dual_or",
    ]
}


category_drifts = {
    c: {
        method: []
        for method in [
            "raw",
            "pca16",
            "dual_or",
        ]
    }
    for c in CATEGORIES
}


for category in CATEGORIES:

    iids = sorted({
        k[1]
        for k in image_scores
        if k[0] == category
    })


    for iid in iids:

        for method in [
            "raw",
            "pca16",
            "dual_or",
        ]:

            reg_key = (
                category,
                iid,
                "regular",
                method,
            )


            if reg_key not in image_scores:
                continue


            reg = image_scores[
                reg_key
            ]


            for condition in SHIFTS:

                key = (
                    category,
                    iid,
                    condition,
                    method,
                )


                if key not in image_scores:
                    continue


                d = abs(
                    image_scores[
                        key
                    ]
                    -
                    reg
                )


                drifts[
                    method
                ].append(
                    d
                )


                category_drifts[
                    category
                ][
                    method
                ].append(
                    d
                )


median_drift = {
    method:
        float(
            np.median(
                drifts[
                    method
                ]
            )
        )
    for method in drifts
}


# ============================================================
# Pre-registered decision
# ============================================================

auc_gain = (
    macro[
        "dual_or"
    ]
    -
    macro[
        "pca16"
    ]
)


drift_ratio = (
    median_drift[
        "dual_or"
    ]
    /
    max(
        median_drift[
            "pca16"
        ],
        1e-12,
    )
)


category_wins = sum(
    r[
        "dual_minus_pca"
    ]
    >
    0
    for r in category_rows
)


worst_loss = min(
    r[
        "dual_minus_pca"
    ]
    for r in category_rows
)


signal_auc = (
    auc_gain
    >=
    0.010
)


signal_drift = (
    drift_ratio
    <=
    1.10
)


signal_categories = (
    category_wins
    >=
    5
)


signal_worst = (
    worst_loss
    >=
    -0.030
)


decision = (
    "GO"
    if (
        signal_auc
        and
        signal_drift
        and
        signal_categories
        and
        signal_worst
    )
    else
    "NO_GO"
)


# ============================================================
# Save
# ============================================================

CATEGORY_CSV = (
    OUT
    /
    "a10_0_category.csv"
)


with CATEGORY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            category_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        category_rows
    )


summary = {
    "protocol": {
        "all_categories":
            True,

        "training":
            False,

        "gpu":
            False,

        "calibration":
            "validation_good_empirical_cdf",

        "dual_rule":
            "max(raw_cdf,pca16_cdf)",

        "public_gt_role":
            "development_audit_only",
    },

    "macro_patch_auc": {
        "raw":
            macro["raw"],

        "pca16":
            macro["pca16"],

        "dual_or":
            macro["dual_or"],

        "dual_minus_pca":
            auc_gain,
    },

    "normal_shift_drift": {
        "raw":
            median_drift["raw"],

        "pca16":
            median_drift["pca16"],

        "dual_or":
            median_drift["dual_or"],

        "dual_over_pca":
            drift_ratio,
    },

    "category_consistency": {
        "dual_auc_wins":
            int(
                category_wins
            ),

        "worst_auc_delta":
            float(
                worst_loss
            ),
    },

    "signals": {
        "macro_auc_gain":
            bool(
                signal_auc
            ),

        "shift_stability":
            bool(
                signal_drift
            ),

        "category_wins":
            bool(
                signal_categories
            ),

        "worst_category":
            bool(
                signal_worst
            ),
    },

    "decision":
        decision,
}


SUMMARY = (
    OUT
    /
    "a10_0_summary.json"
)


with SUMMARY.open(
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Console
# ============================================================

print()
print("=" * 110)
print("A10.0 CATEGORY RESULTS")
print("=" * 110)

print(
    f"{'category':15s}"
    f"{'RAW':>10s}"
    f"{'PCA16':>10s}"
    f"{'DUAL':>10s}"
    f"{'D-PCA':>10s}"
)

print("-" * 55)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['raw_auc']:10.4f}"
        f"{r['pca16_auc']:10.4f}"
        f"{r['dual_or_auc']:10.4f}"
        f"{r['dual_minus_pca']:10.4f}"
    )


print()
print("[MACRO PATCH AUC]")

print(
    "raw       :",
    f"{macro['raw']:.4f}"
)

print(
    "pca16     :",
    f"{macro['pca16']:.4f}"
)

print(
    "dual OR   :",
    f"{macro['dual_or']:.4f}"
)

print(
    "dual-PCA  :",
    f"{auc_gain:+.4f}"
)


print()
print("[NORMAL REAL-SHIFT DRIFT]")

print(
    "raw       :",
    f"{median_drift['raw']:.4f}"
)

print(
    "pca16     :",
    f"{median_drift['pca16']:.4f}"
)

print(
    "dual OR   :",
    f"{median_drift['dual_or']:.4f}"
)

print(
    "dual/PCA  :",
    f"{drift_ratio:.4f}"
)


print()
print("[PRE-REGISTERED A10.0]")

print(
    "macro AUC gain >= +.010 :",
    signal_auc
)

print(
    "shift drift ratio <=1.10:",
    signal_drift
)

print(
    "AUC wins >=5/8          :",
    signal_categories,
    f"({category_wins}/8)"
)

print(
    "worst loss >= -.030     :",
    signal_worst,
    f"({worst_loss:+.4f})"
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "SUMMARY  :",
    SUMMARY
)

print(
    "CATEGORY :",
    CATEGORY_CSV
)

print("=" * 110)
