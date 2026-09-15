from pathlib import Path
import csv
import json

import cv2
import numpy as np
from scipy.stats import rankdata


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A3_ROOT = Path(
    "orbitad/results/a3_resolution_audit"
)

OUT_ROOT = Path(
    "orbitad/results/a9_resolution_response"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


META_PATH = (
    A3_ROOT / "regular_public_meta.csv"
)

MAP_PATHS = {
    256:
        A3_ROOT / "regular_maps_r256.npy",

    512:
        A3_ROOT / "regular_maps_r512.npy",

    1024:
        A3_ROOT / "regular_maps_r1024.npy",
}


DETAIL_CSV = (
    OUT_ROOT / "a9_0a_image_response.csv"
)

CATEGORY_CSV = (
    OUT_ROOT / "a9_0a_category_response.csv"
)

SIZE_CSV = (
    OUT_ROOT / "a9_0a_size_strata.csv"
)

SUMMARY_JSON = (
    OUT_ROOT / "a9_0a_summary.json"
)


# ============================================================
# Load metadata
# ============================================================

with META_PATH.open() as f:

    rows = list(
        csv.DictReader(f)
    )


print("=" * 112)
print("A9.0a — RESOLUTION RESPONSE FIELD AUDIT")
print("=" * 112)

print(
    "metadata rows:",
    len(rows)
)

print(
    "metadata fields:",
    list(
        rows[0].keys()
    )
)


if len(rows) != 184:

    raise RuntimeError(
        f"Expected 184 rows, "
        f"got {len(rows)}"
    )


# ============================================================
# Load maps
# ============================================================

maps = {}

expected_shapes = {
    256: (184, 16, 16),
    512: (184, 32, 32),
    1024: (184, 64, 64),
}


for resolution, path in (
    MAP_PATHS.items()
):

    x = np.load(
        path,
        mmap_mode="r",
    )


    if x.shape != expected_shapes[
        resolution
    ]:

        raise RuntimeError(
            f"{resolution}: "
            f"{x.shape} != "
            f"{expected_shapes[resolution]}"
        )


    if not np.isfinite(
        x
    ).all():

        raise RuntimeError(
            f"non-finite map: "
            f"{path}"
        )


    maps[
        resolution
    ] = x


    print(
        f"r{resolution}: "
        f"{x.shape}"
    )


# ============================================================
# Metadata helpers
# ============================================================

def first_existing(
    row,
    names,
):

    for name in names:

        if (
            name in row
            and
            row[name] not in (
                "",
                None,
            )
        ):

            return row[name]


    return None


def get_category(
    row
):

    value = first_existing(
        row,
        [
            "category",
            "class",
            "object",
        ],
    )


    if value is None:

        raise RuntimeError(
            "Cannot identify category "
            f"from fields {list(row.keys())}"
        )


    return value


def get_image_path(
    row
):

    value = first_existing(
        row,
        [
            "image_path",
            "image",
            "path",
            "img_path",
        ],
    )


    return value


def get_mask_path(
    row
):

    value = first_existing(
        row,
        [
            "mask_path",
            "ground_truth_path",
            "gt_path",
            "mask",
            "gt",
        ],
    )


    if value is not None:

        path = Path(
            value
        )

        if not path.is_absolute():

            path = (
                DATA_ROOT
                /
                path
            )


        if path.exists():

            return path


    # --------------------------------------------------------
    # Fallback:
    #
    # category/test_public/bad/004_regular.png
    #
    # ->
    #
    # category/test_public/ground_truth/bad/
    # 004_regular_mask.png
    # --------------------------------------------------------

    image_value = get_image_path(
        row
    )


    if image_value is None:

        raise RuntimeError(
            "Neither mask_path nor "
            "image_path available."
        )


    image_path = Path(
        image_value
    )


    parts = list(
        image_path.parts
    )


    try:

        test_idx = parts.index(
            "test_public"
        )

    except ValueError:

        raise RuntimeError(
            "Cannot derive GT path from: "
            f"{image_value}"
        )


    category = get_category(
        row
    )


    stem = (
        image_path.stem
    )


    filename = (
        stem
        +
        "_mask.png"
    )


    candidate = (
        DATA_ROOT
        /
        category
        /
        "test_public"
        /
        "ground_truth"
        /
        "bad"
        /
        filename
    )


    if not candidate.exists():

        raise RuntimeError(
            "Derived GT does not exist: "
            f"{candidate}"
        )


    return candidate


# ============================================================
# Rank normalization
# ============================================================

def percentile_map(
    score
):

    x = np.asarray(
        score,
        dtype=np.float64,
    )


    flat = x.reshape(
        -1
    )


    ranks = rankdata(
        flat,
        method="average",
    )


    if len(
        flat
    ) <= 1:

        return np.zeros_like(
            x,
            dtype=np.float32,
        )


    ranks = (
        ranks - 1.0
    ) / (
        len(flat) - 1.0
    )


    return (
        ranks
        .reshape(
            x.shape
        )
        .astype(
            np.float32
        )
    )


def contrast(
    rank_map,
    mask,
):

    # Resize rank map to the GT coordinate system.
    response = cv2.resize(
        rank_map,
        (
            mask.shape[1],
            mask.shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )


    defect = (
        mask > 0
    )

    background = ~defect


    if (
        defect.sum() == 0
        or
        background.sum() == 0
    ):

        return np.nan


    defect_mean = float(
        response[
            defect
        ].mean()
    )


    background_mean = float(
        response[
            background
        ].mean()
    )


    return (
        defect_mean
        -
        background_mean
    )


# ============================================================
# Image-level response audit
# ============================================================

detail = []


for i, row in enumerate(
    rows
):

    category = get_category(
        row
    )


    mask_path = get_mask_path(
        row
    )


    mask = cv2.imread(
        str(
            mask_path
        ),
        cv2.IMREAD_GRAYSCALE,
    )


    if mask is None:

        raise RuntimeError(
            f"Cannot read: "
            f"{mask_path}"
        )


    mask = (
        mask > 0
    ).astype(
        np.uint8
    )


    defect_fraction = float(
        mask.mean()
    )


    C = {}


    for resolution in (
        256,
        512,
        1024,
    ):

        rank_map = percentile_map(
            maps[
                resolution
            ][
                i
            ]
        )


        C[
            resolution
        ] = contrast(
            rank_map,
            mask,
        )


    gain_256_512 = (
        C[512]
        -
        C[256]
    )


    gain_512_1024 = (
        C[1024]
        -
        C[512]
    )


    gain_256_1024 = (
        C[1024]
        -
        C[256]
    )


    detail.append({
        "index":
            i,

        "category":
            category,

        "image_path":
            get_image_path(
                row
            )
            or "",

        "mask_path":
            str(
                mask_path
            ),

        "defect_fraction":
            defect_fraction,

        "contrast_r256":
            C[256],

        "contrast_r512":
            C[512],

        "contrast_r1024":
            C[1024],

        "gain_256_512":
            gain_256_512,

        "gain_512_1024":
            gain_512_1024,

        "gain_256_1024":
            gain_256_1024,

        "positive_512_1024":
            bool(
                gain_512_1024
                >
                0
            ),
    })


    if (
        (i + 1) % 25 == 0
        or
        i + 1 == len(rows)
    ):

        print(
            f"processed "
            f"{i+1}/"
            f"{len(rows)}"
        )


with DETAIL_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            detail[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        detail
    )


# ============================================================
# Category summaries
# ============================================================

categories = sorted({
    r[
        "category"
    ]
    for r in detail
})


category_rows = []


for category in categories:

    members = [
        r
        for r in detail
        if r[
            "category"
        ]
        ==
        category
    ]


    gains = np.asarray([
        r[
            "gain_512_1024"
        ]
        for r in members
    ], dtype=np.float64)


    gains_256_512 = np.asarray([
        r[
            "gain_256_512"
        ]
        for r in members
    ], dtype=np.float64)


    gains_256_1024 = np.asarray([
        r[
            "gain_256_1024"
        ]
        for r in members
    ], dtype=np.float64)


    category_rows.append({
        "category":
            category,

        "N":
            len(
                members
            ),

        "median_gain_256_512":
            float(
                np.median(
                    gains_256_512
                )
            ),

        "median_gain_512_1024":
            float(
                np.median(
                    gains
                )
            ),

        "median_gain_256_1024":
            float(
                np.median(
                    gains_256_1024
                )
            ),

        "positive_fraction_512_1024":
            float(
                np.mean(
                    gains > 0
                )
            ),

        "median_positive":
            bool(
                np.median(
                    gains
                )
                >
                0
            ),
    })


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


# ============================================================
# Defect-size strata
#
# Diagnostic only.
# Not part of GO / NO-GO.
# ============================================================

fractions = np.asarray([
    r[
        "defect_fraction"
    ]
    for r in detail
])


q1, q2, q3 = np.quantile(
    fractions,
    [
        0.25,
        0.50,
        0.75,
    ],
)


def size_group(
    value
):

    if value <= q1:
        return "Q1_tiny"

    if value <= q2:
        return "Q2"

    if value <= q3:
        return "Q3"

    return "Q4_large"


for r in detail:

    r[
        "size_group"
    ] = size_group(
        r[
            "defect_fraction"
        ]
    )


size_rows = []


for group in [
    "Q1_tiny",
    "Q2",
    "Q3",
    "Q4_large",
]:

    members = [
        r
        for r in detail
        if r[
            "size_group"
        ]
        ==
        group
    ]


    gains = np.asarray([
        r[
            "gain_512_1024"
        ]
        for r in members
    ])


    size_rows.append({
        "group":
            group,

        "N":
            len(
                members
            ),

        "median_gain_512_1024":
            float(
                np.median(
                    gains
                )
            ),

        "mean_gain_512_1024":
            float(
                np.mean(
                    gains
                )
            ),

        "positive_fraction":
            float(
                np.mean(
                    gains > 0
                )
            ),
    })


with SIZE_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            size_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        size_rows
    )


# ============================================================
# Aggregate
# ============================================================

all_gains = np.asarray([
    r[
        "gain_512_1024"
    ]
    for r in detail
])


positive_fraction = float(
    np.mean(
        all_gains > 0
    )
)


median_gain = float(
    np.median(
        all_gains
    )
)


mean_gain = float(
    np.mean(
        all_gains
    )
)


category_positive = sum(
    r[
        "median_positive"
    ]
    for r in category_rows
)


signal_fraction = (
    positive_fraction
    >=
    0.70
)


signal_magnitude = (
    median_gain
    >=
    0.03
)


signal_categories = (
    category_positive
    >=
    6
)


decision = (
    "GO"
    if (
        signal_fraction
        and
        signal_magnitude
        and
        signal_categories
    )
    else "NO_GO"
)


payload = {
    "protocol": {
        "data":
            "existing A3 regular-public multiresolution maps",

        "gpu_compute":
            False,

        "response":
            (
                "per-image percentile-rank "
                "defect-background contrast"
            ),

        "primary_comparison":
            "512_to_1024",

        "public_data_used_for_method_tuning":
            False,
    },

    "pre_registered": {
        "minimum_positive_image_fraction":
            0.70,

        "minimum_median_gain":
            0.03,

        "minimum_positive_categories":
            6,
    },

    "observed": {
        "positive_image_fraction":
            positive_fraction,

        "median_gain_512_1024":
            median_gain,

        "mean_gain_512_1024":
            mean_gain,

        "positive_categories":
            int(
                category_positive
            ),
    },

    "signals": {
        "image_consistency":
            bool(
                signal_fraction
            ),

        "effect_magnitude":
            bool(
                signal_magnitude
            ),

        "category_consistency":
            bool(
                signal_categories
            ),
    },

    "decision":
        decision,
}


with SUMMARY_JSON.open(
    "w"
) as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


# ============================================================
# Console
# ============================================================

print()
print("=" * 112)
print("A9.0a RESOLUTION RESPONSE RESULTS")
print("=" * 112)

print(
    f"{'category':15s}"
    f"{'N':>5s}"
    f"{'256->512':>12s}"
    f"{'512->1024':>12s}"
    f"{'256->1024':>12s}"
    f"{'P(+)':>9s}"
)

print("-" * 69)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['N']:5d}"
        f"{r['median_gain_256_512']:12.4f}"
        f"{r['median_gain_512_1024']:12.4f}"
        f"{r['median_gain_256_1024']:12.4f}"
        f"{r['positive_fraction_512_1024']:9.3f}"
    )


print()
print("[DEFECT SIZE — DIAGNOSTIC ONLY]")

print(
    f"{'group':12s}"
    f"{'N':>6s}"
    f"{'median gain':>15s}"
    f"{'mean gain':>13s}"
    f"{'P(+)':>10s}"
)

print("-" * 58)


for r in size_rows:

    print(
        f"{r['group']:12s}"
        f"{r['N']:6d}"
        f"{r['median_gain_512_1024']:15.4f}"
        f"{r['mean_gain_512_1024']:13.4f}"
        f"{r['positive_fraction']:10.3f}"
    )


print()
print("[AGGREGATE]")

print(
    "positive image fraction :",
    f"{positive_fraction:.4f}"
)

print(
    "median response gain     :",
    f"{median_gain:+.4f}"
)

print(
    "mean response gain       :",
    f"{mean_gain:+.4f}"
)

print(
    "positive categories      :",
    f"{category_positive}/8"
)


print()
print("[PRE-REGISTERED A9.0a GO / NO-GO]")

print(
    "positive images >= .70 :",
    signal_fraction
)

print(
    "median gain >= .03     :",
    signal_magnitude
)

print(
    "positive cats >= 6/8   :",
    signal_categories
)

print()
print(
    "DECISION :",
    decision
)

print()
print(
    "DETAIL   :",
    DETAIL_CSV
)

print(
    "CATEGORY :",
    CATEGORY_CSV
)

print(
    "SIZE     :",
    SIZE_CSV
)

print(
    "SUMMARY  :",
    SUMMARY_JSON
)

print("=" * 112)
