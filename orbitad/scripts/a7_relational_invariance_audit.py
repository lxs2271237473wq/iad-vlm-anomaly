from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import (
    resize,
    to_tensor,
)
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

A2_ROOT = Path(
    "orbitad/results/a2_counterfactual_persistence"
)

OUT_ROOT = Path(
    "orbitad/results/a7_relational"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


PUBLIC_META = (
    A2_ROOT / "public_orbit_meta.csv"
)

DETAIL_CSV = (
    OUT_ROOT / "a7_0_pair_invariance.csv"
)

CATEGORY_CSV = (
    OUT_ROOT / "a7_0_category_summary.csv"
)

SUMMARY_JSON = (
    OUT_ROOT / "a7_0_summary.json"
)


MODEL_NAME = (
    "vit_base_patch16_dinov3.lvd1689m"
)

INPUT_SIZE = 512
GRID = 32
FEATURE_DIM = 768

PRIMARY_CONDITIONS = [
    "shift_1",
    "shift_2",
    "shift_3",
]


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable"
    )

device = torch.device(
    "cuda"
)


# ============================================================
# Metadata
# ============================================================

rows = []

with PUBLIC_META.open() as f:

    for r in csv.DictReader(f):

        r["label"] = int(
            r["label"]
        )

        rows.append(r)


normal_rows = [
    r
    for r in rows
    if r["label"] == 0
]


pairs = defaultdict(
    dict
)


for row in normal_rows:

    key = (
        row["category"],
        row["instance_id"],
    )

    pairs[
        key
    ][
        row["condition"]
    ] = row


CATEGORIES = sorted({
    r["category"]
    for r in normal_rows
})


# ============================================================
# Backbone
# ============================================================

print("=" * 112)
print(
    "A7.0 — LOCAL RELATIONAL SIGNATURE "
    "INVARIANCE AUDIT"
)
print("=" * 112)


model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
).eval().to(device)


cfg = resolve_model_data_config(
    model
)


mean = torch.tensor(
    cfg["mean"],
    dtype=torch.float32,
    device=device,
).view(
    1,
    3,
    1,
    1,
)

std = torch.tensor(
    cfg["std"],
    dtype=torch.float32,
    device=device,
).view(
    1,
    3,
    1,
    1,
)


def load_image(
    rel_path
):

    path = (
        DATA_ROOT
        / rel_path
    )

    with Image.open(path) as im:

        im = im.convert(
            "RGB"
        )

        im = resize(
            im,
            [
                INPUT_SIZE,
                INPUT_SIZE,
            ],
            interpolation=(
                InterpolationMode.BICUBIC
            ),
            antialias=True,
        )

        return to_tensor(
            im
        )


def extract_tokens(
    raw
):

    x = raw.to(
        device
    )

    x = (
        x
        -
        mean
    ) / std


    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            out = model.forward_features(
                x
            )


    if isinstance(
        out,
        dict,
    ):

        if "x_norm_patchtokens" in out:

            p = out[
                "x_norm_patchtokens"
            ]

        elif "x" in out:

            prefix = getattr(
                model,
                "num_prefix_tokens",
                1,
            )

            p = out["x"][
                :,
                prefix:,
                :
            ]

        else:

            raise RuntimeError(
                str(
                    list(
                        out.keys()
                    )
                )
            )

    else:

        prefix = getattr(
            model,
            "num_prefix_tokens",
            1,
        )

        p = out[
            :,
            prefix:,
            :
        ]


    p = F.normalize(
        p.float(),
        dim=-1,
    )


    if p.shape[1:] != (
        1024,
        FEATURE_DIM,
    ):

        raise RuntimeError(
            f"Unexpected feature shape: "
            f"{tuple(p.shape)}"
        )


    return p


# ============================================================
# Local relational signature
#
# 5x5 self-similarity around each patch:
# 24 center-to-neighbour cosine relations.
#
# Per-patch standardization removes absolute
# similarity level and retains relational shape.
# ============================================================

def relational_signature(
    tokens
):

    # tokens:
    # [B,1024,768]

    B = tokens.shape[0]


    z = tokens.reshape(
        B,
        GRID,
        GRID,
        FEATURE_DIM,
    )


    # [B,D,H,W]
    z_chw = z.permute(
        0,
        3,
        1,
        2,
    )


    # Reflection avoids wrap-around artifacts.
    padded = F.pad(
        z_chw,
        (
            2,
            2,
            2,
            2,
        ),
        mode="reflect",
    )


    # [B,D,H,W,5,5]
    windows = (
        padded
        .unfold(
            2,
            5,
            1,
        )
        .unfold(
            3,
            5,
            1,
        )
    )


    center = z_chw[
        :,
        :,
        :,
        :,
        None,
        None,
    ]


    # [B,H,W,5,5]
    sim = (
        windows
        *
        center
    ).sum(
        dim=1
    )


    sim = sim.reshape(
        B,
        GRID,
        GRID,
        25,
    )


    # Remove center self-similarity.
    descriptor = torch.cat([
        sim[
            ...,
            :12
        ],
        sim[
            ...,
            13:
        ],
    ], dim=-1)


    mu = descriptor.mean(
        dim=-1,
        keepdim=True,
    )

    sigma = descriptor.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(
        1e-6
    )


    descriptor = (
        descriptor
        -
        mu
    ) / sigma


    return descriptor.reshape(
        B,
        1024,
        24,
    )


# ============================================================
# Cache every public normal feature + relation descriptor
# ============================================================

cache = {}


all_needed_rows = []


for (
    category,
    instance_id,
), conditions in sorted(
    pairs.items()
):

    for condition, row in sorted(
        conditions.items()
    ):

        if (
            condition == "regular"
            or
            condition in PRIMARY_CONDITIONS
        ):

            all_needed_rows.append(
                (
                    category,
                    instance_id,
                    condition,
                    row,
                )
            )


for i, (
    category,
    instance_id,
    condition,
    row,
) in enumerate(
    all_needed_rows,
    1,
):

    raw = load_image(
        row["image_path"]
    )[None]


    tokens = extract_tokens(
        raw
    )


    relation = relational_signature(
        tokens
    )


    cache[
        (
            category,
            instance_id,
            condition,
        )
    ] = {
        "raw":
            tokens[
                0
            ].cpu(),

        "relation":
            relation[
                0
            ].cpu(),
    }


    if (
        i % 25 == 0
        or
        i == len(
            all_needed_rows
        )
    ):

        print(
            f"features "
            f"{i}/{len(all_needed_rows)}"
        )


# ============================================================
# Distances
# ============================================================

def raw_distance(
    a,
    b
):

    # features already L2 normalized
    cos = (
        a * b
    ).sum(
        dim=-1
    ).clamp(
        -1.0,
        1.0,
    )


    d = torch.sqrt(
        (
            2.0
            -
            2.0
            * cos
        ).clamp_min(
            0.0
        )
    )


    return float(
        d.mean().item()
    )


def relation_distance(
    a,
    b
):

    d = torch.sqrt(
        (
            (
                a - b
            ) ** 2
        ).mean(
            dim=-1
        )
    )


    return float(
        d.mean().item()
    )


# ============================================================
# Pair audit
#
# Other-instance control:
# deterministic cyclic next regular instance within
# category + condition availability.
# ============================================================

detail = []


for category in CATEGORIES:

    for condition in PRIMARY_CONDITIONS:

        valid_instances = sorted([
            instance_id
            for (
                cat,
                instance_id,
            ), conditions
            in pairs.items()

            if (
                cat == category
                and
                "regular" in conditions
                and
                condition in conditions
            )
        ])


        if len(
            valid_instances
        ) < 2:

            continue


        for idx, instance_id in enumerate(
            valid_instances
        ):

            other_id = valid_instances[
                (
                    idx + 1
                )
                %
                len(
                    valid_instances
                )
            ]


            regular = cache[
                (
                    category,
                    instance_id,
                    "regular",
                )
            ]


            shifted = cache[
                (
                    category,
                    instance_id,
                    condition,
                )
            ]


            other_regular = cache[
                (
                    category,
                    other_id,
                    "regular",
                )
            ]


            raw_same = raw_distance(
                shifted["raw"],
                regular["raw"],
            )

            raw_other = raw_distance(
                shifted["raw"],
                other_regular["raw"],
            )


            rel_same = relation_distance(
                shifted["relation"],
                regular["relation"],
            )

            rel_other = relation_distance(
                shifted["relation"],
                other_regular["relation"],
            )


            raw_ratio = (
                raw_same
                /
                max(
                    raw_other,
                    1e-12,
                )
            )

            rel_ratio = (
                rel_same
                /
                max(
                    rel_other,
                    1e-12,
                )
            )


            detail.append({
                "category":
                    category,

                "instance_id":
                    instance_id,

                "condition":
                    condition,

                "other_instance_id":
                    other_id,

                "raw_same":
                    raw_same,

                "raw_other":
                    raw_other,

                "raw_ratio":
                    raw_ratio,

                "relation_same":
                    rel_same,

                "relation_other":
                    rel_other,

                "relation_ratio":
                    rel_ratio,

                "relative_ratio":
                    (
                        rel_ratio
                        /
                        max(
                            raw_ratio,
                            1e-12,
                        )
                    ),
            })


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
# Instance retrieval
#
# Query = shifted image.
# Gallery = all regular images in same category
# having that condition.
#
# Distance is mean same-position patch distance.
# ============================================================

retrieval = defaultdict(
    lambda: {
        "raw_correct": 0,
        "relation_correct": 0,
        "total": 0,
    }
)


for category in CATEGORIES:

    for condition in PRIMARY_CONDITIONS:

        valid_instances = sorted([
            instance_id
            for (
                cat,
                instance_id,
            ), conditions
            in pairs.items()

            if (
                cat == category
                and
                "regular" in conditions
                and
                condition in conditions
            )
        ])


        if len(
            valid_instances
        ) < 2:

            continue


        for instance_id in (
            valid_instances
        ):

            query = cache[
                (
                    category,
                    instance_id,
                    condition,
                )
            ]


            raw_candidates = []
            relation_candidates = []


            for gallery_id in (
                valid_instances
            ):

                gallery = cache[
                    (
                        category,
                        gallery_id,
                        "regular",
                    )
                ]


                raw_candidates.append(
                    (
                        raw_distance(
                            query["raw"],
                            gallery["raw"],
                        ),
                        gallery_id,
                    )
                )


                relation_candidates.append(
                    (
                        relation_distance(
                            query["relation"],
                            gallery["relation"],
                        ),
                        gallery_id,
                    )
                )


            raw_pred = min(
                raw_candidates
            )[1]


            relation_pred = min(
                relation_candidates
            )[1]


            key = (
                category,
                condition,
            )


            retrieval[
                key
            ][
                "raw_correct"
            ] += int(
                raw_pred
                ==
                instance_id
            )


            retrieval[
                key
            ][
                "relation_correct"
            ] += int(
                relation_pred
                ==
                instance_id
            )


            retrieval[
                key
            ][
                "total"
            ] += 1


# ============================================================
# Category aggregates
# ============================================================

category_rows = []


for category in CATEGORIES:

    members = [
        r
        for r in detail
        if r[
            "category"
        ] == category
    ]


    if not members:

        continue


    raw_ratio = float(
        np.mean([
            r[
                "raw_ratio"
            ]
            for r in members
        ])
    )


    relation_ratio = float(
        np.mean([
            r[
                "relation_ratio"
            ]
            for r in members
        ])
    )


    totals = [
        v
        for (
            cat,
            condition
        ), v
        in retrieval.items()
        if cat == category
    ]


    total = sum(
        x["total"]
        for x in totals
    )


    raw_correct = sum(
        x["raw_correct"]
        for x in totals
    )


    relation_correct = sum(
        x["relation_correct"]
        for x in totals
    )


    category_rows.append({
        "category":
            category,

        "pairs":
            len(
                members
            ),

        "raw_ratio":
            raw_ratio,

        "relation_ratio":
            relation_ratio,

        "relation_over_raw":
            (
                relation_ratio
                /
                max(
                    raw_ratio,
                    1e-12,
                )
            ),

        "relation_better":
            bool(
                relation_ratio
                <
                raw_ratio
            ),

        "raw_retrieval":
            (
                raw_correct
                /
                max(
                    total,
                    1,
                )
            ),

        "relation_retrieval":
            (
                relation_correct
                /
                max(
                    total,
                    1,
                )
            ),
    })


with CATEGORY_CSV.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            category_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        category_rows
    )


# ============================================================
# Global decision
# ============================================================

mean_raw_ratio = float(
    np.mean([
        r[
            "raw_ratio"
        ]
        for r in detail
    ])
)


mean_relation_ratio = float(
    np.mean([
        r[
            "relation_ratio"
        ]
        for r in detail
    ])
)


ratio_fraction = (
    mean_relation_ratio
    /
    max(
        mean_raw_ratio,
        1e-12,
    )
)


category_wins = sum(
    r[
        "relation_better"
    ]
    for r in category_rows
)


retrieval_total = sum(
    v["total"]
    for v in retrieval.values()
)


raw_retrieval = (
    sum(
        v[
            "raw_correct"
        ]
        for v in retrieval.values()
    )
    /
    max(
        retrieval_total,
        1,
    )
)


relation_retrieval = (
    sum(
        v[
            "relation_correct"
        ]
        for v in retrieval.values()
    )
    /
    max(
        retrieval_total,
        1,
    )
)


signal_ratio = (
    ratio_fraction
    <= 0.75
)


signal_categories = (
    category_wins
    >= 6
)


signal_retrieval = (
    relation_retrieval
    >=
    raw_retrieval
    -
    0.05
)


decision = (
    "GO"
    if (
        signal_ratio
        and
        signal_categories
        and
        signal_retrieval
    )
    else "NO_GO"
)


payload = {
    "descriptor": {
        "backbone":
            MODEL_NAME,

        "grid":
            "32x32",

        "local_window":
            "5x5",

        "relation_dimension":
            24,

        "per_patch_standardization":
            True,
    },

    "public_bad_or_gt_used":
        False,

    "pre_registered": {
        "maximum_relation_over_raw_ratio":
            0.75,

        "minimum_categories_improved":
            6,

        "maximum_retrieval_drop":
            0.05,
    },

    "observed": {
        "mean_raw_ratio":
            mean_raw_ratio,

        "mean_relation_ratio":
            mean_relation_ratio,

        "relation_over_raw_ratio":
            ratio_fraction,

        "categories_improved":
            int(
                category_wins
            ),

        "raw_instance_retrieval":
            raw_retrieval,

        "relation_instance_retrieval":
            relation_retrieval,

        "retrieval_delta":
            (
                relation_retrieval
                -
                raw_retrieval
            ),
    },

    "signals": {
        "shift_invariance":
            bool(
                signal_ratio
            ),

        "category_consistency":
            bool(
                signal_categories
            ),

        "identity_preservation":
            bool(
                signal_retrieval
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
print("A7.0 RELATIONAL INVARIANCE RESULTS")
print("=" * 112)

print(
    f"{'category':15s}"
    f"{'RawRatio':>11s}"
    f"{'RelRatio':>11s}"
    f"{'Rel/Raw':>10s}"
    f"{'RawRet':>10s}"
    f"{'RelRet':>10s}"
)

print("-" * 67)


for r in category_rows:

    print(
        f"{r['category']:15s}"
        f"{r['raw_ratio']:11.4f}"
        f"{r['relation_ratio']:11.4f}"
        f"{r['relation_over_raw']:10.4f}"
        f"{r['raw_retrieval']:10.3f}"
        f"{r['relation_retrieval']:10.3f}"
    )


print()
print("[AGGREGATE]")

print(
    "mean raw ratio          :",
    f"{mean_raw_ratio:.4f}"
)

print(
    "mean relational ratio   :",
    f"{mean_relation_ratio:.4f}"
)

print(
    "relation / raw          :",
    f"{ratio_fraction:.4f}"
)

print(
    "categories improved     :",
    f"{category_wins}/"
    f"{len(category_rows)}"
)

print(
    "raw retrieval           :",
    f"{raw_retrieval:.4f}"
)

print(
    "relational retrieval    :",
    f"{relation_retrieval:.4f}"
)

print(
    "retrieval delta         :",
    f"{relation_retrieval-raw_retrieval:+.4f}"
)


print()
print("[PRE-REGISTERED A7.0 GO / NO-GO]")

print(
    "relation/raw <= 0.75      :",
    signal_ratio
)

print(
    "improved in >=6/8 cats    :",
    signal_categories
)

print(
    "retrieval drop <= 0.05    :",
    signal_retrieval
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
    "SUMMARY  :",
    SUMMARY_JSON
)

print("=" * 112)
