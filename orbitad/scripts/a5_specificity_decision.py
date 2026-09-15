from pathlib import Path
import csv
import json


ROOT = Path(
    "orbitad/results/a5_orthogonal_scoring"
)

SUMMARY_CSV = (
    ROOT / "a5_2_method_summary.csv"
)

OUT_JSON = (
    ROOT / "a5_2_specificity_summary.json"
)


rows = {}

with SUMMARY_CSV.open() as f:

    for r in csv.DictReader(f):

        for key in list(
            r.keys()
        ):

            if key != "method":

                r[key] = float(
                    r[key]
                )

        rows[
            r["method"]
        ] = r


required = {
    "raw",
    "random16",
    "normalpca16",
    "wrongcat",
    "ant",
}


if set(
    rows.keys()
) != required:

    raise RuntimeError(
        f"Unexpected methods: "
        f"{rows.keys()}"
    )


raw = rows[
    "raw"
]

ant = rows[
    "ant"
]

rnd = rows[
    "random16"
]

pca = rows[
    "normalpca16"
]

wrong = rows[
    "wrongcat"
]


def shift_gain(
    method
):

    return (
        method[
            "shift123_aupro"
        ]
        -
        raw[
            "shift123_aupro"
        ]
    )


ant_gain = shift_gain(
    ant
)

random_gain = shift_gain(
    rnd
)

pca_gain = shift_gain(
    pca
)

wrong_gain = shift_gain(
    wrong
)


regular_delta = (
    ant[
        "regular_aupro"
    ]
    -
    raw[
        "regular_aupro"
    ]
)


effect_signal = (
    ant_gain >= 0.015
)

regular_safe = (
    regular_delta >= -0.015
)

beats_random = (
    ant_gain
    >=
    random_gain
    + 0.005
)

beats_pca = (
    ant_gain
    >=
    pca_gain
    + 0.005
)

beats_wrong = (
    ant_gain
    >=
    wrong_gain
    + 0.005
)


decision = (
    "GO"
    if (
        effect_signal
        and
        regular_safe
        and
        beats_random
        and
        beats_pca
        and
        beats_wrong
    )
    else "NO_GO"
)


print(
    "=" * 92
)

print(
    "A5.2 TANGENT SPECIFICITY DECISION"
)

print(
    "=" * 92
)

print()

print(
    f"{'method':16s}"
    f"{'Reg-PRO':>12s}"
    f"{'Shift-PRO':>12s}"
    f"{'ShiftGain':>12s}"
)

print(
    "-" * 52
)


for method in [
    "raw",
    "random16",
    "normalpca16",
    "wrongcat",
    "ant",
]:

    r = rows[
        method
    ]

    gain = (
        r[
            "shift123_aupro"
        ]
        -
        raw[
            "shift123_aupro"
        ]
    )

    print(
        f"{method:16s}"
        f"{r['regular_aupro']:12.4f}"
        f"{r['shift123_aupro']:12.4f}"
        f"{gain:+12.4f}"
    )


print()

print(
    "[ANT SPECIFICITY MARGINS]"
)

print(
    "ANT - Random16 gain      :",
    f"{ant_gain-random_gain:+.4f}"
)

print(
    "ANT - NormalPCA16 gain   :",
    f"{ant_gain-pca_gain:+.4f}"
)

print(
    "ANT - WrongCategory gain :",
    f"{ant_gain-wrong_gain:+.4f}"
)

print(
    "ANT regular PRO delta    :",
    f"{regular_delta:+.4f}"
)


print()

print(
    "[PRE-REGISTERED A5.2 GO / NO-GO]"
)

print(
    "ANT shift gain >= +.015        :",
    effect_signal
)

print(
    "ANT regular loss <= .015       :",
    regular_safe
)

print(
    "ANT > Random16 by >= .005      :",
    beats_random
)

print(
    "ANT > NormalPCA16 by >= .005   :",
    beats_pca
)

print(
    "ANT > WrongCategory by >= .005 :",
    beats_wrong
)

print()

print(
    "DECISION :",
    decision
)


payload = {
    "pre_registered": {
        "minimum_ant_shift_gain":
            0.015,

        "maximum_regular_loss":
            0.015,

        "minimum_margin_vs_random16":
            0.005,

        "minimum_margin_vs_normalpca16":
            0.005,

        "minimum_margin_vs_wrongcategory":
            0.005,
    },

    "observed": {
        "raw_shift_aupro":
            raw[
                "shift123_aupro"
            ],

        "ant_shift_aupro":
            ant[
                "shift123_aupro"
            ],

        "ant_shift_gain":
            ant_gain,

        "random16_shift_gain":
            random_gain,

        "normalpca16_shift_gain":
            pca_gain,

        "wrongcategory_shift_gain":
            wrong_gain,

        "ant_regular_delta":
            regular_delta,

        "ant_minus_random16":
            ant_gain
            -
            random_gain,

        "ant_minus_normalpca16":
            ant_gain
            -
            pca_gain,

        "ant_minus_wrongcategory":
            ant_gain
            -
            wrong_gain,
    },

    "signals": {
        "effect":
            effect_signal,

        "regular_safe":
            regular_safe,

        "random_specificity":
            beats_random,

        "normal_pca_specificity":
            beats_pca,

        "category_specificity":
            beats_wrong,
    },

    "decision":
        decision,
}


with OUT_JSON.open(
    "w"
) as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


print()

print(
    "SUMMARY :",
    OUT_JSON
)

print(
    "=" * 92
)
