from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image


PROJECT = Path(
    "/root/private_data/iad-vlm-anomaly/orbitad"
)

sys.path.insert(
    0,
    str(PROJECT / "src")
)

from orbitad.illumination.spatial_orbit import (
    exposure,
    directional_gradient,
    spotlight,
    shadow,
    low_frequency_field,
)


DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

OUT = Path(
    "orbitad/results/a2_synthetic_alignment/"
    "synthetic_orbit_sanity.jpg"
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


image_path = (
    DATA_ROOT
    / "can"
    / "validation"
    / "good"
    / "000_regular.png"
)

image = Image.open(
    image_path
).convert("RGB")

image = image.resize(
    (512, 512)
)

x = to_tensor(
    image
)[None]


g = torch.Generator()
g.manual_seed(
    20260913
)


variants = [
    (
        "regular",
        x
    ),

    (
        "exposure_0.70",
        exposure(
            x,
            0.70,
        )
    ),

    (
        "exposure_1.30",
        exposure(
            x,
            1.30,
        )
    ),

    (
        "gradient_0.30_45deg",
        directional_gradient(
            x,
            0.30,
            45,
        )
    ),

    (
        "gradient_0.45_135deg",
        directional_gradient(
            x,
            0.45,
            135,
        )
    ),

    (
        "spotlight",
        spotlight(
            x,
            0.35,
            center_x=-0.35,
            center_y=0.20,
            sigma_fraction=0.30,
        )
    ),

    (
        "shadow",
        shadow(
            x,
            0.35,
            center_x=0.35,
            center_y=-0.20,
            sigma_fraction=0.30,
        )
    ),

    (
        "lowfreq_g4",
        low_frequency_field(
            x,
            0.35,
            4,
            generator=g,
        )
    ),
]


thumb = 320
label_h = 40

canvas = Image.new(
    "RGB",
    (
        thumb * 4,
        (thumb + label_h) * 2,
    ),
    "white",
)

draw = ImageDraw.Draw(
    canvas
)


for i, (name, tensor) in enumerate(
    variants
):

    im = to_pil_image(
        tensor[0].clamp(
            0,
            1,
        )
    )

    im = im.resize(
        (thumb, thumb)
    )

    row = i // 4
    col = i % 4

    x0 = col * thumb
    y0 = row * (
        thumb + label_h
    )

    canvas.paste(
        im,
        (x0, y0)
    )

    draw.text(
        (
            x0 + 5,
            y0 + thumb + 10,
        ),
        name,
        fill="black",
    )


canvas.save(
    OUT,
    quality=92,
)

print("saved :", OUT)
print("STATUS: PASS")
