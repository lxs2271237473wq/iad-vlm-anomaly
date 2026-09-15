import math
import torch
import torch.nn.functional as F


def _coordinate_grid(x):
    """
    x: [B,C,H,W]
    returns xx, yy in [-1,1], shape [1,1,H,W]
    """

    _, _, h, w = x.shape

    yy = torch.linspace(
        -1.0,
        1.0,
        h,
        device=x.device,
        dtype=x.dtype,
    )

    xx = torch.linspace(
        -1.0,
        1.0,
        w,
        device=x.device,
        dtype=x.dtype,
    )

    yy, xx = torch.meshgrid(
        yy,
        xx,
        indexing="ij",
    )

    return (
        xx[None, None],
        yy[None, None],
    )


def apply_field(x, field):
    """
    Multiplicative illumination field.

    x     : [B,C,H,W], assumed [0,1]
    field : [B or 1,1,H,W]
    """

    return torch.clamp(
        x * field,
        0.0,
        1.0,
    )


def exposure(x, factor):
    return torch.clamp(
        x * float(factor),
        0.0,
        1.0,
    )


def directional_gradient(
    x,
    amplitude,
    direction_deg,
):
    """
    Smooth directional illumination gradient.
    Mean approximately centered around 1.
    """

    xx, yy = _coordinate_grid(x)

    theta = math.radians(
        float(direction_deg)
    )

    direction = (
        math.cos(theta) * xx
        +
        math.sin(theta) * yy
    )

    field = (
        1.0
        +
        float(amplitude)
        * direction
    )

    return apply_field(
        x,
        field,
    )


def gaussian_field(
    x,
    amplitude,
    center_x,
    center_y,
    sigma_fraction,
    sign=1.0,
):
    xx, yy = _coordinate_grid(x)

    cx = float(center_x)
    cy = float(center_y)

    sigma = (
        2.0
        * float(sigma_fraction)
    )

    d2 = (
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
    )

    blob = torch.exp(
        -d2
        /
        (
            2.0
            * sigma * sigma
        )
    )

    field = (
        1.0
        +
        float(sign)
        * float(amplitude)
        * blob
    )

    return apply_field(
        x,
        field,
    )


def spotlight(
    x,
    amplitude,
    center_x=0.0,
    center_y=0.0,
    sigma_fraction=0.30,
):
    return gaussian_field(
        x,
        amplitude,
        center_x,
        center_y,
        sigma_fraction,
        sign=1.0,
    )


def shadow(
    x,
    amplitude,
    center_x=0.0,
    center_y=0.0,
    sigma_fraction=0.30,
):
    return gaussian_field(
        x,
        amplitude,
        center_x,
        center_y,
        sigma_fraction,
        sign=-1.0,
    )


def low_frequency_field(
    x,
    amplitude,
    grid_size,
    generator=None,
):
    """
    Random low-frequency multiplicative illumination field.
    Geometry is untouched.
    """

    b, _, h, w = x.shape

    noise = torch.rand(
        b,
        1,
        int(grid_size),
        int(grid_size),
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )

    # Center around zero.
    noise = (
        noise - 0.5
    ) * 2.0

    noise = F.interpolate(
        noise,
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )

    # Normalize each generated field.
    denom = (
        noise
        .abs()
        .flatten(1)
        .amax(dim=1)
        .clamp_min(1e-6)
        [:, None, None, None]
    )

    noise = (
        noise / denom
    )

    field = (
        1.0
        +
        float(amplitude)
        * noise
    )

    return apply_field(
        x,
        field,
    )
