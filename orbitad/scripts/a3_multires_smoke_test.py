import gc
import math

import torch
import torch.nn.functional as F
import timm


MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

RESOLUTIONS = [
    256,
    512,
    1024,
]


if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")

device = torch.device("cuda")


print("=" * 100)
print("OrbitAD A3.0-0 — MULTI-RESOLUTION DINOv3 SMOKE TEST")
print("=" * 100)


results = []


for resolution in RESOLUTIONS:

    print()
    print("-" * 100)
    print("RESOLUTION:", resolution)
    print("-" * 100)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
        img_size=resolution,
    )

    model = model.eval().to(device)

    x = torch.randn(
        1,
        3,
        resolution,
        resolution,
        device=device,
    )

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            out = model.forward_features(x)

    if isinstance(out, dict):

        if "x_norm_patchtokens" in out:

            patches = out[
                "x_norm_patchtokens"
            ]

        elif "x" in out:

            prefix = getattr(
                model,
                "num_prefix_tokens",
                1,
            )

            patches = out["x"][
                :,
                prefix:,
                :
            ]

        else:

            raise RuntimeError(
                f"Unsupported keys: {list(out.keys())}"
            )

    else:

        prefix = getattr(
            model,
            "num_prefix_tokens",
            1,
        )

        patches = out[
            :,
            prefix:,
            :
        ]

    patches = F.normalize(
        patches.float(),
        dim=-1,
    )

    B, N, D = patches.shape

    grid = int(
        round(
            math.sqrt(N)
        )
    )

    expected_grid = (
        resolution // 16
    )

    expected_patches = (
        expected_grid ** 2
    )

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    norm = patches.norm(
        dim=-1
    )

    print(
        "input             :",
        f"{resolution} x {resolution}"
    )

    print(
        "prefix tokens     :",
        getattr(
            model,
            "num_prefix_tokens",
            "unknown",
        )
    )

    print(
        "patch tensor      :",
        tuple(
            patches.shape
        )
    )

    print(
        "grid              :",
        f"{grid} x {grid}"
    )

    print(
        "expected grid     :",
        f"{expected_grid} x {expected_grid}"
    )

    print(
        "feature dim       :",
        D
    )

    print(
        "peak GPU memory   :",
        f"{peak_gb:.3f} GB"
    )

    print(
        "norm range        :",
        float(
            norm.min()
        ),
        "->",
        float(
            norm.max()
        )
    )

    if B != 1:
        raise RuntimeError(
            "Unexpected batch size."
        )

    if N != expected_patches:
        raise RuntimeError(
            f"{resolution}: expected "
            f"{expected_patches} patches, "
            f"got {N}"
        )

    if grid != expected_grid:
        raise RuntimeError(
            f"{resolution}: grid mismatch."
        )

    if D != 768:
        raise RuntimeError(
            f"{resolution}: expected D=768, got {D}"
        )

    if not torch.isfinite(
        patches
    ).all():

        raise RuntimeError(
            f"{resolution}: non-finite features."
        )

    results.append(
        (
            resolution,
            grid,
            N,
            peak_gb,
        )
    )

    del x
    del out
    del patches
    del model

    gc.collect()
    torch.cuda.empty_cache()


print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    f"{'resolution':>12s}"
    f"{'grid':>12s}"
    f"{'patches':>12s}"
    f"{'peak_GB':>12s}"
)

print("-" * 48)

for resolution, grid, n, peak in results:

    print(
        f"{resolution:12d}"
        f"{(str(grid)+'x'+str(grid)):>12s}"
        f"{n:12d}"
        f"{peak:12.3f}"
    )


print()
print("STATUS: PASS")
print("=" * 100)
