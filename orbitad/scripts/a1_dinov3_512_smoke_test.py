import math
import torch
import timm
import torch.nn.functional as F

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
INPUT_SIZE = 512

print("=" * 90)
print("OrbitAD A1.3-0 — DINOv3 512 PATCH SMOKE TEST")
print("=" * 90)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")

print("Loading:", MODEL_NAME)

# Explicitly instantiate model for 512x512.
# timm will adapt the pretrained positional embedding.
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    img_size=INPUT_SIZE,
)

model = model.eval().to(device)

print("num_prefix_tokens:",
      getattr(model, "num_prefix_tokens", "unknown"))

x = torch.randn(
    1, 3, INPUT_SIZE, INPUT_SIZE,
    device=device
)

with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        output = model.forward_features(x)

print("forward_features type:", type(output))

# ------------------------------------------------------------
# Robustly extract patch tokens
# ------------------------------------------------------------

if isinstance(output, dict):

    print("dict keys:", list(output.keys()))

    if "x_norm_patchtokens" in output:
        patches = output["x_norm_patchtokens"]

    elif "x" in output:
        tokens = output["x"]

        prefix = getattr(
            model,
            "num_prefix_tokens",
            1
        )

        patches = tokens[:, prefix:, :]

    else:
        raise RuntimeError(
            "Unsupported forward_features dictionary."
        )

else:

    tokens = output

    if tokens.ndim != 3:
        raise RuntimeError(
            f"Expected [B,N,D], got {tuple(tokens.shape)}"
        )

    prefix = getattr(
        model,
        "num_prefix_tokens",
        1
    )

    patches = tokens[:, prefix:, :]


if patches.ndim != 3:
    raise RuntimeError(
        f"Invalid patch shape: {tuple(patches.shape)}"
    )

patches = F.normalize(
    patches.float(),
    dim=-1
)

B, N, D = patches.shape

grid = int(round(math.sqrt(N)))

print()
print("input shape        :", tuple(x.shape))
print("patch tensor shape :", tuple(patches.shape))
print("patch count        :", N)
print("feature dim        :", D)
print("inferred grid      :", f"{grid} x {grid}")

print()
print(
    "GPU allocated     :",
    f"{torch.cuda.memory_allocated()/1024**3:.3f} GB"
)

print(
    "GPU reserved      :",
    f"{torch.cuda.memory_reserved()/1024**3:.3f} GB"
)

# ------------------------------------------------------------
# Strict checks
# ------------------------------------------------------------

if N != 1024:
    raise RuntimeError(
        f"Expected 1024 patches for 512/16, got {N}"
    )

if grid != 32:
    raise RuntimeError(
        f"Expected 32x32 grid, got {grid}x{grid}"
    )

if D != 768:
    raise RuntimeError(
        f"Expected feature dim 768, got {D}"
    )

if not torch.isfinite(patches).all():
    raise RuntimeError("Non-finite feature detected")

norms = patches.norm(dim=-1)

print()
print(
    "normalized feature norm range:",
    float(norms.min()),
    "->",
    float(norms.max())
)

print()
print("STATUS: PASS")
print("=" * 90)
