import torch
import timm

print("=" * 80)
print("OrbitAD A1.0 — DINOv3 ViT-B/16 SMOKE TEST")
print("=" * 80)

print("torch :", torch.__version__)
print("timm  :", timm.__version__)
print("cuda  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = torch.device("cuda")

model_name = "vit_base_patch16_dinov3.lvd1689m"

print("\nLoading:", model_name)

model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=0,
)

model = model.eval().to(device)

num_params = sum(p.numel() for p in model.parameters())

print("parameters :", f"{num_params / 1e6:.2f} M")

# First smoke test: synthetic image
x = torch.randn(
    1, 3, 256, 256,
    device=device
)

with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        y = model(x)

print("input shape :", tuple(x.shape))
print("output shape:", tuple(y.shape))
print("output dtype:", y.dtype)

if y.ndim != 2:
    raise RuntimeError(
        f"Expected global feature [B,D], got {tuple(y.shape)}"
    )

if y.shape[0] != 1:
    raise RuntimeError("Unexpected batch dimension")

if y.shape[-1] != 768:
    raise RuntimeError(
        f"Expected 768-D ViT-B feature, got {y.shape[-1]}"
    )

if not torch.isfinite(y).all():
    raise RuntimeError("Non-finite feature detected")

print("\nGPU memory:")
print(
    " allocated:",
    f"{torch.cuda.memory_allocated() / 1024**3:.3f} GB"
)
print(
    " reserved :",
    f"{torch.cuda.memory_reserved() / 1024**3:.3f} GB"
)

print("\nSTATUS: PASS")
