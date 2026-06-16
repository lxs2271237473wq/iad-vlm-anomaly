"""Check whether the industrial anomaly detection baseline environment is usable."""
from __future__ import annotations

import importlib
import platform


def version_of(pkg: str) -> str:
    try:
        module = importlib.import_module(pkg)
        return getattr(module, "__version__", "version attribute not found")
    except Exception as exc:  # noqa: BLE001
        return f"NOT OK: {exc}"


print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("torch:", version_of("torch"))

try:
    import torch

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
except Exception as exc:  # noqa: BLE001
    print("Torch CUDA check failed:", exc)

print("anomalib:", version_of("anomalib"))
print("cv2:", version_of("cv2"))
print("pandas:", version_of("pandas"))
