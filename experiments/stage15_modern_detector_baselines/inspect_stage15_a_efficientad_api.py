from __future__ import annotations

from pathlib import Path
import importlib
import inspect
import traceback


ROOT = Path(".").resolve()

DOC_DIR = ROOT / "docs/stage15_modern_detector_baselines"
RES_DIR = ROOT / "results/stage15_modern_detector_baselines"

OUT_REPORT = DOC_DIR / "stage15_a_efficientad_api_inspection.md"
OUT_TXT = RES_DIR / "stage15_a_efficientad_api_inspection.txt"

DOC_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


IMPORT_CANDIDATES = [
    ("anomalib.models", "EfficientAd"),
    ("anomalib.models", "EfficientAD"),
    ("anomalib.models.image.efficient_ad", "EfficientAd"),
    ("anomalib.models.image.efficient_ad", "EfficientAD"),
    ("anomalib.models.image.efficientad", "EfficientAd"),
    ("anomalib.models.image.efficientad", "EfficientAD"),
    ("anomalib.engine", "Engine"),
    ("anomalib.data", "Folder"),
]


DATA_ROOTS = [
    ROOT / "datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder",
    ROOT / "datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder",
    ROOT / "datasets/MVTec_AD_2_anomalib_all/vial_folder",
    ROOT / "datasets/MVTec_AD_2_anomalib_all/walnuts_folder",
]


def get_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception as e:
        return f"<signature_error: {repr(e)}>"


def safe_import(module_name: str, attr_name: str):
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr_name)
        return True, obj, ""
    except Exception as e:
        return False, None, repr(e)


def path_status(path: Path):
    return {
        "path": str(path.relative_to(ROOT) if path.exists() else path),
        "exists": path.exists(),
        "train_good": (path / "train/good").exists(),
        "test_good": (path / "test/good").exists(),
        "test_bad": (path / "test/bad").exists(),
        "ground_truth_bad": (path / "ground_truth/bad").exists(),
        "num_train_good": len(list((path / "train/good").glob("*"))) if (path / "train/good").exists() else 0,
        "num_test_good": len(list((path / "test/good").glob("*"))) if (path / "test/good").exists() else 0,
        "num_test_bad": len(list((path / "test/bad").glob("*"))) if (path / "test/bad").exists() else 0,
        "num_gt_bad": len(list((path / "ground_truth/bad").glob("*"))) if (path / "ground_truth/bad").exists() else 0,
    }


def main():
    lines = []
    txt = []

    lines.append("# Stage 15-A EfficientAD API Inspection")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage checks whether EfficientAD is available in the current Anomalib 2.5.0 environment.")
    lines.append("")
    lines.append("It does not train or evaluate any model.")
    lines.append("")
    lines.append("## 2. Import and Signature Results")
    lines.append("")
    lines.append("| Module | Attribute | Import | Signature / Error |")
    lines.append("|---|---|---:|---|")

    imported_objects = {}

    for module_name, attr_name in IMPORT_CANDIDATES:
        ok, obj, err = safe_import(module_name, attr_name)
        key = f"{module_name}.{attr_name}"

        if ok:
            sig = get_signature(obj)
            imported_objects[key] = obj
            lines.append(f"| `{module_name}` | `{attr_name}` | True | `{sig}` |")
            txt.append(f"[OK] {key}: {sig}")
        else:
            lines.append(f"| `{module_name}` | `{attr_name}` | False | `{err}` |")
            txt.append(f"[ERR] {key}: {err}")

    lines.append("")
    lines.append("## 3. Dataset Path Check")
    lines.append("")
    lines.append("| Dataset root | Exists | train/good | test/good | test/bad | ground_truth/bad | #train good | #test good | #test bad | #gt bad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for p in DATA_ROOTS:
        st = path_status(p)
        lines.append(
            f"| `{st['path']}` | {st['exists']} | {st['train_good']} | {st['test_good']} | "
            f"{st['test_bad']} | {st['ground_truth_bad']} | {st['num_train_good']} | "
            f"{st['num_test_good']} | {st['num_test_bad']} | {st['num_gt_bad']} |"
        )
        txt.append(str(st))

    lines.append("")
    lines.append("## 4. Minimal Instantiation Check")
    lines.append("")

    efficientad_obj = None
    efficientad_name = ""

    for key, obj in imported_objects.items():
        if "Efficient" in key:
            efficientad_obj = obj
            efficientad_name = key
            break

    if efficientad_obj is None:
        lines.append("- EfficientAD instantiation: skipped because no EfficientAD class was imported.")
        txt.append("[SKIP] EfficientAD instantiation: no class imported.")
    else:
        try:
            model = efficientad_obj()
            lines.append(f"- EfficientAD instantiation from `{efficientad_name}`: success, type = `{type(model)}`")
            txt.append(f"[OK] EfficientAD instantiated from {efficientad_name}: {type(model)}")
        except Exception:
            err = traceback.format_exc()
            lines.append(f"- EfficientAD instantiation from `{efficientad_name}`: failed.")
            lines.append("")
            lines.append("```text")
            lines.append(err)
            lines.append("```")
            txt.append("[ERR] EfficientAD instantiation failed:\n" + err)

    lines.append("")
    lines.append("## 5. Decision")
    lines.append("")
    lines.append("If EfficientAD can be imported and instantiated, Stage 15-B should run a one-category pilot on fruit_jelly or vial.")
    lines.append("")
    lines.append("If EfficientAD requires extra constructor parameters, Stage 15-B should use the inspected signature to create a valid pilot script.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    OUT_TXT.write_text("\n".join(txt), encoding="utf-8")

    print("[DONE]", OUT_REPORT)
    print("[DONE]", OUT_TXT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
