from __future__ import annotations

from pathlib import Path
import inspect
import importlib
import traceback


ROOT = Path(".").resolve()

DOC_DIR = ROOT / "docs/stage14_strong_vlm_baselines"
RES_DIR = ROOT / "results/stage14_strong_vlm_baselines"

DOC_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT = DOC_DIR / "stage14_c1_winclip_api_inspection.md"
OUT_TXT = RES_DIR / "stage14_c1_winclip_api_inspection.txt"

CANDIDATE_DATA_ROOTS = [
    ROOT / "datasets/MVTec_AD_2_anomalib_all/fruit_jelly_folder",
    ROOT / "datasets/MVTec_AD_2_anomalib_all/vial_folder",
    ROOT / "datasets/MVTec_AD_2_anomalib_all/walnuts_folder",
    ROOT / "datasets/MVTec_AD_2_anomalib_all/sheet_metal_folder",
]


def sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as e:
        return f"<signature_error: {repr(e)}>"


def safe_import(module_name: str, attr_name: str | None = None):
    try:
        module = importlib.import_module(module_name)
        if attr_name is None:
            return True, module, ""
        return True, getattr(module, attr_name), ""
    except Exception as e:
        return False, None, repr(e)


def path_status(path: Path) -> dict:
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


def main() -> None:
    lines = []
    txt = []

    lines.append("# Stage 14-C1 WinCLIP API Inspection")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report inspects the current Anomalib WinClip, Engine, and Folder APIs before running a WinCLIP pilot.")
    lines.append("It does not train or evaluate models.")
    lines.append("")

    imports = [
        ("anomalib", None),
        ("anomalib.models", "WinClip"),
        ("anomalib.engine", "Engine"),
        ("anomalib.data", "Folder"),
        ("anomalib.data.datamodules.image.folder", "Folder"),
    ]

    lines.append("## 2. Import and Signature Check")
    lines.append("")
    lines.append("| Module | Attribute | Import | Signature / Error |")
    lines.append("|---|---|---:|---|")

    objects = {}

    for module_name, attr_name in imports:
        ok, obj, err = safe_import(module_name, attr_name)
        key = attr_name or module_name
        objects[key] = obj if ok else None

        if ok:
            version = getattr(obj, "__version__", "") if attr_name is None else ""
            signature = version if version else sig(obj)
            lines.append(f"| `{module_name}` | `{attr_name or ''}` | True | `{signature}` |")
            txt.append(f"[OK] {module_name} {attr_name or ''}: {signature}")
        else:
            lines.append(f"| `{module_name}` | `{attr_name or ''}` | False | `{err}` |")
            txt.append(f"[ERR] {module_name} {attr_name or ''}: {err}")

    lines.append("")
    lines.append("## 3. Dataset Path Check")
    lines.append("")
    lines.append("| Dataset root | Exists | train/good | test/good | test/bad | ground_truth/bad | #train good | #test good | #test bad | #gt bad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for p in CANDIDATE_DATA_ROOTS:
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

    winclip = objects.get("WinClip")
    engine = objects.get("Engine")

    if winclip is not None:
        try:
            model = winclip()
            lines.append(f"- WinClip instantiation: success, type = `{type(model)}`")
            txt.append(f"[OK] WinClip instantiated: {type(model)}")
        except Exception as e:
            lines.append(f"- WinClip instantiation: failed, error = `{repr(e)}`")
            txt.append("[ERR] WinClip instantiation failed:\n" + traceback.format_exc())
    else:
        lines.append("- WinClip instantiation: skipped because import failed.")

    if engine is not None:
        try:
            eng = engine()
            lines.append(f"- Engine instantiation: success, type = `{type(eng)}`")
            txt.append(f"[OK] Engine instantiated: {type(eng)}")
        except Exception as e:
            lines.append(f"- Engine instantiation: failed, error = `{repr(e)}`")
            txt.append("[ERR] Engine instantiation failed:\n" + traceback.format_exc())
    else:
        lines.append("- Engine instantiation: skipped because import failed.")

    lines.append("")
    lines.append("## 5. Next Decision")
    lines.append("")
    lines.append("If Folder API is available and AD2 folder roots are valid, Stage 14-C2 should create a one-category WinCLIP pilot on fruit_jelly.")
    lines.append("If Folder API parameters are incompatible, Stage 14-C2 should first build a small adapter script based on the inspected signature.")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    OUT_TXT.write_text("\n".join(txt), encoding="utf-8")

    print("[DONE]", OUT_REPORT)
    print("[DONE]", OUT_TXT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
