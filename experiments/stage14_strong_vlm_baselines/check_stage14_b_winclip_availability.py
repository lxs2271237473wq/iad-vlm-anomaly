from pathlib import Path
import importlib
import traceback

DOC_DIR = Path("docs/stage14_strong_vlm_baselines")
RES_DIR = Path("results/stage14_strong_vlm_baselines")

DOC_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

report_path = DOC_DIR / "stage14_b_winclip_availability_report.md"
txt_path = RES_DIR / "stage14_b_winclip_import_check.txt"

candidate_imports = [
    ("anomalib.models", "WinClip"),
    ("anomalib.models", "WinCLIP"),
    ("anomalib.models.image.winclip", "WinClip"),
    ("anomalib.models.image.winclip", "WinCLIP"),
    ("anomalib.models.image.winclip.torch_model", "WinClipModel"),
]

results = []
available = []

for module_name, attr_name in candidate_imports:
    item = {
        "module": module_name,
        "attribute": attr_name,
        "import_success": False,
        "instantiate_success": False,
        "error": "",
    }

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr_name)
        item["import_success"] = True

        try:
            if attr_name in {"WinClip", "WinCLIP"}:
                instance = obj()
                item["instantiate_success"] = True
                item["instance_type"] = str(type(instance))
            else:
                item["instantiate_success"] = "not_attempted"
                item["instance_type"] = str(obj)
        except Exception as e:
            item["error"] = "instantiate_error: " + repr(e)

        available.append(item)

    except Exception as e:
        item["error"] = repr(e)

    results.append(item)

lines = []
lines.append("# Stage 14-B WinCLIP Availability Report")
lines.append("")
lines.append("## 1. Purpose")
lines.append("")
lines.append("This report checks whether the current environment can directly import and instantiate WinCLIP from Anomalib.")
lines.append("")
lines.append("## 2. Import Results")
lines.append("")
lines.append("| Module | Attribute | Import | Instantiate | Error |")
lines.append("|---|---|---:|---:|---|")

for r in results:
    lines.append(
        f"| `{r['module']}` | `{r['attribute']}` | {r['import_success']} | "
        f"{r['instantiate_success']} | `{r.get('error', '')}` |"
    )

lines.append("")
lines.append("## 3. Decision")
lines.append("")

if any(r["import_success"] for r in results):
    lines.append("WinCLIP-related classes are available in the current environment.")
    lines.append("")
    lines.append("Next step: create a small Stage 14-C pilot script for one AD2 primary category.")
else:
    lines.append("WinCLIP was not found through the tested Anomalib import paths.")
    lines.append("")
    lines.append("Next step: inspect the installed Anomalib version and decide whether to upgrade Anomalib or use an external WinCLIP implementation.")

report_path.write_text("\n".join(lines), encoding="utf-8")

with txt_path.open("w", encoding="utf-8") as f:
    for r in results:
        f.write(str(r) + "\n")

print("[DONE]", report_path)
print("[DONE]", txt_path)
print("\n".join(lines))
