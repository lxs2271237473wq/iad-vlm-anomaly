from pathlib import Path
from PIL import Image
import csv
import json
import re

DATA_ROOT = Path("/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2")
OUT_CSV = Path("orbitad/data/manifests/private_pairs.csv")
OUT_JSON = Path("orbitad/data/metadata/private_pairs_summary.json")

CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]

REGULAR_RE = re.compile(r"^(\d+)_regular\.png$", re.I)
MIXED_RE = re.compile(r"^(\d+)_mixed\.png$", re.I)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

rows = []
summary = {}

for category in CATEGORIES:
    reg_dir = DATA_ROOT / category / "test_private"
    mix_dir = DATA_ROOT / category / "test_private_mixed"

    regular = {}
    mixed = {}

    for p in sorted(reg_dir.iterdir()):
        if p.is_file():
            m = REGULAR_RE.match(p.name)
            if m:
                regular[int(m.group(1))] = p

    for p in sorted(mix_dir.iterdir()):
        if p.is_file():
            m = MIXED_RE.match(p.name)
            if m:
                mixed[int(m.group(1))] = p

    if set(regular) != set(mixed):
        raise RuntimeError(f"{category}: regular/mixed ID mismatch")

    category_rows = 0

    for pair_id in sorted(regular):
        reg_path = regular[pair_id]
        mix_path = mixed[pair_id]

        with Image.open(reg_path) as im:
            reg_size = im.size

        with Image.open(mix_path) as im:
            mix_size = im.size

        if reg_size != mix_size:
            raise RuntimeError(
                f"{category}/{pair_id}: resolution mismatch "
                f"{reg_size} vs {mix_size}"
            )

        width, height = reg_size

        rows.append({
            "category": category,
            "pair_id": f"{pair_id:03d}",
            "regular_path": str(reg_path.relative_to(DATA_ROOT)),
            "mixed_path": str(mix_path.relative_to(DATA_ROOT)),
            "width": width,
            "height": height,
        })

        category_rows += 1

    summary[category] = {
        "pairs": category_rows,
    }

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "pair_id",
            "regular_path",
            "mixed_path",
            "width",
            "height",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

summary["total_pairs"] = len(rows)
summary["status"] = "PASS"

with OUT_JSON.open("w") as f:
    json.dump(summary, f, indent=2)

print("=" * 80)
print("OrbitAD A0.3d — PRIVATE PAIR MANIFEST")
print("=" * 80)

for category in CATEGORIES:
    print(f"{category:15s}: {summary[category]['pairs']:4d}")

print("-" * 80)
print(f"TOTAL PAIRS     : {len(rows)}")
print(f"CSV             : {OUT_CSV}")
print(f"SUMMARY         : {OUT_JSON}")
print("STATUS          : PASS")
