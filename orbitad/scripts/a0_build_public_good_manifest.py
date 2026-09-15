from pathlib import Path
import csv
import re
from collections import Counter

DATA_ROOT = Path(
    "/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2"
)

OUT = Path(
    "orbitad/data/manifests/test_public_good.csv"
)

OUT.parent.mkdir(parents=True, exist_ok=True)

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

EXTS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
}

rows = []
group_counter = Counter()

for category in CATEGORIES:

    root = (
        DATA_ROOT
        / category
        / "test_public"
        / "good"
    )

    for path in sorted(root.iterdir()):

        if (
            not path.is_file()
            or path.suffix.lower() not in EXTS
        ):
            continue

        m = re.match(
            r"^(\d+)_(.+)$",
            path.stem.lower()
        )

        if not m:
            raise RuntimeError(
                f"Cannot parse filename: {path}"
            )

        instance_id = m.group(1)
        condition = m.group(2)

        rows.append({
            "category": category,
            "instance_id": instance_id,
            "condition": condition,
            "image_path": str(
                path.relative_to(DATA_ROOT)
            ),
        })

        group_counter[
            (category, instance_id)
        ] += 1


with OUT.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "instance_id",
            "condition",
            "image_path",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print("=" * 80)
print("OrbitAD — PUBLIC GOOD ORBIT MANIFEST")
print("=" * 80)

print("images             :", len(rows))
print("physical instances :", len(group_counter))

hist = Counter(group_counter.values())

for n, count in sorted(hist.items()):
    print(
        f"{n} conditions/instance : "
        f"{count} instances"
    )

print("output             :", OUT)

assert len(rows) == 379
assert len(group_counter) == 64

print("STATUS             : PASS")
