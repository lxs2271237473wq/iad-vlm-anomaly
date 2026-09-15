from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import csv

DATA_ROOT = Path("/root/private_data/iad-vlm-anomaly/datasets/MVTec_AD_2")
MANIFEST = Path("orbitad/data/manifests/private_pairs.csv")
OUT_ROOT = Path("orbitad/results/a0_dataset_audit/private_pair_visual_check")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

by_category = {}

with MANIFEST.open() as f:
    for row in csv.DictReader(f):
        by_category.setdefault(row["category"], []).append(row)

for category, rows in by_category.items():
    rows = sorted(rows, key=lambda x: int(x["pair_id"]))

    indices = sorted(set([
        0,
        len(rows) // 2,
        len(rows) - 1,
    ]))

    selected = [rows[i] for i in indices]

    thumb_w = 500
    thumb_h = 400
    label_h = 40

    canvas = Image.new(
        "RGB",
        (thumb_w * 2, (thumb_h + label_h) * len(selected)),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for row_idx, row in enumerate(selected):
        reg = Image.open(DATA_ROOT / row["regular_path"]).convert("RGB")
        mix = Image.open(DATA_ROOT / row["mixed_path"]).convert("RGB")

        reg = ImageOps.contain(reg, (thumb_w, thumb_h))
        mix = ImageOps.contain(mix, (thumb_w, thumb_h))

        y0 = row_idx * (thumb_h + label_h)

        reg_x = (thumb_w - reg.width) // 2
        mix_x = thumb_w + (thumb_w - mix.width) // 2

        canvas.paste(reg, (reg_x, y0))
        canvas.paste(mix, (mix_x, y0))

        label = (
            f"{category} | ID {row['pair_id']} | "
            f"LEFT=regular | RIGHT=mixed"
        )

        draw.text((10, y0 + thumb_h + 10), label, fill="black")

    out = OUT_ROOT / f"{category}.jpg"
    canvas.save(out, quality=90)

    print(out)

print("\nVisual audit images generated.")
