from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()
DATASET_ROOT = ROOT / "datasets" / "MVTec_AD_2"

OUT_CSV = ROOT / "results" / "stage10_dataset_expansion" / "stage10_b0_mvtecad2_layout_validation.csv"
OUT_MD = ROOT / "docs" / "stage10_dataset_expansion" / "stage10_b0_mvtecad2_layout_validation.md"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def inspect_tree() -> pd.DataFrame:
    rows = []

    rows.append({
        "level": "dataset_root",
        "path": str(DATASET_ROOT.relative_to(ROOT)),
        "exists": DATASET_ROOT.exists(),
        "dir_count": sum(1 for p in DATASET_ROOT.rglob("*") if p.is_dir()) if DATASET_ROOT.exists() else 0,
        "file_count": count_files(DATASET_ROOT),
        "image_count": count_images(DATASET_ROOT),
    })

    if DATASET_ROOT.exists():
        for child in sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()]):
            rows.append({
                "level": "top_level_dir",
                "path": str(child.relative_to(ROOT)),
                "exists": True,
                "dir_count": sum(1 for p in child.rglob("*") if p.is_dir()),
                "file_count": count_files(child),
                "image_count": count_images(child),
            })

            for sub in sorted([p for p in child.iterdir() if p.is_dir()]):
                rows.append({
                    "level": "second_level_dir",
                    "path": str(sub.relative_to(ROOT)),
                    "exists": True,
                    "dir_count": sum(1 for p in sub.rglob("*") if p.is_dir()),
                    "file_count": count_files(sub),
                    "image_count": count_images(sub),
                })

    return pd.DataFrame(rows)


def infer_status(df: pd.DataFrame) -> str:
    total_images = int(df[df["level"] == "dataset_root"]["image_count"].iloc[0])
    if not DATASET_ROOT.exists():
        return "missing_dataset_root"
    if total_images == 0:
        return "dataset_root_exists_but_no_images"
    if total_images < 100:
        return "images_found_but_probably_incomplete"
    return "images_found"


def write_report(df: pd.DataFrame, status: str) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    total_images = int(df[df["level"] == "dataset_root"]["image_count"].iloc[0])
    total_files = int(df[df["level"] == "dataset_root"]["file_count"].iloc[0])

    lines = []
    lines.append("# Stage 10-B0 MVTec AD 2 Layout Validation")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This stage validates whether MVTec AD 2 has been placed under the expected project path.")
    lines.append("It does not train models, run anomaly detectors, run VLM reasoning, or modify old results.")
    lines.append("")
    lines.append("## 2. Expected Dataset Root")
    lines.append("")
    lines.append(f"`{DATASET_ROOT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 3. Validation Status")
    lines.append("")
    lines.append(f"- Status: `{status}`")
    lines.append(f"- Total files: `{total_files}`")
    lines.append(f"- Total images: `{total_images}`")
    lines.append("")
    lines.append("## 4. Directory Summary")
    lines.append("")
    lines.append("| Level | Path | Exists | Dirs | Files | Images |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for _, r in df.iterrows():
        lines.append(
            f"| {r['level']} | `{r['path']}` | {r['exists']} | "
            f"{int(r['dir_count'])} | {int(r['file_count'])} | {int(r['image_count'])} |"
        )

    lines.append("")
    lines.append("## 5. Next Step")
    lines.append("")

    if status == "images_found":
        lines.append("MVTec AD 2 images are available. Next step: implement Stage 10-B1 manifest builder.")
    else:
        lines.append("MVTec AD 2 is not ready. Download and extract the dataset into the expected root before building the manifest.")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = inspect_tree()
    status = infer_status(df)

    df.to_csv(OUT_CSV, index=False)
    write_report(df, status)

    print("[DONE]", OUT_CSV)
    print("[DONE]", OUT_MD)
    print("status:", status)
    print("dataset_root:", DATASET_ROOT)
    print("total_images:", int(df[df["level"] == "dataset_root"]["image_count"].iloc[0]))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
