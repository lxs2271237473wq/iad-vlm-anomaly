from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path("/root/private_data/iad-vlm-anomaly").resolve()
REPO = Path("/root/private_data/third_party/AnomalyCLIP").resolve()
DATA_ROOT = Path("/root/private_data/anomalyclip_data/ad2four").resolve()

SMOKE_JSON = (
    PROJECT
    / "results/stage20_anomalyclip_baseline"
    / "stage20_c3_single_image_smoke.json"
)

OUT_DIR = PROJECT / "results/stage20_anomalyclip_baseline"
DOC_DIR = PROJECT / "docs/stage20_anomalyclip_baseline"
LOG_DIR = PROJECT / "logs"

OFFICIAL_OUT_DIR = OUT_DIR / "stage20_d_official_output"
OUT_LOG = LOG_DIR / "stage20_d_anomalyclip_ad2four_full.log"
OUT_RAW_JSON = OUT_DIR / "stage20_d_anomalyclip_ad2four_full_raw.json"
OUT_METRICS = OUT_DIR / "stage20_d_anomalyclip_ad2four_metrics.csv"
OUT_COMPARISON = OUT_DIR / "stage20_d_anomalyclip_system_comparison.csv"
OUT_REPORT = DOC_DIR / "stage20_d_anomalyclip_ad2four_report.md"

EXPECTED_CATEGORIES = [
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "walnuts",
]

SYSTEM_BASELINES = [
    {
        "method": "WinCLIP fixed protocol",
        "image_AUROC": 0.6138,
        "role": "existing external VLM anomaly baseline",
    },
    {
        "method": "full-image VLM",
        "image_AUROC": 0.6459,
        "role": "full-image VLM baseline",
    },
    {
        "method": "context-aware VLM",
        "image_AUROC": 0.7101,
        "role": "context-aware VLM baseline",
    },
    {
        "method": "EfficientAD-30 fixed-budget",
        "image_AUROC": 0.7604,
        "role": "modern detector fixed-budget baseline",
    },
    {
        "method": "PatchCore",
        "image_AUROC": 0.7853,
        "role": "classic detector baseline",
    },
    {
        "method": "PatchCore + context VLM, LOCO",
        "image_AUROC": 0.8210,
        "role": "primary fair system result",
    },
    {
        "method": "PatchCore + context VLM, same-set",
        "image_AUROC": 0.8453,
        "role": "upper-bound diagnostic only",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def resolve_checkpoint(explicit: str | None) -> Path:
    if explicit:
        checkpoint = Path(explicit).expanduser().resolve()
    else:
        if not SMOKE_JSON.exists():
            raise FileNotFoundError(
                "Smoke-test JSON not found and no --checkpoint supplied: "
                f"{SMOKE_JSON}"
            )

        smoke = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))
        checkpoint_value = smoke.get("checkpoint_path")

        if not checkpoint_value:
            raise RuntimeError(
                f"checkpoint_path missing from {SMOKE_JSON}"
            )

        checkpoint = Path(checkpoint_value).expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    return checkpoint


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def parse_markdown_table(log_text: str) -> list[dict]:
    """
    Parse the official tabulate pipe table:

    | objects | pixel_auroc | pixel_aupro | image_auroc | image_ap |
    | ...     | ...         | ...         | ...         | ...      |
    """
    rows: list[dict] = []

    for line in log_text.splitlines():
        stripped = line.strip()

        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]

        if len(cells) < 5:
            continue

        name = cells[0]

        if name in {"objects", "object", ""}:
            continue

        if set(name) <= {":", "-"}:
            continue

        if name not in EXPECTED_CATEGORIES + ["mean"]:
            continue

        try:
            pixel_auroc_percent = float(cells[1])
            pixel_aupro_percent = float(cells[2])
            image_auroc_percent = float(cells[3])
            image_ap_percent = float(cells[4])
        except ValueError:
            continue

        rows.append(
            {
                "category": name,
                "pixel_AUROC_percent": pixel_auroc_percent,
                "pixel_AUPRO_percent": pixel_aupro_percent,
                "image_AUROC_percent": image_auroc_percent,
                "image_AP_percent": image_ap_percent,
                "pixel_AUROC": pixel_auroc_percent / 100.0,
                "pixel_AUPRO": pixel_aupro_percent / 100.0,
                "image_AUROC": image_auroc_percent / 100.0,
                "image_AP": image_ap_percent / 100.0,
            }
        )

    # The log may contain repeated tables. Keep the final row for each category.
    deduplicated: dict[str, dict] = {}
    for row in rows:
        deduplicated[row["category"]] = row

    ordered = []
    for category in EXPECTED_CATEGORIES + ["mean"]:
        if category in deduplicated:
            ordered.append(deduplicated[category])

    return ordered


def validate_rows(rows: list[dict]) -> None:
    found = {row["category"] for row in rows}
    expected = set(EXPECTED_CATEGORIES + ["mean"])

    missing = expected - found
    if missing:
        raise RuntimeError(
            f"Could not parse complete official result table. Missing: "
            f"{sorted(missing)}. Inspect {OUT_LOG}"
        )

    for row in rows:
        for metric in [
            "pixel_AUROC",
            "pixel_AUPRO",
            "image_AUROC",
            "image_AP",
        ]:
            value = row[metric]
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"Invalid metric: {row['category']} {metric}={value}"
                )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_comparison(mean_image_auroc: float) -> list[dict]:
    rows = []

    rows.append(
        {
            "method": "AnomalyCLIP fixed checkpoint",
            "image_AUROC": mean_image_auroc,
            "delta_vs_anomalyclip": 0.0,
            "role": "new external VLM anomaly baseline",
        }
    )

    for baseline in SYSTEM_BASELINES:
        rows.append(
            {
                "method": baseline["method"],
                "image_AUROC": baseline["image_AUROC"],
                "delta_vs_anomalyclip": (
                    baseline["image_AUROC"] - mean_image_auroc
                ),
                "role": baseline["role"],
            }
        )

    rows.sort(key=lambda x: x["image_AUROC"], reverse=True)
    return rows


def write_report(
    rows: list[dict],
    comparison: list[dict],
    checkpoint: Path,
    elapsed_sec: float,
    repo_commit: str,
) -> None:
    mean_row = next(row for row in rows if row["category"] == "mean")

    lines = [
        "# Stage 20-D: AnomalyCLIP AD2-four Full Evaluation",
        "",
        "## Protocol",
        "",
        "- implementation: official `zqhang/AnomalyCLIP`",
        f"- repository commit: `{repo_commit}`",
        "- dataset: `ad2four`",
        "- categories: `fruit_jelly`, `sheet_metal`, `vial`, `walnuts`",
        "- number of test images: `484`",
        "- checkpoint: fixed checkpoint verified in Stage 20-C3",
        f"- checkpoint path: `{checkpoint}`",
        f"- checkpoint SHA-256: `{sha256(checkpoint)}`",
        "- model: `ViT-L/14@336px`",
        "- image size: `518`",
        "- feature layers: `6, 12, 18, 24`",
        "- DPAM layer: `20`",
        "- metrics: image AUROC, image AP, pixel AUROC, pixel AUPRO",
        "- additional training on AD2: `none`",
        "",
        "## Per-category results",
        "",
        "| Category | Image AUROC | Image AP | Pixel AUROC | Pixel AUPRO |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['category']} | "
            f"{row['image_AUROC']:.4f} | "
            f"{row['image_AP']:.4f} | "
            f"{row['pixel_AUROC']:.4f} | "
            f"{row['pixel_AUPRO']:.4f} |"
        )

    lines += [
        "",
        "## System-level image-AUROC comparison",
        "",
        "| Method | Image AUROC | Difference relative to AnomalyCLIP | Role |",
        "|---|---:|---:|---|",
    ]

    for row in comparison:
        lines.append(
            f"| {row['method']} | "
            f"{row['image_AUROC']:.4f} | "
            f"{row['delta_vs_anomalyclip']:+.4f} | "
            f"{row['role']} |"
        )

    lines += [
        "",
        "## Runtime",
        "",
        f"- complete evaluation time: `{elapsed_sec:.2f}` seconds",
        "",
        "## Claim restrictions",
        "",
        "- This is a fixed-checkpoint evaluation with no AD2-specific tuning.",
        "- The comparison should be described as an external baseline under the",
        "  implemented fixed protocol.",
        "- Do not claim universal superiority over AnomalyCLIP based on one",
        "  adapted AD2 protocol.",
        "- The same-set fusion remains an upper-bound diagnostic only.",
        "",
        "## Main result",
        "",
        f"- AnomalyCLIP mean image AUROC: `{mean_row['image_AUROC']:.4f}`",
        f"- AnomalyCLIP mean image AP: `{mean_row['image_AP']:.4f}`",
        f"- AnomalyCLIP mean pixel AUROC: `{mean_row['pixel_AUROC']:.4f}`",
        f"- AnomalyCLIP mean pixel AUPRO: `{mean_row['pixel_AUPRO']:.4f}`",
        "",
    ]

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_checkpoint(args.checkpoint)

    required = [
        REPO / "test.py",
        REPO / "dataset.py",
        DATA_ROOT / "meta.json",
        checkpoint,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    command = [
        sys.executable,
        str(REPO / "test.py"),
        "--data_path",
        str(DATA_ROOT),
        "--save_path",
        str(OFFICIAL_OUT_DIR),
        "--checkpoint_path",
        str(checkpoint),
        "--dataset",
        "ad2four",
        "--features_list",
        "6",
        "12",
        "18",
        "24",
        "--image_size",
        "518",
        "--depth",
        "9",
        "--n_ctx",
        "12",
        "--t_n_ctx",
        "4",
        "--feature_map_layer",
        "0",
        "1",
        "2",
        "3",
        "--metrics",
        "image-pixel-level",
        "--seed",
        "111",
        "--sigma",
        "4",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["CUDA_VISIBLE_DEVICES"] = env.get(
        "CUDA_VISIBLE_DEVICES",
        "0",
    )

    print("===== Stage 20-D command =====")
    print(" ".join(command))
    print()
    print("[CHECKPOINT]", checkpoint)
    print("[CHECKPOINT SHA256]", sha256(checkpoint))
    print()

    started = time.perf_counter()

    process = subprocess.Popen(
        command,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines = []

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        captured_lines.append(line)

    return_code = process.wait()
    elapsed_sec = time.perf_counter() - started
    log_text = "".join(captured_lines)

    OUT_LOG.write_text(
        log_text,
        encoding="utf-8",
        errors="replace",
    )

    raw = {
        "status": "success" if return_code == 0 else "failed",
        "return_code": return_code,
        "command": command,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "repository_commit": git_commit(REPO),
        "elapsed_sec": elapsed_sec,
        "log_path": str(OUT_LOG),
    }

    OUT_RAW_JSON.write_text(
        json.dumps(raw, indent=2),
        encoding="utf-8",
    )

    if return_code != 0:
        raise RuntimeError(
            f"Official AnomalyCLIP test failed with exit code "
            f"{return_code}. Inspect {OUT_LOG}"
        )

    rows = parse_markdown_table(log_text)
    validate_rows(rows)
    write_csv(OUT_METRICS, rows)

    mean_row = next(row for row in rows if row["category"] == "mean")
    comparison = build_comparison(mean_row["image_AUROC"])
    write_csv(OUT_COMPARISON, comparison)

    write_report(
        rows=rows,
        comparison=comparison,
        checkpoint=checkpoint,
        elapsed_sec=elapsed_sec,
        repo_commit=git_commit(REPO),
    )

    print()
    print("===== Stage 20-D SUCCESS =====")
    print()
    for row in rows:
        print(
            f"{row['category']:12s} "
            f"image_AUROC={row['image_AUROC']:.4f} "
            f"image_AP={row['image_AP']:.4f} "
            f"pixel_AUROC={row['pixel_AUROC']:.4f} "
            f"pixel_AUPRO={row['pixel_AUPRO']:.4f}"
        )

    print()
    print("[DONE]", OUT_METRICS)
    print("[DONE]", OUT_COMPARISON)
    print("[DONE]", OUT_RAW_JSON)
    print("[DONE]", OUT_REPORT)


if __name__ == "__main__":
    main()
