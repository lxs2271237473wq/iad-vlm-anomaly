from __future__ import annotations

import re
from pathlib import Path


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

SEARCH_ROOTS = [
    ROOT / "experiments/stage7_generalization",
    ROOT / "experiments/stage9_qcr_u",
    ROOT / "experiments/stage16_qcru_ablation",
]

OUT = (
    ROOT
    / "docs/stage22_selective_qcr"
    / "stage22_c2a_runtime_path_audit.md"
)

PATTERNS = {
    "prediction_outputs": re.compile(
        r"visa_binary_prompt_predictions|"
        r"stage9_a1_qcr_u_fusion_predictions|"
        r"OUT_.*PRED|prediction.*csv",
        re.I,
    ),
    "model_loading": re.compile(
        r"load_model|from_pretrained|clip\.load|"
        r"AutoModel|AutoProcessor|model\s*=|processor\s*=",
        re.I,
    ),
    "inference": re.compile(
        r"inference_mode|no_grad|generate\(|"
        r"encode_image|encode_text|predict|"
        r"vlm_anomaly_score",
        re.I,
    ),
    "iteration": re.compile(
        r"DataLoader|batch_size|num_workers|"
        r"for\s+.*\s+in\s+.*loader|"
        r"for\s+.*image|iterrows\(",
        re.I,
    ),
    "timing": re.compile(
        r"perf_counter|time\.time|cuda\.Event|"
        r"synchronize|elapsed|latency|runtime",
        re.I,
    ),
    "device": re.compile(
        r"cuda|CUDA_VISIBLE_DEVICES|device|"
        r"float16|half\(|autocast",
        re.I,
    ),
}


def merge_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not ranges:
        return []

    ranges = sorted(ranges)
    merged = [ranges[0]]

    for start, end in ranges[1:]:
        old_start, old_end = merged[-1]

        if start <= old_end + 1:
            merged[-1] = (
                old_start,
                max(old_end, end),
            )
        else:
            merged.append((start, end))

    return merged


def relevant_files() -> list[Path]:
    files = []

    for root in SEARCH_ROOTS:
        if root.exists():
            files.extend(root.rglob("*.py"))

    return sorted(set(files))


def contexts(
    lines: list[str],
    pattern: re.Pattern,
    before: int = 4,
    after: int = 7,
    maximum: int = 12,
) -> list[tuple[int, int]]:
    found = []

    for index, line in enumerate(lines):
        if pattern.search(line):
            found.append(
                (
                    max(0, index - before),
                    min(len(lines), index + after + 1),
                )
            )

    return merge_ranges(found)[:maximum]


def main() -> None:
    files = relevant_files()

    report = [
        "# Stage 22-C2a: Runtime Path Audit",
        "",
        "## Purpose",
        "",
        "Identify the existing VisA VLM inference path before",
        "implementing actual selective invocation and timing.",
        "",
        f"- Python files scanned: `{len(files)}`",
        "- model execution: `none`",
        "- GPU use: `none`",
        "",
    ]

    for section, pattern in PATTERNS.items():
        report += [
            f"## {section}",
            "",
        ]

        count = 0

        for path in files:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

            ranges = contexts(lines, pattern)

            if not ranges:
                continue

            for start, end in ranges:
                count += 1

                report += [
                    (
                        f"### `{path.relative_to(ROOT)}` "
                        f"lines {start + 1}–{end}"
                    ),
                    "",
                    "```python",
                ]

                for line_number in range(start, end):
                    report.append(
                        f"{line_number + 1:4d}: "
                        f"{lines[line_number]}"
                    )

                report += [
                    "```",
                    "",
                ]

        if count == 0:
            report += [
                "No matching code found.",
                "",
            ]

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        "\n".join(report),
        encoding="utf-8",
        newline="\n",
    )

    print("[DONE]", OUT)
    print("files scanned:", len(files))
    print()
    print("查看标题：")
    print(
        "grep -nE '^## |^### ' "
        + str(OUT.relative_to(ROOT))
    )


if __name__ == "__main__":
    main()
