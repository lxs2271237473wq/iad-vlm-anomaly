from __future__ import annotations

import ast
import csv
import re
from pathlib import Path


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

CODE_ROOTS = [
    ROOT / "experiments/stage16_qcru_ablation",
    ROOT / "experiments/stage18_ad2_qcr_ablation",
]

OUTPUT = (
    ROOT
    / "docs/stage22_selective_qcr"
    / "stage22_a1c_compact_qcr_trace.md"
)

QUALITY_PATTERN = re.compile(
    r"\bquality\b|quality_score|candidate_quality|"
    r"quality_weight|quality_gate|\bq_score\b",
    re.IGNORECASE,
)

CONSISTENCY_PATTERN = re.compile(
    r"\bconsistency\b|consistency_score|agreement|"
    r"disagreement|adaptive_consistency|\bc_score\b",
    re.IGNORECASE,
)

QCR_PATTERN = re.compile(
    r"\bqcr\b|naive_fusion|quality_calibrated|"
    r"adaptive_refinement|\bV3\b|\bV4\b|\bV5\b|\bV6\b",
    re.IGNORECASE,
)

CSV_PATTERN = re.compile(
    r"""(?P<quote>["'])(?P<path>[^"' ]+\.csv)(?P=quote)"""
)

READ_CSV_PATTERN = re.compile(
    r"read_csv\s*\((.*?)\)",
    re.IGNORECASE,
)

VISA_REQUIRED_ALIASES = {
    "detector": [
        "patchcore_score",
        "fastflow_score",
        "detector_score",
        "base_score",
    ],
    "vlm": [
        "vlm_margin",
        "context_vlm_margin",
        "tight_vlm_margin",
        "vlm_score",
        "context_anomaly_score",
        "tight_anomaly_score",
    ],
    "quality": [
        "quality",
        "quality_score",
        "candidate_quality",
        "candidate_score_max",
        "candidate_score_mean",
        "candidate_mask_density",
    ],
    "label": [
        "gt_binary",
        "gt_label",
        "label",
        "target",
        "anomaly",
        "y_true",
    ],
    "path": [
        "image_path",
        "img_path",
        "filename",
    ],
    "category": [
        "category",
        "class_name",
        "cls_name",
        "object",
    ],
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_lines(path: Path) -> list[str]:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()


def merge_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not ranges:
        return []

    ranges = sorted(ranges)
    merged = [ranges[0]]

    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]

        if start <= previous_end + 1:
            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged.append((start, end))

    return merged


def matching_context(
    lines: list[str],
    pattern: re.Pattern,
    before: int = 4,
    after: int = 7,
    max_blocks: int = 12,
) -> list[tuple[int, int]]:
    ranges = []

    for index, line in enumerate(lines):
        if pattern.search(line):
            ranges.append(
                (
                    max(0, index - before),
                    min(len(lines), index + after + 1),
                )
            )

    return merge_ranges(ranges)[:max_blocks]


def format_code_blocks(
    path: Path,
    ranges: list[tuple[int, int]],
) -> list[str]:
    lines = read_lines(path)
    output = []

    for start, end in ranges:
        output.append(
            f"**`{rel(path)}`，第 {start + 1}–{end} 行**"
        )
        output.append("")
        output.append("```python")

        for index in range(start, end):
            output.append(
                f"{index + 1:4d}: {lines[index]}"
            )

        output.append("```")
        output.append("")

    return output


def ast_function_ranges(
    path: Path,
    pattern: re.Pattern,
    max_functions: int = 8,
) -> list[tuple[int, int, str]]:
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    output = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)

        function_text = "\n".join(
            text.splitlines()[start:end]
        )

        if pattern.search(function_text):
            output.append(
                (start, end, node.name)
            )

    output.sort()
    return output[:max_functions]


def format_functions(
    path: Path,
    functions: list[tuple[int, int, str]],
    max_lines_per_function: int = 80,
) -> list[str]:
    lines = read_lines(path)
    output = []

    for start, end, name in functions:
        original_end = end
        end = min(end, start + max_lines_per_function)

        output.append(
            f"**函数 `{name}`：`{rel(path)}`，"
            f"第 {start + 1}–{original_end} 行**"
        )
        output.append("")
        output.append("```python")

        for index in range(start, end):
            output.append(
                f"{index + 1:4d}: {lines[index]}"
            )

        if end < original_end:
            output.append(
                f"... 函数剩余 {original_end - end} 行已省略 ..."
            )

        output.append("```")
        output.append("")

    return output


def code_files() -> list[Path]:
    paths = []

    for root in CODE_ROOTS:
        if root.exists():
            paths.extend(root.rglob("*.py"))

    return sorted(set(paths))


def collect_csv_references(
    paths: list[Path],
) -> list[dict]:
    records = []

    for path in paths:
        lines = read_lines(path)

        for index, line in enumerate(lines):
            if (
                "read_csv" not in line
                and ".csv" not in line
                and "input_csv" not in line
                and "source_csv" not in line
            ):
                continue

            start = max(0, index - 2)
            end = min(len(lines), index + 4)

            snippet = "\n".join(
                f"{i + 1:4d}: {lines[i]}"
                for i in range(start, end)
            )

            literal_paths = [
                match.group("path")
                for match in CSV_PATTERN.finditer(
                    "\n".join(lines[start:end])
                )
            ]

            records.append(
                {
                    "file": rel(path),
                    "line": index + 1,
                    "literal_paths": literal_paths,
                    "snippet": snippet,
                }
            )

    deduplicated = []
    seen = set()

    for record in records:
        key = (
            record["file"],
            record["snippet"],
        )

        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(record)

    return deduplicated[:30]


def count_csv_rows(path: Path) -> int | None:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            return max(
                sum(1 for _ in handle) - 1,
                0,
            )
    except Exception:
        return None


def detect_groups(
    columns: list[str],
) -> dict[str, list[str]]:
    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    groups = {}

    for group, aliases in VISA_REQUIRED_ALIASES.items():
        matches = []

        for column_lower, original in normalized.items():
            for alias in aliases:
                if (
                    column_lower == alias
                    or alias in column_lower
                ):
                    matches.append(original)
                    break

        if matches:
            groups[group] = sorted(set(matches))

    return groups


def visa_candidates() -> list[dict]:
    records = []

    results_root = ROOT / "results"

    if not results_root.exists():
        return []

    for path in results_root.rglob("*.csv"):
        path_text = str(path).lower()

        if (
            "visa" not in path_text
            and "stage16" not in path_text
        ):
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as handle:
                reader = csv.reader(handle)
                columns = next(reader, [])
        except Exception:
            continue

        groups = detect_groups(columns)

        if not groups:
            continue

        score = (
            8 * int("detector" in groups)
            + 8 * int("vlm" in groups)
            + 7 * int("quality" in groups)
            + 8 * int("label" in groups)
            + 4 * int("path" in groups)
            + 3 * int("category" in groups)
        )

        records.append(
            {
                "path": rel(path),
                "score": score,
                "size_bytes": path.stat().st_size,
                "row_count": count_csv_rows(path),
                "groups": groups,
                "columns": columns,
            }
        )

    records.sort(
        key=lambda record: (
            -record["score"],
            -record["size_bytes"],
            record["path"],
        )
    )

    return records[:12]


def main() -> None:
    paths = code_files()

    report = [
        "# Stage 22-A1c：旧 QCR 与 VisA 缓存精简追踪",
        "",
        "该报告只提取后续冻结 Selective QCR 协议所需的信息。",
        "",
        f"- 扫描 Python 文件：`{len(paths)}`",
        "- 未运行检测器或 VLM",
        "- 未修改已有实验结果",
        "",
    ]

    report += [
        "## 1. 旧 QCR：Quality 计算代码",
        "",
    ]

    quality_found = False

    for path in paths:
        functions = ast_function_ranges(
            path,
            QUALITY_PATTERN,
        )

        if functions:
            quality_found = True
            report.extend(
                format_functions(path, functions)
            )

    if not quality_found:
        report.append(
            "未发现独立的 Quality 函数，下面给出关键词上下文。"
        )
        report.append("")

        for path in paths:
            ranges = matching_context(
                read_lines(path),
                QUALITY_PATTERN,
            )

            if ranges:
                report.extend(
                    format_code_blocks(path, ranges)
                )

    report += [
        "## 2. 旧 QCR：Consistency 计算代码",
        "",
    ]

    consistency_found = False

    for path in paths:
        functions = ast_function_ranges(
            path,
            CONSISTENCY_PATTERN,
        )

        if functions:
            consistency_found = True
            report.extend(
                format_functions(path, functions)
            )

    if not consistency_found:
        report.append(
            "未发现独立的 Consistency 函数，下面给出关键词上下文。"
        )
        report.append("")

        for path in paths:
            ranges = matching_context(
                read_lines(path),
                CONSISTENCY_PATTERN,
            )

            if ranges:
                report.extend(
                    format_code_blocks(path, ranges)
                )

    report += [
        "## 3. 旧 QCR 与 V3–V6 融合公式上下文",
        "",
    ]

    for path in paths:
        ranges = matching_context(
            read_lines(path),
            QCR_PATTERN,
            before=4,
            after=8,
            max_blocks=8,
        )

        if ranges:
            report.extend(
                format_code_blocks(path, ranges)
            )

    report += [
        "## 4. Stage 16 输入 CSV 路径",
        "",
    ]

    csv_records = collect_csv_references(paths)

    if not csv_records:
        report.append("未发现 CSV 路径引用。")
        report.append("")
    else:
        for record in csv_records:
            report.append(
                f"### `{record['file']}`，第 {record['line']} 行附近"
            )
            report.append("")

            if record["literal_paths"]:
                report.append(
                    "- 字符串形式 CSV："
                    + ", ".join(
                        f"`{value}`"
                        for value in record["literal_paths"]
                    )
                )
                report.append("")

            report.append("```python")
            report.append(record["snippet"])
            report.append("```")
            report.append("")

    report += [
        "## 5. VisA 样本级缓存候选",
        "",
        "| 排名 | 文件 | 行数 | 大小 MiB | 字段组 |",
        "|---:|---|---:|---:|---|",
    ]

    visa_records = visa_candidates()

    for index, record in enumerate(
        visa_records,
        start=1,
    ):
        groups_text = "; ".join(
            f"{group}={','.join(fields)}"
            for group, fields in record["groups"].items()
        )

        report.append(
            f"| {index} | `{record['path']}` | "
            f"{record['row_count']} | "
            f"{record['size_bytes'] / 1024**2:.3f} | "
            f"{groups_text} |"
        )

    report += [
        "",
        "### VisA 候选完整字段",
        "",
    ]

    for index, record in enumerate(
        visa_records[:6],
        start=1,
    ):
        report.append(
            f"#### {index}. `{record['path']}`"
        )
        report.append("")
        report.append("```text")
        report.append(", ".join(record["columns"]))
        report.append("```")
        report.append("")

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        "\n".join(report),
        encoding="utf-8",
        newline="\n",
    )

    print("[DONE]", OUTPUT)
    print("python files scanned:", len(paths))
    print("CSV references:", len(csv_records))
    print("VisA candidates:", len(visa_records))
    print()
    print("查看精简报告：")
    print(f"sed -n '1,260p' {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
