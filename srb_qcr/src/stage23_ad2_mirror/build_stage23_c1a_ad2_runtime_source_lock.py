from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/root/private_data/iad-vlm-anomaly").resolve()

CATEGORIES = [
    "fruit_jelly",
    "sheet_metal",
    "vial",
    "walnuts",
]

B2B = (
    ROOT
    / "results/stage22_selective_qcr"
    / "stage22_b2b_ad2_frozen_predictions.csv"
)

CANDIDATES = (
    ROOT
    / "results/stage11_mvtecad2_multicategory"
    / "stage11_d_vlm_candidate_scores.csv"
)

IMAGE_PREDICTIONS = (
    ROOT
    / "results/stage11_mvtecad2_multicategory"
    / "stage11_d_vlm_image_predictions.csv"
)

OUT_JSON = (
    ROOT
    / "results/stage23_ad2_mirror"
    / "stage23_c1a_ad2_runtime_source_lock.json"
)

OUT_TXT = (
    ROOT
    / "docs/stage23_ad2_mirror"
    / "stage23_c1a_ad2_runtime_source_lock.txt"
)

SEARCH_TERMS = [
    "stage11_d_vlm_candidate_scores.csv",
    "stage11_d_vlm_image_predictions.csv",
    "tight_vlm_margin",
    "context_vlm_margin",
    "clip_backend",
    "open_clip",
    "inspection_binary",
    "normal_prompts",
    "anomaly_prompts",
]


def normalize_path(value: Any) -> str:
    text = str(value).replace("\\", "/").strip()

    for marker in [
        "/datasets/MVTec_AD_2_anomalib_all/",
        "datasets/MVTec_AD_2_anomalib_all/",
        "/datasets/",
        "datasets/",
    ]:
        if marker in text:
            return text.split(marker, 1)[1]

    return text.removeprefix("./")


def function_names(path: Path) -> list[str]:
    try:
        tree = ast.parse(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except Exception:
        return []

    return [
        node.name
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    ]


def source_hits(path: Path) -> list[dict]:
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()
    hits = []

    for index, line in enumerate(lines):
        lowered = line.lower()

        matched = [
            term
            for term in SEARCH_TERMS
            if term.lower() in lowered
        ]

        if not matched:
            continue

        start = max(0, index - 4)
        end = min(len(lines), index + 7)

        hits.append(
            {
                "line": index + 1,
                "terms": matched,
                "context": [
                    f"{line_number + 1:5d}: {lines[line_number]}"
                    for line_number in range(
                        start,
                        end,
                    )
                ],
            }
        )

    return hits


def aggregate_candidate_scores(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "category",
        "image_path",
        "candidate_rank",
        "tight_vlm_margin",
        "context_vlm_margin",
    }

    missing = sorted(
        required - set(candidates.columns)
    )

    if missing:
        raise RuntimeError(
            f"Candidate score CSV missing {missing}"
        )

    frame = candidates.copy()

    for column in [
        "candidate_rank",
        "tight_vlm_margin",
        "context_vlm_margin",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame[
        frame["category"]
        .astype(str)
        .isin(CATEGORIES)
    ].copy()

    frame["path_key"] = (
        frame["image_path"]
        .astype(str)
        .map(normalize_path)
    )

    frame = frame[
        frame["candidate_rank"] > 0
    ].copy()

    if frame[
        [
            "candidate_rank",
            "tight_vlm_margin",
            "context_vlm_margin",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Candidate score rows contain missing values."
        )

    rows = []

    for (
        category,
        path_key,
    ), group in frame.groupby(
        [
            "category",
            "path_key",
        ],
        sort=False,
    ):
        group = group.sort_values(
            "candidate_rank"
        )

        tight = group[
            "tight_vlm_margin"
        ].to_numpy(dtype=float)

        context = group[
            "context_vlm_margin"
        ].to_numpy(dtype=float)

        rows.append(
            {
                "category": category,
                "path_key": path_key,
                "num_candidates": len(group),
                "tight_top1": float(tight[0]),
                "tight_topk_max": float(tight.max()),
                "tight_topk_mean": float(tight.mean()),
                "context_top1": float(context[0]),
                "context_topk_max": float(context.max()),
                "context_topk_mean": float(context.mean()),
            }
        )

    return pd.DataFrame(rows)


def compare_m_source(
    b2b: pd.DataFrame,
    aggregates: pd.DataFrame,
) -> list[dict]:
    merged = b2b.merge(
        aggregates,
        on=[
            "category",
            "path_key",
        ],
        how="left",
        validate="one_to_one",
    )

    available = merged[
        merged["M_available"].astype(bool)
    ].copy()

    candidates = [
        "tight_top1",
        "tight_topk_max",
        "tight_topk_mean",
        "context_top1",
        "context_topk_max",
        "context_topk_mean",
    ]

    rows = []

    for column in candidates:
        valid = available[
            [
                "M_raw_crop_topk",
                column,
            ]
        ].dropna()

        difference = (
            valid[column]
            - valid["M_raw_crop_topk"]
        ).abs()

        rows.append(
            {
                "candidate_source": column,
                "rows_compared": len(valid),
                "max_abs_diff": (
                    float(difference.max())
                    if len(valid)
                    else None
                ),
                "mean_abs_diff": (
                    float(difference.mean())
                    if len(valid)
                    else None
                ),
                "exact_within_1e_12": bool(
                    len(valid)
                    and difference.max()
                    <= 1e-12
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            float("inf")
            if row["max_abs_diff"] is None
            else row["max_abs_diff"]
        ),
    )


def crop_path_audit(
    candidates: pd.DataFrame,
) -> dict:
    path_columns = [
        "tight_crop_path",
        "context_1p50_crop_path",
    ]

    result = {}

    for column in path_columns:
        if column not in candidates.columns:
            result[column] = {
                "column_exists": False,
            }
            continue

        subset = candidates[
            candidates["category"]
            .astype(str)
            .isin(CATEGORIES)
        ].copy()

        paths = (
            subset[column]
            .dropna()
            .astype(str)
        )

        existence = paths.map(
            lambda value: Path(value).exists()
        )

        result[column] = {
            "column_exists": True,
            "nonempty_rows": int(len(paths)),
            "existing_rows": int(existence.sum()),
            "missing_rows": int((~existence).sum()),
            "all_exist": bool(
                len(paths)
                and existence.all()
            ),
            "examples": paths.head(5).tolist(),
        }

    return result


def main() -> None:
    for path in [
        B2B,
        CANDIDATES,
        IMAGE_PREDICTIONS,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    b2b = pd.read_csv(B2B)

    candidates = pd.read_csv(
        CANDIDATES
    )

    image_predictions = pd.read_csv(
        IMAGE_PREDICTIONS
    )

    b2b = b2b[
        b2b["category"]
        .astype(str)
        .isin(CATEGORIES)
    ].copy()

    b2b["path_key"] = (
        b2b["image_path"]
        .astype(str)
        .map(normalize_path)
    )

    if "M_available" not in b2b.columns:
        b2b["M_available"] = (
            pd.to_numeric(
                b2b["M_raw_crop_topk"],
                errors="coerce",
            ).notna()
        )

    else:
        numeric = pd.to_numeric(
            b2b["M_available"],
            errors="coerce",
        )

        if numeric.notna().all():
            b2b["M_available"] = numeric.gt(0)

        else:
            b2b["M_available"] = (
                b2b["M_available"]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "yes",
                    ]
                )
            )

    aggregates = aggregate_candidate_scores(
        candidates
    )

    comparisons = compare_m_source(
        b2b,
        aggregates,
    )

    source_inventory = []

    for path in (
        ROOT / "experiments"
    ).rglob("*.py"):
        hits = source_hits(path)

        if not hits:
            continue

        score = sum(
            len(hit["terms"])
            for hit in hits
        )

        source_inventory.append(
            {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "score": score,
                "functions": function_names(
                    path
                ),
                "hits": hits,
            }
        )

    source_inventory = sorted(
        source_inventory,
        key=lambda record: (
            -record["score"],
            record["path"],
        ),
    )

    clip_backends = (
        candidates["clip_backend"]
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
        if "clip_backend"
        in candidates.columns
        else {}
    )

    stage18_sources = {
        column: (
            b2b[column]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )
        for column in [
            "stage18_m_score_source",
            "stage18_q_score_source",
            "stage18_full_image_source",
            "stage18_note",
        ]
        if column in b2b.columns
    }

    payload = {
        "protocol_id": (
            "stage23_c1a_ad2_runtime_source_lock_v1"
        ),
        "rows": int(len(b2b)),
        "categories": CATEGORIES,
        "clip_backends": clip_backends,
        "stage18_sources": stage18_sources,
        "m_source_comparison": comparisons,
        "crop_path_audit": crop_path_audit(
            candidates
        ),
        "top_source_files": source_inventory[:12],
        "image_prediction_columns": list(
            image_predictions.columns
        ),
    }

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_TXT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "===== STAGE 23-C1a AD2 RUNTIME SOURCE LOCK =====",
        "",
        "===== LOCKED INPUTS =====",
        f"B2b rows: {len(b2b)}",
        f"categories: {sorted(b2b['category'].unique())}",
        f"CLIP backends: {clip_backends}",
        f"Stage 18 sources: {stage18_sources}",
        "",
        "===== M SOURCE COMPARISON =====",
    ]

    for row in comparisons:
        lines.append(
            (
                f"{row['candidate_source']}: "
                f"rows={row['rows_compared']}, "
                f"max_abs_diff={row['max_abs_diff']}, "
                f"mean_abs_diff={row['mean_abs_diff']}, "
                "exact_within_1e_12="
                f"{row['exact_within_1e_12']}"
            )
        )

    lines += [
        "",
        "===== CROP PATH AUDIT =====",
    ]

    for column, record in payload[
        "crop_path_audit"
    ].items():
        lines.append(
            f"{column}: {record}"
        )

    lines += [
        "",
        "===== TOP SOURCE FILES =====",
    ]

    for index, record in enumerate(
        source_inventory[:8],
        start=1,
    ):
        lines += [
            (
                f"[{index}] {record['path']} "
                f"score={record['score']}"
            ),
            (
                "  functions="
                + ", ".join(
                    record["functions"]
                )
            ),
        ]

        for hit in record["hits"][:8]:
            lines.append(
                (
                    f"  hit line {hit['line']}: "
                    f"{hit['terms']}"
                )
            )

            lines.extend(
                f"    {context}"
                for context in hit[
                    "context"
                ]
            )

    best = comparisons[0]

    ready = (
        best[
            "exact_within_1e_12"
        ]
        and any(
            record.get(
                "all_exist",
                False,
            )
            for record in payload[
                "crop_path_audit"
            ].values()
        )
        and bool(clip_backends)
        and bool(source_inventory)
    )

    lines += [
        "",
        "===== DECISION =====",
        (
            "exact_m_source_identified: "
            f"{best['exact_within_1e_12']}"
        ),
        (
            "selected_m_source: "
            f"{best['candidate_source']}"
        ),
        (
            "crop_assets_ready: "
            f"{any(record.get('all_exist', False) for record in payload['crop_path_audit'].values())}"
        ),
        (
            "clip_backend_identified: "
            f"{bool(clip_backends)}"
        ),
        (
            "generator_source_identified: "
            f"{bool(source_inventory)}"
        ),
        f"runtime_runner_ready: {ready}",
        "",
        f"[DONE] {OUT_JSON}",
        f"[DONE] {OUT_TXT}",
    ]

    OUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
