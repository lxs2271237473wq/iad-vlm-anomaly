from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/root/private_data/iad-vlm-anomaly').resolve()
INPUT = (
    ROOT / 'results/stage22_selective_qcr'
    / 'mvtec15_srb_qcr_transfer'
    / 'stage22_d6b_mvtec15_unified_predictions.csv'
)
OUT_DIR = ROOT / 'results/stage22_selective_qcr/mvtec15_case_analysis'
OUT_CSV = OUT_DIR / 'stage22_d7b_label_aware_case_inventory.csv'
OUT_SUMMARY = OUT_DIR / 'stage22_d7b_label_aware_case_summary.json'
OUT_REPORT = (
    ROOT / 'docs/stage22_selective_qcr'
    / 'stage22_d7b_label_aware_case_inventory.md'
)

REQUIRED = [
    'category', 'image_path', 'Y', 'D', 'M', 'Q',
    'score_D0', 'score_V3', 'score_V4', 'score_V6',
    'score_S1', 'srb_pre_gate', 'srb_weight',
]


def select_cases(
    frame: pd.DataFrame,
    case_type: str,
    sort_column: str,
    ascending: bool,
    count: int = 5,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    selected = (
        frame.sort_values(
            [sort_column, 'category', 'image_path'],
            ascending=[ascending, True, True],
        )
        .head(count)
        .copy()
    )
    selected.insert(0, 'case_type', case_type)
    selected.insert(1, 'case_rank', range(1, len(selected) + 1))
    return selected


if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)
missing = [column for column in REQUIRED if column not in df.columns]
if missing:
    raise RuntimeError(f'Missing columns: {missing}')

for column in REQUIRED[2:]:
    df[column] = pd.to_numeric(df[column], errors='coerce')

if df[REQUIRED[2:]].isna().any().any():
    bad = df[df[REQUIRED[2:]].isna().any(axis=1)].head(20)
    raise RuntimeError(
        'The unified prediction table contains missing numeric values:\n'
        + bad.to_string(index=False)
    )

labels = sorted(df['Y'].astype(int).unique().tolist())
if labels != [0, 1]:
    raise RuntimeError(f'Expected binary labels [0, 1], found {labels}')

# Increasing anomaly score is useful for anomalies, harmful for normal images.
df['label_orientation'] = np.where(df['Y'].astype(int) == 1, 1.0, -1.0)

df['srb_benefit_vs_detector'] = (
    df['label_orientation'] * (df['score_S1'] - df['score_D0'])
)
df['srb_benefit_vs_naive'] = (
    df['label_orientation'] * (df['score_S1'] - df['score_V3'])
)
df['srb_benefit_vs_old_quality'] = (
    df['label_orientation'] * (df['score_S1'] - df['score_V4'])
)
df['naive_harm_vs_detector'] = (
    df['label_orientation'] * (df['score_D0'] - df['score_V3'])
)
df['old_quality_harm_vs_detector'] = (
    df['label_orientation'] * (df['score_D0'] - df['score_V4'])
)
df['vlm_directional_advantage'] = (
    df['label_orientation'] * (df['M'] - df['D'])
)
df['detector_vlm_disagreement'] = (df['D'] - df['M']).abs()

gate_on = df[df['srb_pre_gate'] > 0].copy()
gate_off = df[df['srb_pre_gate'] <= 0].copy()

cases = [
    select_cases(
        df[df['srb_benefit_vs_detector'] > 0],
        'srb_success_vs_detector',
        'srb_benefit_vs_detector',
        False,
    ),
    select_cases(
        df[df['srb_benefit_vs_detector'] < 0],
        'srb_failure_vs_detector',
        'srb_benefit_vs_detector',
        True,
    ),
    select_cases(
        df[df['naive_harm_vs_detector'] > 0],
        'largest_naive_harm',
        'naive_harm_vs_detector',
        False,
    ),
    select_cases(
        df[df['srb_benefit_vs_naive'] > 0],
        'largest_srb_repair_over_naive',
        'srb_benefit_vs_naive',
        False,
    ),
    select_cases(
        df[df['srb_benefit_vs_old_quality'] > 0],
        'largest_srb_repair_over_old_quality',
        'srb_benefit_vs_old_quality',
        False,
    ),
    select_cases(
        gate_off[gate_off['naive_harm_vs_detector'] > 0],
        'gate_off_protected_from_harmful_vlm',
        'naive_harm_vs_detector',
        False,
    ),
    select_cases(
        gate_off[gate_off['vlm_directional_advantage'] > 0],
        'gate_off_missed_vlm_opportunity',
        'vlm_directional_advantage',
        False,
    ),
    select_cases(
        gate_on[gate_on['srb_benefit_vs_detector'] > 0],
        'gate_on_helpful',
        'srb_benefit_vs_detector',
        False,
    ),
    select_cases(
        gate_on[gate_on['srb_benefit_vs_detector'] < 0],
        'gate_on_harmful',
        'srb_benefit_vs_detector',
        True,
    ),
]

cases = [frame for frame in cases if not frame.empty]
if not cases:
    raise RuntimeError('No label-aware cases were selected.')

inventory = pd.concat(cases, ignore_index=True)
output_columns = [
    'case_type', 'case_rank', 'category', 'image_path', 'Y',
    'label_orientation', 'D', 'M', 'Q', 'score_D0', 'score_V3',
    'score_V4', 'score_V6', 'score_S1', 'srb_pre_gate', 'srb_weight',
    'srb_benefit_vs_detector', 'srb_benefit_vs_naive',
    'srb_benefit_vs_old_quality', 'naive_harm_vs_detector',
    'old_quality_harm_vs_detector', 'vlm_directional_advantage',
    'detector_vlm_disagreement',
]
inventory = inventory[output_columns]

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
inventory.to_csv(OUT_CSV, index=False, lineterminator='\n')

summary = {
    'source': str(INPUT.relative_to(ROOT)),
    'source_rows': len(df),
    'source_categories': int(df['category'].nunique()),
    'inventory_rows': len(inventory),
    'selection_rule': (
        'Label-aware direction: +1 for anomaly images and -1 for normal images.'
    ),
    'case_types': inventory['case_type'].value_counts().sort_index().to_dict(),
    'category_counts': inventory['category'].value_counts().sort_index().to_dict(),
    'largest_srb_benefit_vs_detector': float(df['srb_benefit_vs_detector'].max()),
    'worst_srb_benefit_vs_detector': float(df['srb_benefit_vs_detector'].min()),
    'largest_naive_harm_vs_detector': float(df['naive_harm_vs_detector'].max()),
    'largest_srb_repair_over_old_quality': float(
        df['srb_benefit_vs_old_quality'].max()
    ),
    'gate_on_rows': int((df['srb_pre_gate'] > 0).sum()),
    'gate_off_rows': int((df['srb_pre_gate'] <= 0).sum()),
}
OUT_SUMMARY.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

lines = [
    '# Stage 22-D7b: Label-Aware Case Inventory',
    '',
    'The earlier D7a inventory ranked raw anomaly-score changes without '
    'accounting for the true image label. A score increase is beneficial '
    'for anomaly images but harmful for normal images. This corrected '
    'inventory uses:',
    '',
    '```text',
    'orientation = +1 for anomaly images',
    'orientation = -1 for normal images',
    'label-aware benefit(A vs B) = orientation * (score_A - score_B)',
    '```',
    '',
    f'- source rows: `{len(df)}`',
    f'- categories: `{df["category"].nunique()}`',
    f'- selected rows: `{len(inventory)}`',
    '',
    '## Case counts',
    '',
    '| Case type | Count |',
    '|---|---:|',
]
for case_type, count in inventory['case_type'].value_counts().sort_index().items():
    lines.append(f'| {case_type} | {int(count)} |')

lines += [
    '',
    '## Top representative cases',
    '',
    '| Case type | Rank | Category | Y | D | M | Q | SRB | '
    'Benefit vs detector | Image |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|---|',
]
for _, row in inventory.groupby('case_type', sort=True).head(3).iterrows():
    lines.append(
        f'| {row["case_type"]} | {int(row["case_rank"])} | '
        f'{row["category"]} | {int(row["Y"])} | {row["D"]:.4f} | '
        f'{row["M"]:.4f} | {row["Q"]:.4f} | {row["score_S1"]:.4f} | '
        f'{row["srb_benefit_vs_detector"]:+.4f} | `{row["image_path"]}` |'
    )
OUT_REPORT.write_text('\n'.join(lines), encoding='utf-8', newline='\n')

print('===== LABEL-AWARE CASE TYPES =====')
print(inventory['case_type'].value_counts().sort_index().to_string())
print('\n===== TOP 3 PER CASE TYPE =====')
print(
    inventory.groupby('case_type', sort=True).head(3)[
        [
            'case_type', 'case_rank', 'category', 'Y', 'D', 'M', 'Q',
            'score_S1', 'srb_benefit_vs_detector',
            'naive_harm_vs_detector', 'image_path',
        ]
    ].to_string(index=False)
)
print()
print('[DONE]', OUT_CSV)
print('[DONE]', OUT_SUMMARY)
print('[DONE]', OUT_REPORT)
