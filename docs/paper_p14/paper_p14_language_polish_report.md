# Paper Stage P14: Language Polish and Venue-style Compression

## 1. Summary

- input manuscript: `paper/quality_calibrated_qcr/main.tex`
- polished copy: `paper/quality_calibrated_qcr/main_p14_polished.tex`
- word count before: `927`
- word count after: `919`
- word count delta: `-8`

P14 generates a polished copy only. It does not overwrite `main.tex`.

## 2. Patch Inventory

| Patch | Type | Applied | Reason |
|---|---|---:|---|
| P14-01 | tone_softening | 1 | Avoid overgeneralizing VLM unreliability. |
| P14-02 | tone_softening | 1 | Sharper and less absolute. |
| P14-03 | claim_precision | 1 | Clarifies what is reported as the main method. |
| P14-04 | claim_precision | 1 | More compact and conservative. |
| P14-05 | compression | 1 | Tighter conclusion. |
| P14-06 | claim_safety | 0 | Remove possible duplicated limitation wording. |
| P14-07 | claim_safety | 1 | Remove forbidden wording while preserving limitation. |
| P14-08 | baseline_precision | 1 | Make baseline budget explicit. |

## 3. Claim Safety Scan

| Item | Status | Note |
|---|---|---|
| state-of-the-art segmentation | ok |  |
| SOTA segmentation | ok |  |
| manufacturing cause | ok |  |
| manufacturing-cause reasoning | flag | Review and remove/soften this phrase. |
| full anomaly understanding | ok |  |
| universally beneficial | ok |  |
| defeat EfficientAD | ok |  |
| beats EfficientAD | ok |  |
| outperforms AnomalyCLIP | ok |  |
| SOTA | flag | Review and remove/soften this phrase. |
| Quality-Calibrated QCR | ok | main method name |
| EfficientAD-30 fixed-budget | ok | fixed-budget baseline wording |
| diagnostic | ok | fixed Q+C diagnostic wording |
| AnomalyCLIP | ok | missing external VLM baseline limitation |
| candidate quality | ok | main method mechanism |
| adaptive | ok | adaptive refinement wording |

## 4. Section Word Counts

| Section | Before | After | Delta |
|---|---:|---:|---:|
| Ablation Study | 57 | 57 | +0 |
| Conclusion | 51 | 41 | -10 |
| EfficientAD Budget Sensitivity | 40 | 40 | +0 |
| Experiments | 57 | 57 | +0 |
| Failure and Boundary Analysis | 55 | 55 | +0 |
| Introduction | 175 | 175 | +0 |
| Limitations | 42 | 45 | +3 |
| Main Results | 54 | 54 | +0 |
| Method | 168 | 166 | -2 |
| Related Work | 97 | 97 | +0 |
| preamble_and_abstract | 131 | 132 | +1 |

## 5. Decision

P14 found scan flags or missing required phrases. Review before using the polished copy.

## 6. Next Step

```text
Patch P14 polished copy and rerun P14/P13 checks
```
