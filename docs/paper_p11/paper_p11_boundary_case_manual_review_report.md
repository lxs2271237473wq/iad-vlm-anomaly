# Paper Stage P11: Figure 2 Manual Image Montage Preparation

## 1. Summary

- selected panels: `6`
- unresolved image assets: `0`
- contact sheet: `docs/paper_p11/figures/figure2_boundary_cases_contact_sheet.png`

This stage prepares manual inspection assets only. It does not finalize Figure 2.

## 2. Outputs

- asset manifest: `results/paper_p11/paper_p11_boundary_case_asset_manifest.csv`
- manual review sheet: `results/paper_p11/paper_p11_boundary_case_manual_review_sheet.csv`
- montage plan: `docs/paper_p11/figure2_boundary_cases_montage_plan.md`
- contact sheet: `docs/paper_p11/figures/figure2_boundary_cases_contact_sheet.png`

## 3. Panel review table

| Panel | Case type | Category | Asset status | Manual decision |
|---|---|---|---|---|
| A | quality_helps_anomaly_boost | chewinggum | resolved | pending |
| B | quality_helps_normal_suppression | pcb1 | resolved | pending |
| C | quality_boundary_anomaly_suppression | pipe_fryum | resolved | pending |
| D | quality_boundary_normal_boost | macaroni2 | resolved | pending |
| E | fixed_consistency_boundary_normal_boost | pcb1 | resolved | pending |
| F | detector_vlm_disagreement_boundary | pipe_fryum | resolved | pending |

## 4. Next manual action

Open the copied assets or contact sheet and edit:

`results/paper_p11/paper_p11_boundary_case_manual_review_sheet.csv`

Set each `manual_decision` to one of:

```text
keep
reject
replace
```

Only panels marked `keep` should be used for the final Figure 2 montage.

## 5. Next stage

```text
Paper Stage P12: build final Figure 2 montage from manually approved cases
```
