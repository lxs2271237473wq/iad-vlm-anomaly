# Paper Stage P11-B Manual Replacement Decision

## Decision

Final Figure 2 should use five panels:

```text
A: keep original P11 panel A
B: keep original P11 panel B
C: keep original P11 panel C
E: use E_replacement rank 5
F: keep original P11 panel F
D: drop
```

## Rationale

- D replacement candidates are all weak; delta_quality_minus_naive is only about +0.01 or 0.
- E_replacement rank 1-4 are pcb1-like and risk duplicating panel B.
- E_replacement rank 5 is the preferred fixed-consistency boundary case: it is distinct, resolved, and has strong fixed-minus-quality delta.
- E_alt candidates are not selected because they overlap with panel F's detector-VLM disagreement role.

## Selected replacement

- final_panel: `E`
- target_panel: `E_replacement`
- candidate_rank: `5`
- case_type: `fixed_consistency_boundary_normal_boost`
- category: `pcb3`
- backbone: `FastFlow`
- selection_score_col: `delta_fixed_minus_quality`
- selection_score: `0.1001895687658122`
- copied_image_path: `docs/paper_p11b/replacement_candidate_assets/E_replacement_5_pcb3_test_normal_0402.JPG`
