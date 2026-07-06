# Paper Stage P11-B: Boundary Case Replacement Candidates

## 1. Purpose

Panels D and E from P11 were marked as `replace`. This stage proposes replacement candidates.

Rules:

- D replacement should be a stronger `quality_boundary_normal_boost` case.
- E replacement should be a distinct fixed-consistency boundary case, not duplicating panel B.
- If D remains weak, it is acceptable to drop D and use a 4-panel or 5-panel Figure 2.

## 2. Candidate list

| Target | Rank | Case type | Category | Image key | Score col | Score | Strength |
|---|---:|---|---|---|---|---:|---|
| D_replacement | 1 | quality_boundary_normal_boost | pcb2 | datasets/VisA_anomalib_1cls/pcb2/test/good/pcb2_test_normal_0011.JPG | delta_quality_minus_naive | 0.0094 | weak |
| D_replacement | 2 | quality_boundary_normal_boost | pcb2 | datasets/VisA_anomalib_1cls/pcb2/test/good/pcb2_test_normal_0024.JPG | delta_quality_minus_naive | 0.0080 | weak |
| D_replacement | 3 | quality_boundary_normal_boost | macaroni2 | datasets/VisA_anomalib_1cls/macaroni2/test/good/macaroni2_test_normal_0674.JPG | delta_quality_minus_naive | 0.0065 | weak |
| D_replacement | 4 | quality_boundary_normal_boost | capsules | datasets/VisA_anomalib_1cls/capsules/test/good/capsules_test_normal_377.JPG | delta_quality_minus_naive | 0.0051 | weak |
| D_replacement | 5 | quality_boundary_normal_boost | candle | datasets/VisA_anomalib_1cls/candle/test/good/candle_test_normal_0730.JPG | delta_quality_minus_naive | 0.0041 | weak |
| E_replacement | 1 | fixed_consistency_boundary_normal_boost | pcb1 | datasets/VisA_anomalib_1cls/pcb1/test/good/pcb1_test_normal_1000.JPG | delta_fixed_minus_quality | 0.1219 | strong |
| E_replacement | 2 | fixed_consistency_boundary_normal_boost | pcb1 | datasets/VisA_anomalib_1cls/pcb1/test/good/pcb1_test_normal_0923.JPG | delta_fixed_minus_quality | 0.1194 | strong |
| E_replacement | 3 | fixed_consistency_boundary_normal_boost | pcb1 | datasets/VisA_anomalib_1cls/pcb1/test/good/pcb1_test_normal_0952.JPG | delta_fixed_minus_quality | 0.1190 | strong |
| E_replacement | 4 | fixed_consistency_boundary_normal_boost | pcb1 | datasets/VisA_anomalib_1cls/pcb1/test/good/pcb1_test_normal_0187.JPG | delta_fixed_minus_quality | 0.1181 | strong |
| E_replacement | 5 | fixed_consistency_boundary_normal_boost | pcb3 | datasets/VisA_anomalib_1cls/pcb3/test/good/pcb3_test_normal_0402.JPG | delta_fixed_minus_quality | 0.1002 | strong |
| E_alt_replacement | 1 | fixed_consistency_boundary_anomaly_suppression | pcb3 | datasets/VisA_anomalib_1cls/pcb3/test/anomaly/pcb3_test_anomaly_054.JPG | delta_fixed_minus_quality | -0.0908 | inspect |
| E_alt_replacement | 2 | fixed_consistency_boundary_anomaly_suppression | pcb4 | datasets/VisA_anomalib_1cls/pcb4/test/anomaly/pcb4_test_anomaly_035.JPG | delta_fixed_minus_quality | -0.0874 | inspect |
| E_alt_replacement | 3 | fixed_consistency_boundary_anomaly_suppression | pipe_fryum | datasets/VisA_anomalib_1cls/pipe_fryum/test/anomaly/pipe_fryum_test_anomaly_090.JPG | delta_fixed_minus_quality | -0.0860 | inspect |
| E_alt_replacement | 4 | fixed_consistency_boundary_anomaly_suppression | pcb4 | datasets/VisA_anomalib_1cls/pcb4/test/anomaly/pcb4_test_anomaly_039.JPG | delta_fixed_minus_quality | -0.0808 | inspect |
| E_alt_replacement | 5 | fixed_consistency_boundary_anomaly_suppression | macaroni2 | datasets/VisA_anomalib_1cls/macaroni2/test/anomaly/macaroni2_test_anomaly_090.JPG | delta_fixed_minus_quality | -0.0770 | inspect |

## 3. Contact sheet

- `docs/paper_p11b/figures/paper_p11b_replacement_candidates_contact_sheet.png`

## 4. Manual decision required

Open the contact sheet and choose:

```text
D replacement: one candidate, or drop panel D
E replacement: one candidate, preferably not visually/numerically duplicate with B
```

After choosing, update the final Figure 2 panel list in P12.
