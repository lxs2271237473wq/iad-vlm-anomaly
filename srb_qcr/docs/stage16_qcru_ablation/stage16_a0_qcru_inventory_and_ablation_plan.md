# Stage 16-A0 QCR-U 输入审计与消融计划

## 1. 本阶段目的

Stage 15 已经完成强基线结论锁定。下一阶段进入 QCR-U，但不能直接写新方法或乱调融合权重。

本阶段只做三件事：

1. 审计 Stage 9 / Stage 13 / Stage 15 已有结果文件。
2. 判断哪些文件可以作为 QCR-U ablation 的输入。
3. 锁定 QCR-U 消融变量，避免把 heuristic fusion 包装成方法。

## 2. 输入文件审计结果

- total_sources_checked: `15`
- usable_sources: `15`
- missing_sources: `0`
- malformed_or_empty_sources: `0`

完整审计表见：

`results/stage16_qcru_ablation/stage16_a0_qcru_source_inventory.csv`

## 3. 必须解决的硬问题

QCR-U 不能只是：

```text
score = alpha * detector + beta * vlm
```

它必须至少证明：

1. candidate quality 是否有效。
2. detector-VLM consistency 是否有效。
3. QCR-U 是否稳定优于 naive fusion。
4. 参数是否来自固定协议，而不是 test-set 调参。

## 4. QCR-U 消融计划

| Variant | Detector | Crop VLM | Quality | Consistency | Unknown | Purpose |
|---|---:|---:|---:|---:|---:|---|
| detector_only | 1 | 0 | 0 | 0 | 0 | Anchor baseline; proves whether QCR-U beats the detector alone. |
| full_image_vlm | 0 | 0 | 0 | 0 | 0 | Weak VLM sanity baseline; should not be the main comparison target. |
| crop_topk_vlm | 0 | 1 | 0 | 0 | 0 | Tests whether localization-guided crops improve VLM scoring. |
| naive_detector_crop_fusion | 1 | 1 | 0 | 0 | 0 | Naive fusion baseline; QCR-U must beat this or the method is not justified. |
| quality_weighted_crop | 1 | 1 | 1 | 0 | 0 | Tests whether candidate quality contributes beyond crop scoring. |
| quality_consistency_fusion | 1 | 1 | 1 | 1 | 0 | Core QCR-U binary anomaly recognition variant. |
| qcr_u_full_optional_unknown | 1 | 1 | 1 | 1 | 1 | Only valid if a strict known/unknown protocol is available. |

## 5. 下一步决策

如果 Stage 9 的旧 QCR-U 文件已经包含足够字段，下一步进入：

```text
Stage 16-A1: QCR-U fixed-protocol ablation implementation
```

如果字段不够，先补：

```text
Stage 16-A1-input: 构建统一 prediction table
```

统一 prediction table 至少要包含：

- category
- image_id 或 image_path
- gt_binary
- detector_score
- full_image_vlm_score
- crop_topk_vlm_score
- candidate_quality
- consistency_score
- candidate_count

## 6. 本阶段输出

- `results/stage16_qcru_ablation/stage16_a0_qcru_source_inventory.csv`
- `results/stage16_qcru_ablation/stage16_a0_qcru_ablation_plan.csv`
- `docs/stage16_qcru_ablation/stage16_a0_qcru_inventory_and_ablation_plan.md`
