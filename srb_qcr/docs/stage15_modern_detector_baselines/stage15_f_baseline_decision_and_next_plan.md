# Stage 15-F 强基线结论锁定与后续实验决策

## 1. 本阶段目的

Stage 15-E 已经把 WinCLIP、full-image VLM、context-aware VLM、PatchCore、EfficientAD-30、PatchCore+context VLM fusion 放进统一四类别对比表。

Stage 15-F 的目的不是继续跑实验，而是把当前强基线结论和下一步实验优先级锁定下来。

## 2. 当前平均 Image AUROC 排名

| Rank | Method | Mean Image AUROC | Fairness Tag |
|---:|---|---:|---|
| 1 | PatchCore + context VLM, same-set | 0.8453 | mean_summary |
| 2 | PatchCore + context VLM, LOCO | 0.8210 | mean_summary |
| 3 | PatchCore | 0.7853 | mean_summary |
| 4 | EfficientAD-30 fixed-budget | 0.7604 | mean_summary |
| 5 | context-aware VLM | 0.7101 | mean_summary |
| 6 | full-image VLM | 0.6459 | mean_summary |
| 7 | WinCLIP fixed protocol | 0.6138 | mean_summary |

## 3. 关键差值

- LOCO fusion minus EfficientAD-30: `0.0606` mean image AUROC.
- EfficientAD-30 minus PatchCore: `-0.0249` mean image AUROC.
- EfficientAD-30 minus context-aware VLM: `0.0503` mean image AUROC.
- LOCO fusion minus PatchCore: `0.0356` mean image AUROC.

## 4. 当前可以安全使用的结论

### 4.1 可以作为主结论

`PatchCore + context VLM, LOCO` 是当前最重要的公平 fusion 结果。

它比单独 PatchCore 和 EfficientAD-30 fixed-budget 都更高，因此当前 localization-guided VLM fusion 路线仍然站得住。

### 4.2 只能作为 upper-bound / diagnostic

`PatchCore + context VLM, same-set` 不能作为最终公平结论。它可以展示同类别调参或同集合融合的上界，但不能过度声称为真实泛化性能。

### 4.3 EfficientAD 的定位

`EfficientAD-30 fixed-budget` 是现代非 VLM detector baseline。它比 WinCLIP 和普通 VLM 分支更强，但没有超过 LOCO fusion。

它不能被写成 EfficientAD official/full-budget baseline。正式论文中必须标注为 fixed-budget 结果。

## 5. 是否现在跑 EfficientAD-100

当前决策：**不立即跑四类别 EfficientAD-100**。

理由：

1. EfficientAD-30 没有推翻当前 LOCO fusion 结论。
2. EfficientAD 在 Anomalib 下 `train_batch_size=1`，验证阶段还有 quantile/metric 开销，四类别 100 epoch 成本较高。
3. 100 epoch 的价值主要是防守性质，即回答“30 epoch 是否低估 EfficientAD”。

后续只需要先补一个：

```text
fruit_jelly EfficientAD-100 sensitivity
```

如果 fruit_jelly 上 100 epoch 明显高于 30 epoch，再考虑四类别 100 epoch。

## 6. 下一阶段主线

下一阶段不应该继续堆 detector baseline，而应该进入：

```text
Stage 16 / QCR-U ablation
```

目标是把当前 pipeline 从：

```text
detector map -> crop -> VLM score -> naive fusion
```

升级成：

```text
candidate quality + VLM abnormal margin + detector-VLM consistency + optional unknown-aware reasoning
```

也就是 QCR-U。

## 7. 下一步实验优先级

| Priority | Task | Why |
|---:|---|---|
| 1 | QCR-U fixed-protocol ablation | 这是论文方法核心，不是 baseline 补丁 |
| 2 | fruit_jelly EfficientAD-100 sensitivity | 防守 30 epoch 是否低估 EfficientAD 的质疑 |
| 3 | AnomalyCLIP feasibility check | 补更强 VLM anomaly baseline，但复现成本可能高 |
| 4 | failure case analysis | 支撑论文边界，避免过度声称 |

## 8. 论文写作边界

后续论文不能写：

- 本文解决完整 industrial anomaly understanding。
- 本文能解释 manufacturing cause。
- 本文达到 pixel-level segmentation SOTA。
- EfficientAD 已经被 full-budget 完整击败。

后续论文应该写：

- 本文研究如何将 anomaly localization evidence 转化为 reliable visual-language evidence。
- 本文提出 quality-consistency guided candidate reasoning/fusion。
- 本文在多类别和多个 baseline 下证明 localization-guided VLM branch 与传统 detector 具有互补性。

## 9. 本阶段输出

- `results/stage15_modern_detector_baselines/stage15_f_baseline_decision_summary.csv`
- `docs/stage15_modern_detector_baselines/stage15_f_baseline_decision_and_next_plan.md`
