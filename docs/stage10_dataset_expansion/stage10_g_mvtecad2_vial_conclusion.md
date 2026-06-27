# Stage 10-G MVTec AD 2 / Vial 结论整合

## 1. 当前结论

MVTec AD 2 / vial 上，原始小尺度 candidate crop 失败，但加入足够对象上下文后，VLM reasoning 明显超过 full-image prompting。

最安全的论文表述是：

```text
Naive anomaly crops are insufficient for challenging AD2 vial images, but localization-guided context-aware crops improve VLM anomaly reasoning.
```

## 2. 关键结果

| Method | AUROC | AP | Best F1 | ΔAUROC vs full |
|---|---:|---:|---:|---:|
| patchcore_score | 0.8899 | 0.9605 | 0.9204 | 0.2411 |
| context_1.50_top1 | 0.7746 | 0.9081 | 0.8667 | 0.1258 |
| full_image | 0.6488 | 0.8722 | 0.8548 | 0.0000 |
| stage10e_crop_top1 | 0.3753 | 0.7080 | 0.8689 | -0.2736 |

## 3. 解释

Stage 10-E 中，直接裁剪 PatchCore 高响应小区域会让 VLM 丢失 vial 的对象级上下文，因此 crop_top1 明显低于 full-image。
Stage 10-F 恢复了多尺度上下文后，context_1.50_top1 达到更高 AUROC，说明问题不在于 localization-guided reasoning 本身，而在于 crop construction 过于激进。

## 4. 对论文方法的影响

后续方法部分不应写成简单的 `anomaly map -> crop -> VLM`。应升级为：

```text
anomaly map -> candidate region -> context-aware crop construction -> VLM reasoning
```

这能避免审稿人质疑工作只是“把异常区域抠出来给 VLM 看”。真正的方法点应强调：

1. localization 给出可疑区域；
2. context-aware crop 保留对象语义和局部异常；
3. VLM 在局部异常与对象上下文共同存在时更稳定。

## 5. 下一步

Stage 11 应在 MVTec AD 2 多类别上批量验证该现象，优先类别建议：sheet_metal、can、wallplugs、fruit_jelly。

对应输出表：

- `results/stage10_dataset_expansion/stage10_g_mvtecad2_vial_final_table.csv`
- `results/stage10_dataset_expansion/stage10_f_multiscale_context_summary.csv`

<!-- stage10_g_conclusion_20260627_140613_576608 -->