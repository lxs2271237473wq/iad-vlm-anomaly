# Stage 10-A 新数据集扩展计划

## 1. 当前判断

MVTec AD 和 VisA 已经完成主要验证。继续在这两个数据集上挤小模块，论文收益有限。
Stage 10 建议引入更具挑战性的数据集，用于验证 localization-guided VLM reasoning 是否仍然成立。

## 2. 推荐优先级

| Priority | Dataset | Status | Image Count | Role |
|---:|---|---|---:|---|
| 1 | MVTec AD 2 | not_found | 0 | main_next_dataset |
| 2 | Real-IAD | not_found | 0 | large_scale_generalization |
| 3 | MVTec LOCO AD | not_found | 0 | logical_anomaly_reasoning_supplement |
| 4 | MPDD | not_found | 0 | small_real_metal_part_sanity_check |

## 3. 数据集选择原则

### 3.1 首选 MVTec AD 2

原因：它仍然是 2D industrial anomaly detection，和当前 PatchCore / FastFlow / crop reasoning pipeline 最接近；
同时它比 MVTec AD 和 VisA 更难，更适合说明方法在新挑战场景下仍有价值。

### 3.2 第二选择 Real-IAD

原因：它规模更大，真实产线、多视角，适合做大规模泛化实验。
风险：下载和整理成本高，不建议在没有先跑通 MVTec AD 2 adapter 前直接全量上 Real-IAD。

### 3.3 MVTec LOCO AD 作为 reasoning supplement

原因：logical anomaly 更适合 VLM 解释能力。
风险：logical anomaly 不一定表现为局部 anomaly map 高响应，和当前 crop assumption 可能冲突。

### 3.4 MPDD 作为小型 sanity check

原因：数据小、真实金属件缺陷，适合快速验证。
风险：规模不够，不建议作为主实验数据集。

## 4. 本地可用性检查结果

结果表：`results/stage10_dataset_expansion/stage10_dataset_availability.csv`

当前没有检测到可直接使用的新数据集。下一步需要先下载并整理 MVTec AD 2。

## 5. 下一步

Stage 10-B 应实现所选数据集的 manifest builder。
manifest 至少包含：

```text
dataset, category, split, image_path, mask_path, is_anomaly, anomaly_type
```

之后再复用当前 pipeline：

```text
detector prediction -> candidate crop -> VLM full/crop reasoning -> table/report
```