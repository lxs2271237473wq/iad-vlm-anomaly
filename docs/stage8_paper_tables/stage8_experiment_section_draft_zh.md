# Stage 8-C 论文实验章节草稿

## 1. 实验目标

本实验部分的目标是验证：工业异常定位结果是否能够作为传统 anomaly detector 与视觉语言模型之间的有效桥梁。

具体而言，我们不只验证 anomaly detector 的图像级或像素级检测能力，而是进一步考察：由 anomaly detector 产生的异常候选区域，是否能够提升视觉语言模型的异常理解与异常判断能力。

实验围绕三个问题展开：

1. 在 MVTec AD 上，异常区域裁剪是否能够提升细粒度缺陷类型理解能力。
2. 在 VisA 上，异常区域裁剪是否能够提升二分类 normal/anomaly reasoning 能力。
3. 该提升是否依赖单一 anomaly backbone，还是可以从 PatchCore 泛化到 FastFlow。

## 2. 数据集

### 2.1 MVTec AD

MVTec AD 用于验证缺陷类型理解任务。该部分实验聚焦于弱类别设置，即 full image 输入下 VLM 对缺陷类型判断较困难的类别。我们采用公平 full-test 设置比较 full image 与 anomaly crop 输入，避免只在局部样本或 GT crop 上给出乐观结论。

### 2.2 VisA

VisA 用于跨数据集泛化验证。与 MVTec AD 不同，VisA 当前实验设置中没有直接提供 scratch、cut、hole、color 等细粒度 defect type 标签。因此，在 VisA 上我们不做 MVTec-style defect-type classification，而是构造 binary normal/anomaly prompt reasoning 任务。

VisA 共使用 12 个类别，测试图像共 2162 张，其中 normal 图像 962 张，anomaly 图像 1200 张。

## 3. 方法与 Baseline

### 3.1 Full-image prompting

Full-image prompting 直接将整张测试图像输入视觉语言模型，并通过 normal/anomaly 或 defect-type prompt 进行相似度匹配。该设置作为最直接的 VLM baseline。

### 3.2 Localization-guided crop prompting

Localization-guided crop prompting 首先使用 anomaly detector 生成 anomaly map，再从 anomaly map 中提取 top-k candidate regions。随后，将这些 anomaly crops 输入视觉语言模型进行 reasoning。

我们主要比较三种输入模式：

| Mode | Description |
|---|---|
| full_all | 所有图像均使用 full image 输入。 |
| crop_or_full | 有 candidate 时使用 top-1 crop，无 candidate 时 fallback 到 full image。 |
| crop_topk_ensemble | 使用 top-k anomaly crops，并对多个 crop 的 anomaly score 做 ensemble。 |

### 3.3 Anomaly backbones

我们使用 PatchCore 和 FastFlow 作为 anomaly candidate generator。PatchCore 是 memory-bank nearest-neighbor anomaly detector，FastFlow 是 flow-based anomaly detector。二者机制不同，因此可以用于验证 backbone-level generalization。

## 4. 评价指标

### 4.1 Detection metrics

对于 anomaly detection backbone，我们报告 image-level AUROC、image-level AP、image-level best F1、pixel-level AUROC、pixel-level AP、pixel-level best F1，以及 candidate coverage。

Candidate coverage 表示异常图像中至少成功提取一个 candidate region 的比例。该指标用于判断后续 crop reasoning 是否存在大量 fallback。

### 4.2 Reasoning metrics

对于 MVTec AD 缺陷类型理解任务，我们报告 Top-1 accuracy 和 Top-2 accuracy。对于 VisA normal/anomaly reasoning 任务，我们报告 AUROC、AP、best F1 和 best accuracy。

## 5. 主实验结果

### 5.1 总体结果

| Dataset | Backbone | Task | Full image | Crop | Improvement |
|---|---|---|---:|---:|---:|
| MVTec AD | PatchCore | Defect-type reasoning | 0.2850 Top-1 | 0.3388 Top-1 | +0.0538 |
| VisA | PatchCore | Binary normal/anomaly reasoning | 0.5950 AUROC | 0.8844 AUROC | +0.2894 |
| VisA | FastFlow | Binary normal/anomaly reasoning | 0.5950 AUROC | 0.9222 AUROC | +0.3272 |

结果显示，在 MVTec AD 和 VisA 两个数据集上，localization-guided crop prompting 均优于 full-image prompting。这说明异常定位信息不仅可以用于检测，也可以作为 VLM reasoning 的有效视觉引导。

## 6. 跨数据集泛化分析

在 MVTec AD 上，PatchCore-guided crop 将 defect-type reasoning 的 Top-1 从 0.2850 提升到 0.3388。

在 VisA 上，PatchCore-guided crop 将 binary anomaly reasoning 的 AUROC 从 0.5950 提升到 0.8844，提升 +0.2894。

这说明该方法不是只在 MVTec AD 上有效。即使任务从 defect-type reasoning 变为 binary normal/anomaly reasoning，crop-guided visual input 仍然显著强于 full image。

## 7. 跨 Backbone 泛化分析

| Backbone | Image AUROC | Pixel AUROC | Candidate Coverage | Full AUROC | Crop AUROC | Delta AUROC | Crop F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | 0.9138 | 0.8971 | 1.0000 | 0.5950 | 0.8844 | +0.2894 | 0.8783 |
| FastFlow | 0.8934 | 0.9511 | 0.9992 | 0.5950 | 0.9222 | +0.3272 | 0.9042 |

PatchCore 和 FastFlow 都能提供接近完整的 candidate coverage。虽然 PatchCore 的 image-level AUROC 更高，FastFlow 的 pixel-level AUROC 更高，但二者产生的 candidate crops 都显著提升了 VLM reasoning。

这说明 localization-guided reasoning 并不严格依赖 PatchCore，而可以泛化到机制不同的 anomaly backbone。

## 8. 消融逻辑

当前实验已经包含以下消融逻辑：

| Ablation | Purpose | Conclusion |
|---|---|---|
| full_all vs crop_topk_ensemble | 验证 anomaly crop 是否优于 full image。 | crop_topk_ensemble 明显更强。 |
| PatchCore vs FastFlow | 验证是否依赖单一 anomaly backbone。 | 两个 backbone 上均成立。 |
| MVTec AD vs VisA | 验证是否依赖单一数据集。 | 两个数据集上均有正向结果。 |

## 9. 局限性

当前方法仍然依赖 anomaly localization 的质量。如果 candidate region 无法覆盖真实缺陷，或者异常区域非常小、非常弥散、对比度很低，则 crop reasoning 的收益可能下降。

此外，当前实验验证的是视觉异常区域对 VLM reasoning 的帮助，而不是完整制造因果链推理。因此，不能声称模型已经能够发现所有未知制造原因。

## 10. 不能过度声称的内容

| 不应声称 | 原因 |
|---|---|
| 方法已经解决像素级精确分割 | pixel F1 仍然有限。 |
| 方法能够自动发现全部制造原因 | 当前实验验证的是视觉区域引导，不是完整因果发现。 |
| GT crop 可与真实 candidate crop 直接公平比较 | GT crop 只能作为 upper-bound diagnostic。 |
| full image prompting 完全无用 | full image 是必要 baseline，结论是 crop guidance 更强。 |

## 11. 实验章节当前结论

当前实验结果支持如下论文主张：

Classical anomaly localization can serve as an effective bridge between industrial anomaly detectors and visual-language reasoning models.

在 MVTec AD 与 VisA 上，以及在 PatchCore 与 FastFlow 两种 candidate generator 上，anomaly-crop prompting 均优于 full-image prompting。
