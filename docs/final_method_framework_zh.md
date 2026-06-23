# 最终方法框架：PatchCore 引导的视觉 Prompt 缺陷推理与制造过程知识解释

## 1. 研究问题

本项目研究的是比传统工业异常检测更进一步的工业异常理解任务。

目标不是只判断一张工业图像是否异常，而是形成完整的结构化理解流程：

```text
异常定位 -> 缺陷类型推理 -> 制造过程知识解释
```

当前实验重点放在 MVTec AD 中较弱的四个类别：

```text
grid, screw, leather, wood
```

选择这四类的原因是：PatchCore 在图像级异常检测上已经很强，但在像素级定位和细粒度缺陷理解上仍然存在明显不足。

## 2. 核心动机

前期实验表明，单纯继续改 segmentation mask，收益非常有限。

已经尝试过的分支包括：

* PatchCore baseline
* PatchCore 失败样本分析
* 阈值诊断
* candidate region 提取
* SAM2 prompt-based refinement
* CLIP semantic reranking
* 手工区域评分
* anomaly map calibration
* trainable anomaly map calibration
* conservative residual calibration
* defect type prompt reasoning
* visual prompt refinement
* manufacturing-aware explanation generation

主要结论是：

```text
PatchCore 的异常定位能力有价值，但单纯做 mask-level refinement 提升有限。
更有希望的方向，是把 PatchCore 异常区域作为视觉聚焦区域，用于缺陷类型推理。
```

## 3. 最终方法流程

最终方法流程如下：

```text
输入工业图像
        |
        v
PatchCore anomaly map
        |
        v
Top-k 异常候选 crop
        |
        v
基于 CLIP 的短视觉 prompt 缺陷类型推理
        |
        v
制造过程知识检索
        |
        v
结构化解释：
缺陷类型 -> 视觉证据 -> 可能制造环节 -> 可能原因
```

## 4. 模块设计

### 4.1 PatchCore 异常定位

PatchCore 被用作异常定位 backbone。

我们不是替换 PatchCore，而是利用 PatchCore 提供的异常图、候选区域、候选 crop 和异常分数，作为后续语义推理的输入。

输出包括：

```text
anomaly map
candidate boxes
candidate crops
candidate anomaly scores
```

### 4.2 Full-test Candidate Region 生成

早期 candidate extraction 只覆盖了大约一半异常测试图像。

Stage 6.4 重新在 full test set 上生成 candidate regions，修复了这个问题。

最终覆盖率为：

```text
327 / 328 abnormal images
coverage ratio = 0.9970
```

这使得 crop-based reasoning 接近完整测试集下的真实可用设置。

### 4.3 短视觉 Prompt 缺陷类型推理

当前最强的真实公平设置是：

```text
prompt strategy = generic_label
input mode = crop_topk_ensemble
image count = 328
```

它对应的公平 baseline 是：

```text
prompt strategy = generic_label
input mode = full_all
image count = 328
```

主结果为：

```text
full_all Top-1: 0.2850
crop_topk_ensemble Top-1: 0.3388
Top-1 improvement: +0.0537

full_all Macro-F1: 0.1543
crop_topk_ensemble Macro-F1: 0.2206
Macro-F1 improvement: +0.0663
```

这说明，在同样数据和同样评价口径下，PatchCore 引导的异常 crop 能提升缺陷类型推理能力。

### 4.4 制造过程知识解释

制造过程知识不再作为长文本 prompt 直接输入 CLIP 做分类。

它被放在缺陷类型预测之后，用于生成结构化解释。

每条解释包含：

* 预测缺陷类型
* Top-2 缺陷候选
* PatchCore candidate region
* anomaly score
* 视觉证据
* 缺陷族
* 可能制造环节
* 可能制造原因
* 检测关注点

重要结论：

```text
制造过程知识更适合用于解释，而不是直接用于 CLIP 分类。
```

## 5. 主要实验证据

### 5.1 主公平结果

| Setting                            |   Images |   Top-1 |   Top-2 | Macro-F1 |
| ---------------------------------- | -------: | ------: | ------: | -------: |
| generic_label + full_all           |      328 |  0.2850 |  0.4990 |   0.1543 |
| generic_label + crop_topk_ensemble |      328 |  0.3388 |  0.5072 |   0.2206 |
| Improvement                        | same 328 | +0.0537 | +0.0082 |  +0.0663 |

这是目前最干净的主结果，因为它满足：

```text
同一数据集
同 328 张图像
同一 prompt strategy
同一 defect label space
同一 evaluation metrics
```

### 5.2 Prompt 与 Crop 消融

消融实验表明：

1. crop-based reasoning 通常能提升短 prompt 方法。
2. `generic_label + crop_topk_ensemble` 得到最好的 full-test Top-1 和 Macro-F1。
3. `visual_ensemble` 能提升 Top-2，但会降低 Top-1 和 Macro-F1。
4. 在真实 PatchCore crop 存在噪声的情况下，更长、更复杂的 prompt 不一定更好。

### 5.3 GT-crop 上限诊断

GT-crop 结果只用于诊断上限，不能作为真实可部署方法的公平结果。

最好的 GT-crop 结果是：

```text
category_visual + gt_crop
Top-1 = 0.3102
Macro-F1 = 0.2460
```

这说明准确的异常区域聚焦有利于类别均衡的缺陷类型推理。

但部署时没有 GT mask，因此这些结果必须单独标注为 upper-bound diagnostic。

### 5.4 负结果与辅助结果

| Branch                            | Conclusion             |
| --------------------------------- | ---------------------- |
| SAM2 prompt refinement            | 通用分割模型不能直接很好地细化缺陷 mask |
| CLIP semantic candidate reranking | 有弱正向效果，但不足以作为主模块       |
| 手工区域评分                            | 有弱正向效果，但收益有限           |
| Trainable anomaly map calibration | 不稳定或收益太小               |
| Manufacturing-aware long prompts  | 不适合直接做 CLIP 分类         |

这些实验虽然不是最终主模块，但它们证明了为什么最终方法要采用当前设计。

## 6. 最终贡献点表述

### 贡献 1：PatchCore 引导的 anomaly crop reasoning

不再只围绕 pixel mask 做小修小补，而是利用 PatchCore anomaly map 引导视觉语言模型进行缺陷类型推理。

### 贡献 2：Full-test 公平评估设置

项目明确区分：

```text
full-test realistic setting
near-full candidate subset
partial candidate setting
GT-crop upper bound
```

避免把不同图像子集、不同评价口径的结果混在一起比较。

### 贡献 3：制造过程知识解释层

方法将分类和解释解耦：

```text
短视觉 prompt 用于 defect type prediction
制造过程知识用于 explanation 和 possible cause reasoning
```

最终输出具有更强解释性的工业异常理解结果。

## 7. 论文结构草案

推荐论文结构：

```text
1. Introduction
2. Related Work
   2.1 Industrial anomaly detection
   2.2 Vision-language models for anomaly understanding
   2.3 Manufacturing knowledge and explainable inspection
3. Method
   3.1 PatchCore anomaly localization
   3.2 Full-test anomaly candidate crop generation
   3.3 Short visual prompt defect type reasoning
   3.4 Manufacturing-aware explanation generation
4. Experiments
   4.1 Dataset and evaluation protocol
   4.2 PatchCore baseline and failure analysis
   4.3 Main fair comparison
   4.4 Prompt and crop ablation
   4.5 GT-crop upper-bound analysis
   4.6 Negative and auxiliary experiments
   4.7 Manufacturing-aware explanation examples
5. Discussion
6. Conclusion
```

## 8. 当前论文主结果

论文中最应该报告的主结果是：

```text
在同 328 张 MVTec AD 弱类别异常图像上，
generic_label + crop_topk_ensemble 相比
generic_label + full_all 提升了缺陷类型推理能力。

Top-1: 0.2850 -> 0.3388
Macro-F1: 0.1543 -> 0.2206
```

## 9. 当前局限性

当前方法仍有以下局限：

1. 缺陷类型识别准确率仍然不高。
2. CLIP 的 confidence margin 较小。
3. 解释是 possible-cause reasoning，不是真实因果诊断。
4. 制造知识库是人工结构化知识，不是真实工厂 SOP。
5. 当前只在 MVTec AD 弱类别上验证，还没有迁移到更复杂的工业异常理解数据集。

## 10. 下一阶段建议

下一步建议是：

```text
Stage 6.9: Final paper outline and experiment narrative
```

如果还需要继续补实验，则进入：

```text
Stage 7.0: Cross-dataset validation on a more complex industrial anomaly dataset
```

当前更建议先写论文叙事，因为现有实验链条已经比较完整。
