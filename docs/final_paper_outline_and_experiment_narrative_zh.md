# Final Paper Outline and Experiment Narrative

## 1. 论文暂定主题

当前论文主题可以暂定为：

```text
PatchCore-guided Visual Prompt Reasoning with Manufacturing-aware Explanation for Industrial Anomaly Understanding
```

中文理解为：

```text
面向工业异常理解的 PatchCore 引导视觉 Prompt 缺陷推理与制造过程知识解释
```

这篇文章的核心不是重新发明一个新的异常检测 backbone，而是在已有异常检测模型的基础上，进一步解决工业异常理解问题：

```text
异常在哪里？
是什么缺陷？
可能与什么制造环节有关？
可能原因是什么？
```

## 2. 核心问题定义

传统工业异常检测通常关注两个任务：

1. image-level anomaly detection：判断整张图是否异常；
2. pixel-level anomaly localization：定位异常区域。

但是这些输出仍然不够接近真实工业质检需求。

真实质检场景更关心：

```text
检测到异常之后，能不能进一步解释这个异常？
```

因此，本项目将任务扩展为：

```text
anomaly localization -> defect type reasoning -> manufacturing-aware explanation
```

也就是：

1. 先定位异常区域；
2. 再识别异常区域可能属于哪种缺陷类型；
3. 最后结合制造过程知识，给出可能的制造环节和原因解释。

## 3. 为什么不继续单纯提升 Pixel F1

前期 PatchCore baseline 表明：

```text
PatchCore 图像级异常检测已经很强；
但在 grid / screw / leather / wood 等类别上，像素级定位仍然存在不足。
```

之后我们尝试了多条提升 mask 或 region quality 的路线：

* 阈值诊断；
* candidate region 提取；
* SAM2 box prompt refinement；
* anomaly-aware SAM2 selection；
* CLIP semantic reranking；
* hand-crafted region scoring；
* anomaly map calibration；
* trainable anomaly map calibration；
* conservative residual calibration。

这些实验给出的结论是：

```text
PatchCore anomaly map 可以提供有效异常候选区域；
但继续围绕 mask refinement 做小幅优化，收益有限。
```

其中：

* SAM2 容易分割完整物体或纹理块，而不是缺陷本身；
* CLIP semantic reranking 有弱正收益，但不够强；
* hand-crafted region scoring 有弱正收益，但收益有限；
* trainable calibration 不稳定；
* conservative residual calibration 稳定但提升太小。

因此，论文叙事不能写成“我们大幅提升了 anomaly segmentation”，而应该写成：

```text
我们发现 PatchCore anomaly map 虽然不能直接给出完美 mask，
但它可以作为有效的 visual focus，
用于引导后续 defect type reasoning。
```

## 4. 最终方法动机

最终方法的动机是：

```text
把 PatchCore 从最终输出模型，转化为 anomaly-guided visual reasoning 的区域提议器。
```

也就是说，PatchCore 的作用不只是输出异常分数，而是提供：

* anomaly map；
* candidate boxes；
* anomaly crops；
* candidate anomaly score。

这些区域再输入视觉语言模型，用短视觉 prompt 做缺陷类型识别。

最后，再把识别到的缺陷类型和制造过程知识库结合，生成结构化解释。

最终流程为：

```text
Input image
    -> PatchCore anomaly map
    -> top-k anomaly crops
    -> CLIP short visual prompt defect reasoning
    -> manufacturing knowledge retrieval
    -> structured explanation
```

## 5. 方法模块叙事

### 5.1 PatchCore anomaly localization

PatchCore 作为无监督工业异常检测 baseline，用于生成 anomaly map。

这里需要强调：

```text
我们没有抛弃 PatchCore，而是复用了它的异常定位能力。
```

PatchCore 的输出被转化为后续推理模块的视觉输入。

### 5.2 Full-test anomaly candidate crop generation

早期 Stage 6.3 的 candidate 覆盖率只有：

```text
164 / 328 = 0.50
```

这导致 crop reasoning 实验存在明显不公平，因为只有一半图像真正使用了 crop，另一半 fallback 到 full image。

Stage 6.4 重新生成 full-test candidate regions，将覆盖率提升到：

```text
327 / 328 = 0.9970
```

这一步很关键，因为它使后续对比更加公平：

```text
不是只在部分图像上验证 crop 有效，
而是在接近完整测试集上验证 crop-based reasoning。
```

### 5.3 Short visual prompt defect type reasoning

实验发现，长 manufacturing-aware prompt 并不适合直接用于 CLIP 分类。

原因是：

```text
制造过程文本会引入过多非视觉语义噪声，
而 CLIP 更擅长匹配短、直接、视觉相关的描述。
```

最终最强的真实 full-test 设置是：

```text
generic_label + crop_topk_ensemble
```

对应公平 baseline 是：

```text
generic_label + full_all
```

两者使用：

* 同 328 张图像；
* 同 prompt strategy；
* 同 defect type label space；
* 同评价指标；
* 同数据来源。

主结果为：

| Setting                            |   Images |   Top-1 |   Top-2 | Macro-F1 |
| ---------------------------------- | -------: | ------: | ------: | -------: |
| generic_label + full_all           |      328 |  0.2850 |  0.4990 |   0.1543 |
| generic_label + crop_topk_ensemble |      328 |  0.3388 |  0.5072 |   0.2206 |
| Improvement                        | same 328 | +0.0537 | +0.0082 |  +0.0663 |

这个结果说明：

```text
PatchCore-guided anomaly crops can improve defect type reasoning
under a fair full-test evaluation setting.
```

### 5.4 Manufacturing-aware explanation generation

制造过程知识不用于直接分类，而用于解释生成。

这是方法设计中的关键解耦：

```text
short visual prompt -> defect type prediction
manufacturing knowledge -> explanation and possible cause reasoning
```

Stage 6.5 在 Stage 6.4 的同一批预测结果上生成结构化解释。

每条解释包含：

* predicted defect type；
* top-2 candidate defect types；
* candidate region；
* anomaly score；
* visual evidence；
* defect family；
* possible manufacturing process；
* possible manufacturing cause；
* inspection focus。

需要注意：

```text
这些解释是 possible-cause reasoning，
不是经过真实工厂工艺数据验证的 causal diagnosis。
```

因此论文中应该谨慎表述为：

```text
manufacturing-aware explanation
possible manufacturing cause
candidate process-level reasoning
```

不要写成：

```text
true root cause diagnosis
```

## 6. 实验章节叙事

实验章节建议按以下逻辑展开。

### 6.1 Dataset and evaluation protocol

说明使用 MVTec AD，并重点分析四个弱类别：

```text
grid, screw, leather, wood
```

这些类别用于 defect type reasoning benchmark。

评价指标包括：

* Top-1 Accuracy；
* Top-2 Accuracy；
* Macro-F1；
* candidate coverage；
* fallback count；
* skipped count。

强调公平比较原则：

```text
Only results under the same image set, same prompt strategy,
same candidate source, and same evaluation protocol are used
as main fair comparisons.
```

### 6.2 PatchCore baseline and failure analysis

先报告 PatchCore baseline：

* image-level 结果强；
* pixel-level localization 仍存在缺陷；
* 弱类别集中在 grid / screw / leather / wood。

然后通过 failure analysis 说明：

```text
PatchCore anomaly map 虽然不能提供完美 mask，
但可以提供有价值的 anomaly candidate regions。
```

### 6.3 Negative and auxiliary localization experiments

这一节解释为什么没有继续沿着 mask refinement 做主贡献。

包括：

| Branch                   | Result                         |
| ------------------------ | ------------------------------ |
| SAM2 refinement          | 不稳定，不能稳定超过 PatchCore component |
| CLIP reranking           | 弱正收益                           |
| region scoring           | 弱正收益                           |
| trainable calibration    | 不稳定                            |
| conservative calibration | 稳定但收益太小                        |

这一节的作用是支撑方法选择：

```text
Instead of over-optimizing anomaly masks,
we use anomaly regions as visual focus for semantic reasoning.
```

### 6.4 Defect type reasoning benchmark

构建 defect type reasoning benchmark：

```text
4 categories
328 abnormal images
20 category-specific defect types
```

缺陷类型包括：

```text
grid: bent / broken / glue / metal_contamination / thread
leather: color / cut / fold / glue / poke
screw: manipulated_front / scratch_head / scratch_neck / thread_side / thread_top
wood: color / combined / hole / liquid / scratch
```

这一节说明我们从“异常检测”进入了“异常理解”。

### 6.5 Prompt and crop ablation

这一节展示：

1. full image prompt reasoning；
2. GT crop upper-bound；
3. real anomaly crop reasoning；
4. full-test anomaly crop reasoning；
5. prompt strategy comparison。

重点强调：

```text
GT crop is only an upper-bound diagnostic.
The main fair result is full-test PatchCore crop reasoning.
```

### 6.6 Main fair comparison

主表放这里。

核心表：

| Setting                            |   Images |   Top-1 |   Top-2 | Macro-F1 |
| ---------------------------------- | -------: | ------: | ------: | -------: |
| generic_label + full_all           |      328 |  0.2850 |  0.4990 |   0.1543 |
| generic_label + crop_topk_ensemble |      328 |  0.3388 |  0.5072 |   0.2206 |
| Improvement                        | same 328 | +0.0537 | +0.0082 |  +0.0663 |

对应结论：

```text
Anomaly-guided crops improve defect type reasoning compared with full-image reasoning,
especially in Top-1 Accuracy and Macro-F1.
```

### 6.7 Manufacturing-aware explanation examples

展示若干解释案例。

每个案例应包含：

* 输入图像；
* PatchCore crop；
* predicted defect type；
* true defect type；
* candidate region；
* visual evidence；
* possible process；
* possible cause。

这一节重点说明方法的解释能力，而不是声称提升分类准确率。

## 7. 论文贡献点写法

当前贡献可以写成三点。

### Contribution 1

We propose a PatchCore-guided visual prompt reasoning pipeline that reuses anomaly localization maps as visual focus for fine-grained defect type reasoning.

### Contribution 2

We construct a defect type reasoning evaluation protocol on weak MVTec AD categories and provide fair full-test comparisons separating realistic settings, partial candidate subsets, and GT-crop upper bounds.

### Contribution 3

We introduce a manufacturing-aware explanation layer that decouples visual defect classification from process-level possible-cause reasoning.

## 8. Introduction 叙事草案

Introduction 可以按如下逻辑写：

1. 工业异常检测已有大量工作，图像级和像素级性能不断提升；
2. 但真实工业场景不仅需要“检测异常”，还需要“理解异常”；
3. 现有 anomaly detection 方法通常不输出缺陷类型和制造原因解释；
4. 视觉语言模型具备语义推理能力，但直接用 full image 或长制造过程 prompt 效果有限；
5. 我们发现 PatchCore anomaly map 可以作为有效的 visual focus；
6. 因此提出 PatchCore-guided visual prompt reasoning；
7. 再通过制造过程知识库生成结构化解释；
8. 实验表明，在同 328 张图的公平设置下，crop_topk_ensemble 比 full_all 提升 Top-1 和 Macro-F1。

## 9. Method 章节叙事草案

Method 可以分为四部分：

### 9.1 PatchCore-guided anomaly candidate generation

介绍如何从 PatchCore anomaly map 得到 top-k candidate crops。

### 9.2 Visual prompt defect type reasoning

介绍不同 prompt strategy：

* generic_label；
* short_visual；
* category_visual；
* visual_ensemble。

最终采用：

```text
generic_label + crop_topk_ensemble
```

作为主方法。

### 9.3 Fair full-test candidate coverage

说明为什么需要 full-test candidate generation，以及如何将覆盖率从 0.50 修复到 0.9970。

### 9.4 Manufacturing-aware explanation generation

说明制造知识库如何映射：

```text
category + predicted defect type -> defect family -> visual evidence -> process -> possible cause
```

## 10. Discussion 重点

Discussion 需要承认当前限制：

1. defect type accuracy 仍然不高；
2. CLIP confidence margin 较小；
3. manufacturing explanation 是 possible-cause reasoning；
4. manufacturing knowledge 不是实际 SOP；
5. 当前只在 MVTec AD 弱类别上验证；
6. 需要未来在更复杂工业数据集上验证。

但同时强调当前价值：

```text
The goal is not to replace anomaly detection backbones,
but to extend anomaly detection outputs toward interpretable industrial anomaly understanding.
```

## 11. 当前论文风险

当前主要风险包括：

### 风险 1：主结果数值不够高

Top-1 从 0.2850 到 0.3388，有提升但绝对值仍低。

应对方式：

```text
强调这是 zero-shot / training-free defect type reasoning；
强调任务从 detection 扩展到 understanding；
强调 Macro-F1 提升明显；
补充 qualitative explanation examples。
```

### 风险 2：制造过程知识可能被认为主观

应对方式：

```text
明确说明它是 structured prior knowledge，
不是 ground-truth causal annotation；
结论写 possible cause，而不是 true cause。
```

### 风险 3：只在 MVTec AD 上做

应对方式：

```text
短期可以把文章定位为 method exploration / diagnostic benchmark；
如果要冲更高层次会议，需要 Stage 7.0 补跨数据集。
```

## 12. 下一步建议

如果目标是先形成论文初稿，下一步应该做：

```text
Stage 6.10: Paper introduction and method section draft
```

如果目标是提高投稿把握，下一步应该做：

```text
Stage 7.0: Cross-dataset validation on a more complex industrial anomaly dataset
```

当前建议优先完成论文初稿，再决定是否补跨数据集。
