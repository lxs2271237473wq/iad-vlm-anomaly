# Stage 8-E 论文贡献点与摘要/引言骨架

## 1. 论文问题定义

工业异常检测领域已有大量 anomaly detector 能够输出 image-level anomaly score 或 pixel-level anomaly map，但这些方法通常只回答“是否异常”和“异常在哪里”。

相比之下，视觉语言模型具备一定语义理解能力，但直接对整张工业图像进行 full-image prompting 时，容易受到背景、正常结构和局部小缺陷的干扰，导致异常理解能力不足。

因此，本文关注的问题是：

```text
Can classical anomaly localization serve as an effective visual bridge between industrial anomaly detectors and visual-language reasoning models?
```

换言之，我们不将 anomaly detector 只作为最终检测器，而是将其 anomaly map 转化为 candidate anomaly crops，再将这些局部异常区域输入视觉语言模型进行 reasoning。

## 2. 研究 Gap

现有工作通常沿着两条路线展开：

1. 传统工业异常检测方法关注 image-level detection 和 pixel-level localization，但缺乏对异常语义、缺陷类型或视觉原因的解释能力。
2. 视觉语言模型具备开放语义能力，但 full-image prompting 在工业异常场景中容易被正常区域稀释，尤其当缺陷区域很小或局部纹理变化细微时。

两者之间缺少一个明确、可复现、可评估的连接机制：

```text
anomaly localization -> anomaly candidate crop -> visual-language reasoning
```

本文的实验结果表明，这种 localization-to-reasoning pipeline 可以在多个数据集和多个 anomaly backbone 上带来稳定收益。

## 3. 核心方法一句话

本文提出一种 localization-guided visual-language reasoning framework：先使用工业异常检测器生成 anomaly maps，再从 anomaly maps 中提取 top-k anomaly candidate crops，最后利用视觉语言模型对候选异常区域进行缺陷理解或 normal/anomaly reasoning。

## 4. 三个核心贡献点

| ID | Contribution | Evidence |
|---|---|---|
| Contribution 1 | Propose a framework that uses classical anomaly localization as a visual bridge between industrial anomaly detectors and visual-language reasoning models. | Full-image prompting is consistently weaker than anomaly-crop prompting on MVTec AD and VisA. |
| Contribution 2 | Show that localization-guided crops improve reasoning on both MVTec AD and VisA, under different reasoning tasks. | MVTec AD Top-1 improves from 0.2850 to 0.3388; VisA PatchCore AUROC improves from 0.5950 to 0.8844. |
| Contribution 3 | Show that the crop-reasoning gain is not tied to PatchCore by validating the same reasoning protocol with FastFlow candidates. | On VisA, PatchCore crop AUROC is 0.8844; FastFlow crop AUROC is 0.9222; both improve substantially over full-image AUROC 0.5950. |

## 5. 中文摘要草稿

工业异常检测方法通常能够有效判断产品是否异常并定位异常区域，但难以进一步提供语义层面的异常理解。视觉语言模型虽然具备一定开放语义能力，但在工业检测场景中，直接使用整图提示容易受到背景和正常结构干扰，导致局部缺陷理解能力不足。为此，本文提出一种定位引导的视觉语言异常理解框架，将经典 anomaly detector 生成的 anomaly map 转化为异常候选区域，并将这些局部 anomaly crops 输入视觉语言模型进行缺陷类型判断或 normal/anomaly reasoning。实验表明，在 MVTec AD 上，PatchCore-guided crops 能够提升缺陷类型理解的 Top-1 accuracy；在 VisA 上，PatchCore-guided crops 将 binary anomaly reasoning 的 AUROC 从 0.5950 提升到 0.8844，FastFlow-guided crops 进一步达到 0.9222。这些结果表明，异常定位可以作为传统工业异常检测器与视觉语言模型之间的有效桥梁，并且该机制在跨数据集和跨 anomaly backbone 设置下均具有泛化性。

## 6. 引言结构建议

### 6.1 第一段：工业异常检测的重要性

工业视觉检测要求模型不仅判断产品是否异常，还需要尽可能定位异常区域并辅助理解异常现象。现有 anomaly detection 方法在 image-level detection 和 pixel-level localization 上已经取得显著进展。

### 6.2 第二段：现有 anomaly detector 的局限

PatchCore、FastFlow 等方法能够输出 anomaly scores 或 anomaly maps，但其输出主要停留在检测与定位层面，缺少自然语言层面的解释能力和语义理解能力。

### 6.3 第三段：直接使用 VLM 的问题

视觉语言模型具有开放语义能力，但工业异常通常局部、微小且低占比。直接将整张图像输入 VLM 时，异常区域容易被正常背景稀释，导致 full-image prompting 表现不稳定。

### 6.4 第四段：本文核心想法

本文将 anomaly detector 的 localization ability 与 VLM 的 semantic reasoning ability 连接起来：使用 anomaly map 生成 candidate crops，再将这些 crops 作为视觉语言模型的输入。

### 6.5 第五段：实验验证

本文在 MVTec AD 和 VisA 上验证该思路，并进一步使用 PatchCore 和 FastFlow 两种不同 anomaly backbone 证明该机制不依赖单一检测器。

### 6.6 第六段：贡献总结

最后列出三个贡献点：framework、dataset generalization、backbone generalization。

## 7. 审稿人可能质疑点与回应

| Potential Concern | Response Strategy |
|---|---|
| 这是否只是 PatchCore 的局部裁剪技巧？ | 用 FastFlow 结果回应：FastFlow candidates 也显著提升 reasoning，说明机制不依赖单一 backbone。 |
| 为什么 VisA 不做 defect-type classification？ | VisA 当前标签不提供 MVTec-style fine-grained defect type，因此采用 binary normal/anomaly reasoning 是更公平的设置。 |
| pixel F1 并不高，为什么 reasoning 会提升？ | 本文目标不是 pixel-perfect segmentation，而是利用 candidate crops 聚焦异常区域；高 candidate coverage 已足够支持 reasoning。 |
| 是否能解释制造原因？ | 当前实验验证 visual anomaly-region reasoning，不声称完整 manufacturing-cause discovery。制造原因解释应作为后续扩展或 discussion。 |
| full-image prompting 是否过弱？ | full image 是必要 baseline；实验显示在工业异常中局部异常容易被整图背景稀释，因此 crop guidance 有明确必要性。 |

## 8. 当前最安全的论文主张

```text
Classical anomaly localization can serve as an effective bridge between industrial anomaly detectors and visual-language reasoning models.
```

这个主张有三类证据支撑：

1. MVTec AD 上 defect-type reasoning Top-1 从 0.2850 提升到 0.3388。
2. VisA 上 PatchCore-guided binary reasoning AUROC 从 0.5950 提升到 0.8844。
3. VisA 上 FastFlow-guided binary reasoning AUROC 从 0.5950 提升到 0.9222。

## 9. 不应过度声称

| 不应声称 | 原因 |
|---|---|
| 本文解决了工业异常的像素级精确分割 | pixel F1 仍然有限，本文主贡献是 localization-guided reasoning。 |
| 本文能发现所有未知制造原因 | 当前实验没有验证完整因果发现。 |
| 方法适用于所有 anomaly detector | 当前只验证 PatchCore 和 FastFlow。 |
| GT crop 与 realistic candidate crop 可以直接公平比较 | GT crop 只能作为 upper-bound diagnostic。 |

## 10. 下一步写作建议

下一步应进入 Stage 8-F：论文方法章节草稿。方法章节需要把 pipeline 写清楚，包括 anomaly map generation、candidate extraction、crop-topk ensemble、prompt strategy 和 fallback rule。
