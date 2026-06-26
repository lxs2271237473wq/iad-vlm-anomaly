# Stage 8-F 论文方法章节草稿

## 1. 方法总览

本文提出一种定位引导的视觉语言异常理解框架。该框架不直接将整张工业图像输入视觉语言模型，而是首先利用工业异常检测器生成 anomaly map，再从 anomaly map 中提取异常候选区域，并将这些局部区域作为视觉输入传递给视觉语言模型进行 reasoning。

整体流程可以概括为：

```text
input image
  -> anomaly detector
  -> anomaly map
  -> candidate region extraction
  -> anomaly crops
  -> VLM prompt reasoning
  -> defect-type or normal/anomaly prediction
```

该设计的核心动机是：工业异常通常只占图像中的很小局部区域。直接使用 full-image prompting 时，异常区域容易被正常背景、规则结构和大面积正常纹理稀释。相比之下，anomaly crops 能够显式聚焦可疑区域，使视觉语言模型更容易进行异常语义判断。

## 2. Anomaly Map Generation

给定测试图像，首先使用传统工业异常检测器生成 image-level anomaly score 和 pixel-level anomaly map。本文实验中使用 PatchCore 和 FastFlow 作为 candidate generators。

PatchCore 代表 memory-bank nearest-neighbor 类 anomaly detector，FastFlow 代表 flow-based anomaly detector。二者机制不同，因此可以用于验证本文方法是否依赖单一 anomaly backbone。

形式化地，对于输入图像 `x`，anomaly detector 输出：

```text
s_img, A = f_detector(x)
```

其中 `s_img` 是图像级异常分数，`A` 是像素级 anomaly map。

## 3. Candidate Region Extraction

得到 anomaly map 后，本文将其转化为显式候选区域。具体做法是对 anomaly map 进行阈值化，并提取 connected components。随后，根据区域的 anomaly response 对候选区域排序，保留 top-k candidate regions。

该步骤将连续的 anomaly map 转化为离散的 candidate boxes，使后续视觉语言模型能够聚焦于局部异常区域。

候选区域提取过程包括：

1. 对 anomaly map 进行阈值化。
2. 提取 connected components。
3. 过滤面积过小的噪声区域。
4. 根据 component rank 或 anomaly response 保留 top-k regions。

## 4. Anomaly Crop Construction

对于每一个 candidate box，本文从原图中裁剪对应局部区域。为了避免候选框过紧导致缺陷上下文丢失，裁剪时加入 padding；同时设置 minimum crop size，避免极小 crop 导致视觉语言模型输入质量下降。

该步骤输出一组 anomaly crops：

```text
C = {c_1, c_2, ..., c_k}
```

其中每个 `c_i` 对应一个候选异常区域。

## 5. Prompt-based VLM Reasoning

得到 full image 或 anomaly crops 后，本文使用视觉语言模型计算图像特征与文本 prompt 特征之间的相似度。

在 MVTec AD 上，任务是 defect-type reasoning，即判断异常属于哪一类细粒度缺陷类型。在 VisA 上，由于当前标签不提供 MVTec-style fine-grained defect type，因此采用 binary normal/anomaly reasoning。

对于 VisA，本文主要使用 inspection-style binary prompts，例如：

```text
normal prompt: a quality inspection image of a normal object
anomaly prompt: a quality inspection image of a defective object
```

最终 anomaly score 可由 anomaly prompt 与 normal prompt 的相似度差值得到：

```text
score(c_i) = sim(c_i, prompt_anomaly) - sim(c_i, prompt_normal)
```

## 6. Crop-topk Ensemble

单个 top-1 candidate crop 并不一定总是语义上最有用的区域。因此，本文采用 crop-topk ensemble：对 top-k anomaly crops 分别计算 anomaly score，然后使用最大 anomaly margin 作为图像级 reasoning score。

```text
score(x) = max_i score(c_i),  i = 1, ..., k
```

该设计可以提升鲁棒性，尤其当 anomaly detector 的最高响应区域不是最适合 VLM 判断的区域时，top-k ensemble 能够降低单一候选框错误带来的影响。

## 7. Fallback Rule

为了保证评价公平，本文不只在成功生成 crop 的图像上进行评估。若某张图像没有可用 candidate region，则使用 full image 作为 fallback 输入。

因此，本文的 evaluation 是 full-test setting，而不是 selective evaluation。

```text
if candidate regions exist:
    use anomaly crops
else:
    use full image
```

该规则避免了只统计容易样本或只统计有 crop 样本所造成的结果偏高。

## 8. Algorithm

```text
Algorithm: Localization-guided VLM Reasoning

Input:
  test image x
  anomaly detector f
  visual-language model g
  prompt set P
  top-k candidate number k

Output:
  reasoning score or predicted label

1. Generate anomaly map:
      s_img, A = f(x)

2. Extract candidate regions:
      B = ConnectedComponents(Threshold(A))
      B_topk = SelectTopK(B, k)

3. Construct visual inputs:
      if B_topk is not empty:
          C = Crop(x, B_topk)
      else:
          C = {x}

4. Compute VLM scores:
      for each visual input c in C:
          score(c) = sim(g_img(c), g_txt(P_anomaly))
                     - sim(g_img(c), g_txt(P_normal))

5. Ensemble:
      score(x) = max_c score(c)

6. Predict label using score(x).
```

## 9. Difference from Existing Settings

### 9.1 Difference from classical anomaly detection

传统 anomaly detector 主要输出 anomaly score 或 anomaly map，目标是检测和定位。本文并不将 detector 的输出作为终点，而是将 anomaly map 转化为 candidate crops，用作视觉语言模型 reasoning 的输入。

### 9.2 Difference from direct VLM prompting

直接 VLM prompting 使用整张图像作为输入，容易受到正常背景干扰。本文通过 anomaly detector 提供的 localization prior，将 VLM 的视觉注意力集中到可疑区域。

### 9.3 Difference from GT crop evaluation

GT crop 使用真实 mask 或人工标注区域，只能作为 upper-bound diagnostic。本文的主实验使用 anomaly detector 自动生成的 candidate crops，因此属于 realistic setting。

## 10. 方法边界

本文方法依赖 candidate localization 的质量。如果 anomaly detector 不能覆盖真实异常区域，则 crop reasoning 的效果会下降。因此，candidate coverage 和 candidate quality 是本文方法的重要影响因素。

此外，本文验证的是 visual anomaly-region reasoning，而不是完整 manufacturing-cause discovery。制造原因推理需要额外的过程知识、工艺先验或因果标注支持。

## 11. 写入正式论文时的建议

正式论文方法章节建议按以下结构组织：

1. Overview
2. Anomaly Localization Backbone
3. Candidate Region Extraction
4. Localization-guided VLM Reasoning
5. Crop-topk Ensemble and Fallback Rule
6. Implementation Details
