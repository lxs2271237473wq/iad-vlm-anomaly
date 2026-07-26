# Stage 23 阶段性备忘录：AD2 冻结迁移与真实选择性运行

## 1. 阶段目标

Stage 23 将在 VisA PatchCore 类别 LOCO 上选定的 SRB-QCR 配置，无目标标签调参地迁移到 AD2 四类别协议：

- `fruit_jelly`
- `sheet_metal`
- `vial`
- `walnuts`

冻结参数：

```text
w_max = 0.35
q_quantile = 0.25
tau_delta = 0.75
non-inferiority margin = 0.002
```

## 2. 已完成工作

### Stage 23-B1：AD2 冻结镜像评估

- 目标标签未参与参数选择。
- SRB-QCR 相对检测器的宏平均 AUROC 增量：`+0.0086`
- SRB-QCR 相对 crop VLM：`+0.1415`
- SRB-QCR 相对朴素融合：`-0.0347`
- SRB-QCR 相对旧 Quality-QCR：`-0.0252`
- SRB-QCR 相对旧 Adaptive-QCR：`-0.0255`
- 相对检测器类别胜数：`3/4`
- 最差类别增量：`-0.0037`
- 预门控潜在调用节省率：`25.10%`

结论边界：AD2 支持“冻结迁移与检测器保持”叙事，但不支持“SRB 在所有目标上均优于更激进融合”。

### Stage 23-C0/C1：运行资产与历史来源审计

- 图像数：`243`
- Full 逻辑 VLM 调用：`243`
- Selective 调用：`182`
- 节省调用：`61`
- Full crop 评估：`502`
- Selective crop 评估：`397`
- 节省 crop：`105`
- 所有 243 张图像的运行资产均可解析。
- 所有 182 个 gate-on 样本均使用上下文 crop。
- fallback 语义保持严格成立。

历史来源审计确认，冻结表中的 `M` 来自：

```text
stage11_d_vlm_image_predictions.csv
context_topk_mean_score
```

但候选级 CSV 与当前文件系统 crop 资产属于不同资产版本，不能通过路径、basename 或 SHA256 建立逐项身份桥接。最终采用文件系统 sibling crop 恢复运行资产。

### Stage 23-C2：三轮真实 GPU 计时

| 轮次 | Full调用 | Selective调用 | Full crops | Selective crops | Full秒 | Selective秒 | 节省秒 | 时间节省率 | 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 243 | 182 | 502 | 397 | 297.023 | 240.746 | 56.277 | 18.947% | 1.234× |
| 2 | 243 | 182 | 502 | 397 | 287.022 | 232.422 | 54.600 | 19.023% | 1.235× |
| 3 | 243 | 182 | 502 | 397 | 287.431 | 243.133 | 44.298 | 15.412% | 1.182× |
| **中位数** | — | — | — | — | **287.431** | **240.746** | **54.600** | **18.947%** | **1.234×** |

锁定结论：

- VLM 调用节省率：`25.10%`
- crop 评估节省率：`20.92%`
- 中位 wall-clock 时间节省率：`18.95%`
- 中位加速：`1.234×`
- 峰值 GPU 已分配显存约：`602.2 MiB`

## 3. 运行重放的来源边界

三轮计时完整执行，但结束后的历史分数审计显示：

- 243 行中 242 行的 raw `M` 差异超过 `1e-4`
- 最大差异：`0.026931643`
- 平均差异：`0.009285855`
- 差异具有明显类别依赖性，`walnuts` 最突出
- 全局翻转 margin 正负号不能解决问题

最合理的来源解释是：原始 Stage 11 的类别特定提示词或文本特征配置未被完整保存。当前运行脚本能够锁定 crop 数量、门控决策和真实执行成本，但不能声称逐样本精确复现历史 VLM margin。

因此论文必须严格区分：

```text
准确率与 Bootstrap：使用 Stage 23-B1 锁定预测缓存
真实运行效率：使用 Stage 23-C2 三轮重新执行
```

允许表述：

> The runtime execution validates the locked crop counts and selective invocation decisions. Exact replay of historical AD2 VLM margins is not claimed because the original category-specific prompt metadata was not fully preserved.

禁止表述：

- “AD2 运行重放在数值精度范围内复现历史分数”
- “通过放宽 tolerance 证明等价”
- “在 AD2 标签上重新选择提示词或参数”

## 4. 论文整合状态

本阶段已回填到中英文整合稿：

- AD2 冻结迁移主结论
- AD2 真实三轮运行时间
- 新增图6：AD2 Full/Selective 运行时间
- 新增表10：AD2 三轮配对计时
- 数据与代码可用性
- 运行来源边界
- 新增局限性条目
- 更新结论、实验清单与 claim 边界表

未填写且不能由实验自动推断的项目：

- 作者姓名
- 作者单位和邮箱
- 目标会议/期刊
- Funding
- Author contributions
- AI 辅助工具披露的最终措辞
- 目标 venue 的参考文献格式

## 5. GitHub 状态

远端：

```text
https://github.com/lxs2271237473wq/iad-vlm-anomaly
```

Stage 23-C2 整合基准提交：

```text
bf049d750898c793211fdda3bf6aff30bf483f07
```

截图中的 `git ls-remote origin refs/heads/main` 与本地 `HEAD` 均指向该提交，说明 push 已成功且本地/远端 main 一致。

## 6. 下一步

1. 在仓库中应用本最终包。
2. 在服务器数据环境运行 `build_stage23_d_ad2_qualitative_figure.py`，生成 AD2 四案例图与 manifest。
3. 人工核对四个案例的视觉可解释性后，再决定是否放入正文或附录。
4. 补作者信息与目标 venue 模板。
5. 最后一次 Git 提交并 push。
