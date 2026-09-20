# iad-vlm-anomaly

面向工业异常检测的实验仓库，当前重点是 AD2 上的微小缺陷定位与正常局部干扰区分。仓库保留从早期 VLM/QCR 融合、Stage24 证据融合、Stage25 微小缺陷分析，到 OrbitAD 跨分辨率近邻检测的完整研究轨迹。

> 本仓库只保存源码、配置、轻量指标和实验说明。数据集、模型权重、特征缓存、逐图预测、日志与第三方源码不上传。

## 从哪里开始

| 入口 | 作用 | 当前定位 |
|---|---|---|
| [docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md) | 整个实验故事、目录映射和复现顺序 | 总导航 |
| [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) | 2026-09-20 方法审计与证据边界 | 投稿判断依据 |
| [srb_qcr](srb_qcr/README.md) | 早期 AD 基线、VLM、QCR/SRB、Stage7–23 | 历史路线 |
| [evidence_fusion](evidence_fusion/README.md) | Stage24 多尺度、局部表征、参考检索与融合 | 机制探索 |
| [tiny_defects](tiny_defects/README.md) | Stage25 微小缺陷专项、裁剪控制与迁移 | 问题聚焦 |
| [orbitad](orbitad/README.md) | A0–A73 正常变化、多层记忆与跨分辨率融合 | 当前主线 |
| [model_comparison](model_comparison/README.md) | EfficientAD 等模型的 FP/FN 与互补性分析 | 基线证据 |
| [ad2_model_zoo](ad2_model_zoo/README.md) | 外部方法的配置和训练入口 | 模型复现 |

## 当前候选方法

当前候选采用冻结的 DINOv2 多层 patch 特征和共享正常记忆库。测试图分别经过完整图分支与规则裁块高分辨率分支，得到全局图 `G` 和局部图 `T`；独立正常校准图估计类别尺度 `qG`、`qT`，候选融合为：

```text
F = 0.5 * (G / qG + T / qT)
```

这一公式仍是候选方法。A69 的七个 AD2 类别存在测试图与校准分辨率不一致，不能作为已验证的主结果；A72 Cable 是配置一致的初步正结果，但还不是盲测。详见 [当前状态](docs/CURRENT_STATUS.md)。

## 目录约定

- `scripts/`、各模块的 `src/` 与 `scripts/`：可复现代码。
- `configs/`：通用配置；模块私有配置放在模块自己的 `configs/`。
- `results/`、`runs/`：Git 只保留 CSV、JSON、Markdown、TXT 等轻量摘要。
- `datasets/`、`models/`、`checkpoints/`、`weights/`：本地资源，不进入 Git。
- `work/`：临时运行与审计工作区，不进入 Git。
- `maintenance/organization_20260915/`：上一次目录迁移、核验和回滚记录。

历史兼容符号链接仍保留，旧脚本应从仓库根目录运行。整理只改变材料归类与导航，不代表重新运行或验证了全部历史实验。

## 本地资源准备

1. 按 [docs/DATASETS.md](docs/DATASETS.md) 放置数据集。
2. 将 DINOv2 等权重放入本地 `models/`、`ad2_model_zoo/pretrained/` 或相应模块的 `checkpoints/`。
3. 按具体模块 README 运行；外部模型源码需单独克隆到本地忽略目录。

仓库不会提供或下载数据集、预训练权重和训练后权重。
