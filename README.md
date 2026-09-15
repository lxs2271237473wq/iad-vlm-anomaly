# iad-vlm-anomaly 实验总目录

## 实验入口

| 文件夹 | 内容 |
|---|---|
| [srb_qcr](srb_qcr/README.md) | 早期 AD 基线、VLM、QCR/SRB、Stage7–23 与论文 |
| [evidence_fusion](evidence_fusion/README.md) | Stage24 多尺度、局部表征、参考检索与融合 |
| [tiny_defects](tiny_defects/README.md) | Stage25 微小缺陷专项、裁剪控制与迁移 |
| [orbitad](orbitad/README.md) | A0–A11 正常变化、局部残差与多层基线探索 |
| [ad2_model_zoo](ad2_model_zoo/README.md) | 外部 AD2 方法源码、依赖、预训练与训练权重；训练已暂停 |
| [model_comparison](model_comparison/README.md) | 跨实验的历史成绩、FP/FN 与模型互补分析 |

## 公共资源

- `datasets/`：数据集，原地保留。
- `models/`：通用预训练模型。
- `configs/`：通用基线配置；实验私有配置放各实验 configs。
- `knowledge/`：领域知识文件。
- `third_party/`：现有公共第三方依赖。
- `scripts/`：公共辅助脚本。
- `downloads/`、`backups/`：数据下载与原有备份，不删除、不重复复制。
- `docs/DATASETS.md`、`docs/BASELINE_PLAN.md`：公共数据/基线说明。
- `experiments/baselines`、`results/baselines`、`runs/baselines`：通用基线代码和已有产物。
- `maintenance/organization_20260915/`：迁移、核验和回滚记录。

## 路径兼容

`experiments/`、`results/`、`runs/`、`docs/`、`logs/` 中的实验专属项目现为兼容符号链接，实体已移到上述实验文件夹。不要删除兼容链接；历史脚本及 JSON 中仍引用它们。浏览与新实验优先使用新入口，运行旧脚本仍从本总目录执行。

权重、数据、预测文件均未重写。14 个旧脚本的根目录定位表达式因目录层级改变而修订，修改前原文已备份；算法未改动。整理通过路径与语法核验，不等同于重跑全部历史实验。Git 未提交、未推送。
