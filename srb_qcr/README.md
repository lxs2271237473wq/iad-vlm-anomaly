# SRB/QCR 与早期 AD 实验

早期 PatchCore、VLM、QCR/SRB、Stage7–23、论文与消融记录。

## 目录

- `src/`：本实验代码。
- `docs/`：方法、报告和说明。
- `results/`：预测、指标和分析。
- `runs/`：历史运行产物；旧权重仍在各次运行的 weights 下，不重复搬动。
- `logs/`：日志。
- `configs/`、`checkpoints/`：后续本实验配置与权重的固定入口，现有记录保持原结构。

公共数据集、预训练权重、基线配置在总目录的 datasets、models、configs。历史脚本默认从总目录执行，旧 experiments/results/runs 等入口已保留兼容链接。

本次仅整理文件，未启动训练。详细迁移记录见总目录 maintenance/organization_20260915。


## 整理后的阶段导航

[各阶段入口](STUDIES.md)。`studies/` 将每个阶段的代码和结果集中展示；实体保持唯一，旧路径兼容。补充审计文档见 `docs/desktop_reports/`。
