# 跨模型结果与误判审计

- `ad2_previous_models/`：历史 AD 模型在 AD2 的结果盘点。
- `ad2_fp_fn/`：PatchCore512 正常验证阈值下的误报/漏报、创新方案。
- `ad2_complementarity/`：PatchCore/DINOv2 逐图错误互补，测试ROC阈值诊断。
- `ad2_pilot/`：早期四类缓存融合，仅作历史诊断。
- `stage24_ad2_highres/`：重算所需的已冻结逐图分数副本。

统计协议不能混用。当前没有恢复训练；EfficientAD 的逐图互补仍待已有权重推理。
