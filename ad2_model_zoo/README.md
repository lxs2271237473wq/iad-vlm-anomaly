# AD2 模型库与训练状态

**最新覆盖状态：用户于 2026-09-15 要求暂停权重训练。RoBiS 及 ISVL 队列已停止，确认相关进程退出、GPU 无计算进程；已有最终权重和检查点保留。下文的运行中/排队描述是暂停前快照，不再代表当前运行状态。未经用户要求，不重启训练。**

状态快照：2026-09-15。训练尚未全部完成；本文件不是精度复现报告。

服务器项目目录：`/root/private_data/iad-vlm-anomaly/ad2_model_zoo`

| 方法 | 官方来源 | 当前状态 | 保存内容 |
|---|---|---|---|
| RoBiS | https://github.com/HUST-SLOW/RoBiS | 正在训练，首个类别 sheet_metal 已有检查点 | `weights/RoBiS/<类别>/latest.pth`，完成后 `model.pth` |
| ISVL：INP 分支 | https://github.com/ISVL119/isvl | 六类别正常数据预处理，等待 RoBiS 完成后串行训练 | `weights/ISVL_INP/` 下按官方实验名保存检查点和最终权重 |
| ISVL：CPR 分支 | 同上 | 尚未启动；需准备 DenseNet、前景、检索及合成数据 | 计划保存 fruit_jelly、vial 两类权重；不能把六类别 INP 当作完整 ISVL |
| SuperAD | https://github.com/Summerdayhurricane/SuperAD | 源码已保存；官方非 register DINOv2-L 权重及记忆库入口待准备 | 冻结骨干权重、16 图参考选择记录、多层正常记忆库；无梯度训练 |
| SuperADD | https://github.com/LukasRoom/SuperADD | 用户确认暂缓，等待官方 DINOv3-H+/16 权重 | 冻结骨干和正常记忆库；已有 B 版适配不算官方 H+ 复现 |

## 运行配置

- RoBiS：官方训练入口，8 类逐类训练，每类 200 epochs、batch 16、输入 518；仅使用 `train/good`。附加首轮及每 10 轮检查点保存，最终保存模型并生成文件哈希。
- RoBiS 的 `latest.pth` 保存 epoch、模型、优化器及配置。它是中间检查点，不能当作完成训练的模型；当前入口未实现自动续训。
- ISVL INP：can、fabric、rice、sheet_metal、wallplugs、walnuts；每类 10 epochs，使用官方切图及正常训练流程，省去测试集读取和逐轮测试。训练完成后保存最终权重。
- ISVL CPR：作者仓库附带 `data/anomaly` 补丁，合成脚本依赖这些补丁。复现时须在协议中披露其使用和来源审计状态，不能宣称与仅正常训练完全相同。尚未使用 AD2 测试异常合成训练样本。
- 独立 venv 位于 `env/`，继承服务器现有 torch 等依赖，不覆盖原实验环境。当前环境并非作者完整锁定版本，结果应称适配复现，待精度验证后判断一致性。
- GPU 串行：RoBiS 完成标记出现后，ISVL INP 才启动；前序进程异常退出时后序暂停并记录错误。

## 代码版本

| 仓库 | 固定 commit |
|---|---|
| RoBiS | c318cf430eaaea6e6f51783a0321d755dd79600c |
| ISVL | a7e52877b466935551c21bdfbb275cf3f305eeec |
| SuperAD | cbe222a7a5668281c3d170ad88332c763e22533f |
| SuperADD | 44cf25144442fbbc1334ea59d1632327a4376d1a |

官方源码保存在 `repos/`，适配训练脚本保存在 `runners/`；RoBiS 源码额外添加参数统计和检查点保存。

## 查看进度

在服务器模型库目录执行：

```bash
tail -c 2000 logs/robis_train_verified.log
tail -c 2000 logs/isvl_inp_queue.log
find weights -name '*.pth' -printf '%p %s bytes\n'
ls results/*COMPLETE.json
```

RoBiS 初次启动因预训练权重传输不完整失败；完整权重经 SHA256 校验后已重新启动，当前日志为 `robis_train_verified.log`，不是旧的 `robis_train.log`。

## 后续验收

1. 确认每类最终权重、哈希、完整轮数和训练日志。
2. 对最终权重进行独立加载与推理检查。
3. 用统一评估入口计算 AD2 指标；训练完成不等于达到论文报告精度。
4. CPR 单独披露额外补丁训练条件；SuperAD/SuperADD 单独记录记忆库大小和预训练骨干，避免把冻结方法当作可训练方法计数。
5. 已反复用于选择方法的 public 集继续标记为开发集，不能将当前结果称为未见测试集上的最终结论。
