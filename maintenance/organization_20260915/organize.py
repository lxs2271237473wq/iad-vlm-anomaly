"""Reversible, same-filesystem organization. Never deletes experiment assets."""
from pathlib import Path
import os,json,datetime,hashlib,ast,sys
R=Path('/root/private_data/iad-vlm-anomaly').resolve()
AUD=R/'maintenance/organization_20260915'
plan=[]
def add(src,dst):
 p=R/src
 if p.exists() and not p.is_symlink():plan.append({'old':src,'new':dst})
def group(name):
 if name.startswith('stage24'):return 'evidence_fusion'
 if name.startswith('stage25'):return 'tiny_defects'
 return 'srb_qcr'
for base,destkind in [('experiments','src'),('docs','docs'),('results','results'),('runs','runs'),('logs','logs')]:
 for p in sorted((R/base).iterdir()):
  name=p.name
  if name in ['baselines','patchcore_mvtec_summary.csv'] or (base=='docs' and name in ['BASELINE_PLAN.md','DATASETS.md']):continue
  if base=='experiments' and name.startswith('glass'):continue
  add(f'{base}/{name}',f'{group(name)}/{destkind}/{name}')
add('paper','srb_qcr/paper')
add('SRB_QCR_Final_Package.zip','srb_qcr/archives/SRB_QCR_Final_Package.zip')
add('ad2_model_zoo_sources.tar.gz','ad2_model_zoo/archives/ad2_model_zoo_sources.tar.gz')
for old,new in [('glass_env','envs/GLASS'),('glass_official_f788d56','repos/GLASS'),('glass_wheels','dependencies/GLASS_wheels'),('glass_source.tar.gz','archives/glass_source.tar.gz'),('glass_wheels.tar.gz','archives/glass_wheels.tar.gz')]:
 add('experiments/'+old,'ad2_model_zoo/'+new)
add('orbitad/superadd_source.tar.gz','orbitad/archives/superadd_source.tar.gz')

def inside(p):
 assert p.resolve().is_relative_to(R),(str(p),'outside project')
for e in plan:
 inside(R/e['old']);inside(R/e['new']);assert not os.path.lexists(R/e['new']),e
AUD.mkdir(parents=True,exist_ok=True)
(AUD/'plan.json').write_text(json.dumps(plan,indent=2))
if '--apply' not in sys.argv:
 print(json.dumps({'moves':len(plan),'plan':plan},indent=2));sys.exit()
assert not (AUD/'applied.json').exists(),'already applied; do not duplicate'
applied=[];patches=[]
try:
 for e in plan:
  src,dst=R/e['old'],R/e['new'];dst.parent.mkdir(parents=True,exist_ok=True)
  st=src.stat();src.rename(dst)
  src.symlink_to(os.path.relpath(dst,src.parent),target_is_directory=dst.is_dir())
  assert src.resolve()==dst.resolve() and dst.stat().st_ino==st.st_ino
  applied.append({**e,'inode':st.st_ino,'device':st.st_dev})
  (AUD/'applied.json').write_text(json.dumps(applied,indent=2))
 # Only previously audited project-root expressions need adjustment after depth changes.
 for base in [R/'srb_qcr/src',R/'evidence_fusion/src']:
  for p in base.rglob('*.py'):
   s=p.read_text(errors='strict'); old='Path(__file__).resolve().parents[2]'
   if old not in s:continue
   backup=AUD/'code_before'/p.relative_to(R);backup.parent.mkdir(parents=True,exist_ok=True);backup.write_text(s)
   new="next(p for p in Path(__file__).resolve().parents if (p / 'datasets').is_dir() and (p / '.git').is_dir())"
   modified=s.replace(old,new);ast.parse(modified);p.write_text(modified)
   patches.append({'file':str(p.relative_to(R)),'backup':str(backup.relative_to(R)),'before_sha256':hashlib.sha256(s.encode()).hexdigest(),'after_sha256':hashlib.sha256(modified.encode()).hexdigest()})
   (AUD/'patches.json').write_text(json.dumps(patches,indent=2))
except Exception:
 print('Partial operation recorded; use rollback.py with applied.json',file=sys.stderr);raise
# Existing numbered stages remain intact. Add browsable project documentation.
desc={
 'srb_qcr':('SRB/QCR 与早期 AD 实验','早期 PatchCore、VLM、QCR/SRB、Stage7–23、论文与消融记录。'),
 'evidence_fusion':('Stage24 表征与融合探索','分辨率、局部学习、外观变化、参考检索、对应与配对干预。日期命名结果保留原样，避免改写实验版本。'),
 'tiny_defects':('Stage25 微小缺陷专项','微小缺陷划分、裁剪控制、局部学习、DINOv2 与冻结迁移。'),
}
for name,(title,description) in desc.items():
 p=R/name;p.mkdir(exist_ok=True)
 for sub in ['src','configs','docs','results','runs','logs','checkpoints']:(p/sub).mkdir(exist_ok=True)
 (p/'README.md').write_text(f'# {title}\n\n{description}\n\n## 目录\n\n- `src/`：本实验代码。\n- `docs/`：方法、报告和说明。\n- `results/`：预测、指标和分析。\n- `runs/`：历史运行产物；旧权重仍在各次运行的 weights 下，不重复搬动。\n- `logs/`：日志。\n- `configs/`、`checkpoints/`：后续本实验配置与权重的固定入口，现有记录保持原结构。\n\n公共数据集、预训练权重、基线配置在总目录的 datasets、models、configs。历史脚本默认从总目录执行，旧 experiments/results/runs 等入口已保留兼容链接。\n\n本次仅整理文件，未启动训练。详细迁移记录见总目录 maintenance/organization_20260915。\n',encoding='utf-8')
# Stage25 reuses the Stage24 runtime; retain one source of truth, explicitly exposed.
shared=R/'tiny_defects/src/stage24_runtime'
if not shared.exists():shared.symlink_to('../../evidence_fusion/src/stage24_evidence',target_is_directory=True)
rootdoc='''# iad-vlm-anomaly 实验总目录

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

权重、数据、预测文件均未重写。仅14个以内/以迁移清单为准的旧脚本根目录定位表达式可能因目录层级改变而修订，修改前原文已备份；算法未改动。整理通过路径与语法核验，不等同于重跑全部历史实验。Git 未提交、未推送。
'''
(R/'README.md').write_text(rootdoc,encoding='utf-8')
(AUD/'summary.json').write_text(json.dumps({'moves':len(applied),'root_locator_patches':len(patches),'training_started':False,'status':'applied_pending_verification'},indent=2))
print(json.dumps({'moves':len(applied),'patches':len(patches)}))
