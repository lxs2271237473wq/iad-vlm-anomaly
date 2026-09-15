from pathlib import Path
import json,os,tarfile,ast,hashlib,subprocess,re
R=Path('/root/private_data/iad-vlm-anomaly');A=R/'maintenance/organization_20260915'
with tarfile.open(A/'reports.tar.gz') as t:
 for m in t.getmembers():
  p=R/m.name;assert p.resolve().is_relative_to(R.resolve()) and m.isfile()
  p.parent.mkdir(parents=True,exist_ok=True)
  content=t.extractfile(m).read()
  if p.exists():assert p.read_bytes()==content,('refuse overwrite',str(p))
  else:p.write_bytes(content)
for name in ['organize.py','rollback.py','finalize.py']:
 p=Path('/root/private_data')/name
 if p.exists():(A/name).write_bytes(p.read_bytes())
p=R/'maintenance_plan_preview.json'
if p.exists():p.rename(A/'preview.json')
(R/'model_comparison/README.md').write_text('''# 跨模型结果与误判审计

- `ad2_previous_models/`：历史 AD 模型在 AD2 的结果盘点。
- `ad2_fp_fn/`：PatchCore512 正常验证阈值下的误报/漏报、创新方案。
- `ad2_complementarity/`：PatchCore/DINOv2 逐图错误互补，测试ROC阈值诊断。
- `ad2_pilot/`：早期四类缓存融合，仅作历史诊断。
- `stage24_ad2_highres/`：重算所需的已冻结逐图分数副本。

统计协议不能混用。当前没有恢复训练；EfficientAD 的逐图互补仍待已有权重推理。
''',encoding='utf-8')
# Stage indices provide one browseable folder per idea without duplicating outputs.
def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not os.path.lexists(dst):dst.symlink_to(os.path.relpath(src,dst.parent),target_is_directory=src.is_dir())
def study(project,key,title,sources,results,docs=()):
 d=R/project/'studies'/key;d.mkdir(parents=True,exist_ok=True)
 for s in sources:link(s,d/'src'/s.name)
 for s in results:link(s,d/'results'/s.name)
 for s in docs:link(s,d/'docs'/s.name)
 (d/'README.md').write_text(f'# {title}\n\n本目录集中展示该阶段代码与结果，使用符号链接引用唯一实体，不复制数据。历史文件名和实验版本保留。\n\n运行历史脚本时请在 iad-vlm-anomaly 总目录执行。此索引不表示实验成功或已完成。\n',encoding='utf-8')
titles=['数据与条件审计','表征和异常分数敏感性','低频/反事实干预','分辨率与路由','OCC 校准','正常变化子空间','RNS 正常残差','关系表征','空间漂移与修复','上下文正常参考','多层高分辨率基线','固定融合与自校准']
for i,title in enumerate(titles):
 pref=f'a{i}_';study('orbitad',f'a{i:02d}',title,list((R/'orbitad/scripts').glob(pref+'*')),list((R/'orbitad/results').glob(pref+'*')))
src=R/'evidence_fusion/src/stage24_evidence'
for p in sorted((R/'evidence_fusion/results/stage24_evidence').iterdir()):
 if p.is_dir():
  matching=[x for x in src.glob('*') if x.suffix in ['.py','.sh'] and p.name in x.read_text(errors='replace')]
  study('evidence_fusion',p.name,'Stage24 '+p.name,matching,[p])
tiny_names={'v1':['build_tiny_defect_protocol.py'],'candidate_v1':['audit_tiny_candidates.py'],'crop_control_v1':['evaluate_local_crop_control.py'],'shallow_local_v1':['train_shallow_local.py'],'dinov2_ad2_v1':['run_dinov2_ad2.py','analyze_dino_ad2_groups.py'],'dinov2_baseline_v1':['run_dinov2_baseline.py'],'dinov2_baseline_v1_offline':['run_dinov2_baseline_offline.py'],'dinov2_validation_v1':['run_dinov2_validation.py'],'dinov2_holdout_v1':['run_dinov2_holdout.py'],'dino_complement_v1':['check_dino_complement.py']}
for p in sorted((R/'tiny_defects/results/stage25_tiny_defects').iterdir()):
 if p.is_dir():study('tiny_defects',p.name,'微小缺陷 '+p.name,[src/n for n in tiny_names.get(p.name,[]) if (src/n).exists()],[p])
keys=set()
for kind in ['src','docs','results','runs']:
 for p in (R/'srb_qcr'/kind).iterdir():
  if p.is_dir():keys.add(p.name)
for k in sorted(keys):
 study('srb_qcr',k,k,[R/'srb_qcr/src'/k] if (R/'srb_qcr/src'/k).exists() else [],[p for p in [R/'srb_qcr/results'/k,R/'srb_qcr/runs'/k] if p.exists()], [R/'srb_qcr/docs'/k] if (R/'srb_qcr/docs'/k).exists() else [])
for project in ['srb_qcr','evidence_fusion','tiny_defects','orbitad']:
 p=R/project/'STUDIES.md'
 p.write_text('# 各阶段入口\n\n'+''.join(f'- [{d.name}](studies/{d.name}/README.md)\n' for d in sorted((R/project/'studies').iterdir())),encoding='utf-8')
# Verify rename identity + all new links + patched code syntax, without GPU execution.
checks=[]
for e in json.loads((A/'applied.json').read_text()):
 p,q=R/e['old'],R/e['new'];assert p.is_symlink() and p.resolve()==q.resolve()
 assert q.stat().st_ino==e['inode'] and q.stat().st_dev==e['device']
 checks.append(e['old'])
for e in json.loads((A/'patches.json').read_text()):
 p=R/e['file'];ast.parse(p.read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==e['after_sha256']
bad=[];count=0
for project in ['srb_qcr','evidence_fusion','tiny_defects','orbitad']:
 for root,dirs,files in os.walk(R/project/'studies',followlinks=False):
  for name in dirs+files:
   p=Path(root)/name
   if p.is_symlink():
    count+=1
    if not p.exists():bad.append(str(p))
assert not bad,bad
assert (R/'tiny_defects/src/stage24_runtime/common.py').exists()
for p in [R/'ad2_model_zoo/env/bin/python',R/'experiments/glass_env/bin/python']:
 assert p.exists(),str(p)
 v=subprocess.run([str(p),'-c','import sys; print(sys.version.split()[0])'],capture_output=True,text=True,timeout=20)
 assert v.returncode==0,v.stderr
report={'status':'verified','moved_entities':len(checks),'code_root_locator_patches':len(json.loads((A/'patches.json').read_text())),'study_links_checked':count,'broken_study_links':bad,'verification':'original inode/device retained; compatibility link target; Python AST + hash; both venv Python entrypoints','training_started':False,'limitations':'Not full re-execution of historical experiments; legacy paths retained; study source matching for date runs is heuristic.'}
(A/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
# Navigation additions without overwriting existing project method descriptions.
for project in ['srb_qcr','evidence_fusion','tiny_defects','orbitad']:
 p=R/project/'README.md';s=p.read_text();addition='\n\n## 整理后的阶段导航\n\n[各阶段入口](STUDIES.md)。`studies/` 将每个阶段的代码和结果集中展示；实体保持唯一，旧路径兼容。补充审计文档见 `docs/desktop_reports/`。\n'
 if '## 整理后的阶段导航' not in s:p.write_text(s+addition,encoding='utf-8')
p=R/'README.md';s=p.read_text().replace('仅14个以内/以迁移清单为准的旧脚本根目录定位表达式可能因目录层级改变而修订','14 个旧脚本的根目录定位表达式因目录层级改变而修订');p.write_text(s)
