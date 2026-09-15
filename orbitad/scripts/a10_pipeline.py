import subprocess,sys
from pathlib import Path
root=Path('/root/private_data/iad-vlm-anomaly')
for script in ['a10_multilayer_baseline.py','a10_evaluate_native.py']:
    print('START',script,flush=True)
    subprocess.run([sys.executable,'-u',str(root/'orbitad/scripts'/script)],cwd=root,check=True)
    print('FINISHED',script,flush=True)
