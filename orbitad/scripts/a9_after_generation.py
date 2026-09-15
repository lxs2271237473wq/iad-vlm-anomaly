import os,time,subprocess
from pathlib import Path
root=Path('/root/private_data/iad-vlm-anomaly')
done=root/'orbitad/results/a9_context_reference_v1/COMPLETE.json'
while not done.exists():
    try: os.kill(63438,0)
    except ProcessLookupError: raise RuntimeError('A9 generation stopped without COMPLETE; evaluation cancelled')
    time.sleep(15)
subprocess.run(['/opt/conda/bin/python','-u','orbitad/scripts/a9_evaluate_diagnostic.py'],cwd=root,check=True)
