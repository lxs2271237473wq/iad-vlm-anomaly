"""Restore moved entities and backed-up code; never deletes experiment outputs."""
from pathlib import Path
import json,os
R=Path('/root/private_data/iad-vlm-anomaly').resolve();A=R/'maintenance/organization_20260915'
if (A/'gitignore.before').exists():
 (R/'.gitignore').write_bytes((A/'gitignore.before').read_bytes())
for e in json.loads((A/'patches.json').read_text()) if (A/'patches.json').exists() else []:
 p=R/e['file'];b=R/e['backup'];assert p.resolve().is_relative_to(R);p.write_bytes(b.read_bytes())
for e in reversed(json.loads((A/'applied.json').read_text())):
 src,dst=R/e['old'],R/e['new']
 assert src.is_symlink() and src.resolve()==dst.resolve(),e
 assert dst.resolve().is_relative_to(R),e
 src.unlink();dst.rename(src)
(A/'ROLLED_BACK.txt').write_text('Moved entities restored. Generated docs/directories retained for review.\n')
print('Restored original paths; generated documentation retained.')
