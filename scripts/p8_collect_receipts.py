"""Collect lossless receipt annotations. Does not open any original corpus."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def api(route):return json.loads(subprocess.check_output(['gh','api','repos/Igorr112323/Smarty_Chumakov/'+route],text=True))
run=api('actions/runs/39268068500/jobs?per_page=100')
artifacts=api('actions/runs/39266451529/artifacts?per_page=100')['artifacts']
for job in run['jobs']:
    if job['conclusion']!='success':continue
    annotations=api(f"check-runs/{job['id']}/annotations?per_page=100")
    parts=sorted((a['title'],a['message']) for a in annotations if a['title'].startswith('P8 receipt'))
    if not parts:continue
    data=json.loads(gzip.decompress(base64.b64decode(''.join(v for _,v in parts))))
    role=data['role'];folder='official' if role=='p8-official177' else ('external' if role=='p8-external-full' else role.removeprefix('p8-'))
    dest=ROOT/'reports/p8/run39266451529'/folder;dest.mkdir(parents=True,exist_ok=True)
    for name,encoded in data.pop('files').items():
        content=base64.b64decode(encoded)
        assert hashlib.sha256(content).hexdigest()==data['manifest'][name]['sha256']
        (dest/name).write_bytes(content)
    artifact=next(a for a in artifacts if a['name']==role)
    data['artifact']=dict(id=artifact['id'],digest=artifact.get('digest'),bytes=artifact['size_in_bytes'],url=f"https://github.com/Igorr112323/Smarty_Chumakov/actions/runs/39266451529/artifacts/{artifact['id']}")
    data['receipt_job']=job['id']
    (dest/'artifact-manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(folder,'verified')
