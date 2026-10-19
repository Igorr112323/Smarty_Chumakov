"""Context-only migration: retain every existing metric value; no inference/fit."""
from __future__ import annotations
import json
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[1]
INDEX='reports/EXPERIMENTS_INDEX.md'

def is_run(path):
    return any(re.fullmatch(r'run\d+',p) for p in path.parts) or any((p/'run.json').is_file() for p in path.parents if p!=ROOT)

def metric_file(path,data):
    if path.name in ('weights.json','combined_bundle.json') or 'model' in path.name or 'calibrator_v1' in path.name:return False
    def contains(x):
        if isinstance(x,dict):return any(k.lower() in ('f1','mae','strict_f1','token_f1','calibrated_auc') for k in x) or any(contains(v) for v in x.values())
        return isinstance(x,list) and any(contains(v) for v in x)
    return contains(data)

def objects(data):
    return [data] if isinstance(data,dict) else [x for x in data if isinstance(x,dict)]

def source_id(data):
    if isinstance(data,dict):
        for key in ('run_id','source_run','run'):
            v=data.get(key)
            if isinstance(v,(str,int)) and str(v).isdigit() and int(v)>1_000_000_000:return str(v)
        for v in data.values():
            found=source_id(v)
            if found:return found
    if isinstance(data,list):
        for v in data:
            found=source_id(v)
            if found:return found
    return None

def migrate():
    records=[]
    for folder in (ROOT/'reports',ROOT/'spanverify-module/reports'):
        for p in folder.rglob('*.json'):
            if is_run(p) or p.name=='evidence-provenance.json':continue
            try:data=json.loads(p.read_text())
            except (ValueError,UnicodeError):continue
            if not metric_file(p,data):continue
            rid=source_id(data)
            commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(p.relative_to(ROOT))],cwd=ROOT,text=True).strip()
            date=subprocess.check_output(['git','log','-1','--format=%cI','--',str(p.relative_to(ROOT))],cwd=ROOT,text=True).strip()[:10] or '2026-10-19'
            for obj in objects(data):
                if 'evidence_context' not in obj:
                    obj['evidence_context']=dict(run_id=rid or f'historical-local-unrecorded:{commit or "new"}',date=date,status='NON_AUTHORITATIVE historical/reference; consult current indices',successor=INDEX,source_commit=commit or None,run_id_scope='Referenced subrun only for historical aggregates; not all values share one run',date_scope='file provenance date, not assertion of every underlying measurement date',limitation=None if rid else 'Original remote run not recorded; do not misrepresent as current independent measurement')
            p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
            records.append(dict(path=str(p.relative_to(ROOT)),run_id=rid,date=date))
    (ROOT/'reports/p8/evidence-provenance.json').write_text(json.dumps(dict(run_id='P8-hygiene',date='2026-10-19',index=INDEX,files=records),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':migrate()
