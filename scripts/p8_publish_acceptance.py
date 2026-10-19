"""Final archive from acceptance sources + original evidence; never fit or infer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]
def run_info(n):return json.loads(subprocess.check_output(['gh','run','view',str(n),'--json','status,conclusion,headSha,url'],text=True))
def validate():
    delivery=json.loads((ROOT/'reports/p8/DELIVERY.json').read_text())
    runs={role:run_info(n) for role,n in {'evaluation':39266451529,'windows':delivery['windows_run'],'ci':delivery['ci_run']}.items()}
    for role,info in runs.items():
        assert info['status']=='completed' and info['conclusion']=='success',(role,info)
        subprocess.run(['git','diff','--exit-code',info['headSha'],'HEAD','--','spanverify-module/spanverify','spanverify-module/config','spanverify-module/reports/experiments/p5_span_filter'],check=True)
    quality=json.loads((ROOT/'reports/p8/run39266451529/official/result.json').read_text())
    assert quality['joint_gate'] is True
    return dict(runs=runs,windows_run=delivery['windows_run'],tests_excluded=False,documentation_source=os.environ['GITHUB_SHA'],release_tag='p8-final-20261019')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args();meta=validate()
    if args.prepare:
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write(f"windows_run={meta['windows_run']}\n")
        return
    out=ROOT/'.arena/p8-publication/p8-final-acceptance.zip';out.parent.mkdir(parents=True,exist_ok=True)
    checks={}
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        for name in files:
            p=ROOT/name
            if not name or not p.is_file():continue
            z.write(p,'source/'+name)
        for folder,prefix in [(ROOT/'.arena/p8-publication/windows','delivery'),(ROOT/'.arena/p8-publication/evidence','evidence')]:
            for p in sorted(folder.rglob('*')):
                if p.is_file():
                    name=prefix+'/'+str(p.relative_to(folder));z.write(p,name);checks[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        z.writestr('FINAL-PUBLICATION.json',json.dumps(dict(meta,files=checks),ensure_ascii=False,indent=2))
    digest=hashlib.sha256(out.read_bytes()).hexdigest();sha=out.with_suffix('.zip.sha256');sha.write_text(f'{digest}  {out.name}\n')
    subprocess.run(['gh','release','upload',meta['release_tag'],str(out),str(sha),'--clobber'],check=True)
    subprocess.run(['gh','release','edit',meta['release_tag'],'--notes-file',str(ROOT/'reports/p8/RELEASE_NOTES.md')],check=True)
    print('::notice title=P8 final archive::'+json.dumps(dict(meta,sha256=digest,bytes=out.stat().st_size)))
if __name__=='__main__':main()
