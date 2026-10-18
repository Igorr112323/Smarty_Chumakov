#!/usr/bin/env python3
"""Fail-closed cached-HF P2 experiment. Dev never participates in selection."""
import sys, os, time, json, hashlib, platform
from pathlib import Path
from dataclasses import replace, asdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from spanverify.core import tokenize_with_offsets
from spanverify.features import is_scored_token
from spanverify.features import FEATURE_NAMES, load_feature_cache, set_feature_cache, feature_cache_key
from spanverify.dataset import read_pairs
from spanverify.engine import WeightsBundle, smooth_token_risks, _binary_metrics, _span_f1
from spanverify.token_head import extract_token_features, train_token_gbdt
from replay_p1_dev import make_verifier, collect_public_rows, compare_records, metrics, read_json, read_jsonl, write_json, write_jsonl

BASE = Path('reports/experiments/p1_public_replay/run39144872719')
def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def fit_documents(states, docs):
    selected = [s for s in states if s['doc'] in docs]
    X = [x for s in selected for x, scored in zip(s['X'], s['scored']) if scored]
    y = [y for s in selected for y, scored in zip(s['gold'], s['scored']) if scored]
    return train_token_gbdt(X, y), {'pairs':len(selected), 'tokens':len(y), 'positive':sum(y), 'negative':len(y)-sum(y)}

def selection(verifier, states, tau):
    verifier.bundle = replace(verifier.bundle, token_threshold=float(tau))
    labels, flags, predicted, truth = [], [], [], []
    for s in states:
        tokens = tokenize_with_offsets(s['answer'])
        _, spans, kept = verifier._localize(s['answer'], s['context'], s['context'], tokens, s['smoothed'], [r for r,m in zip(s['smoothed'],s['scored']) if m] or [0.0])
        labels.extend(y for y,m in zip(s['gold'],s['scored']) if m)
        flags.extend((i in kept if kept is not None else r >= tau) for i,(r,m) in enumerate(zip(s['smoothed'],s['scored'])) if m)
        predicted.append([(p.start,p.end) for p in spans]); truth.append(s['truth'])
    tm = _binary_metrics(labels, flags)
    tp,fp,fn,tn = (tm[k] for k in ('tp','fp','fn','tn'))
    tm.update(precision=tp/(tp+fp) if tp+fp else 0.0, recall=tp/(tp+fn) if tp+fn else 0.0, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0, fpr=fp/(fp+tn) if fp+tn else 0.0)
    sm = _span_f1(predicted, truth, iou_threshold=0.5)
    return {'tau':float(tau), 'tokens':tm, 'spans':sm, 'objective':min(tm['f1'],sm['f1'])}

def run(cache_dir, out):
    started=time.perf_counter(); out.mkdir(parents=True,exist_ok=True)
    runtime=read_json(BASE/'runtime_config.json')
    assert runtime['mode']=='hf' and runtime['chunking'] is True and runtime['hf_max_tokens']==512
    assert runtime['model_name']=='ai-forever/rugpt3small_based_on_gpt2'
    baseline=WeightsBundle.load(BASE/'p1_candidate_bundle.json'); original=baseline.to_dict()
    train=list(read_pairs(Path('data/corpus_a3/splits/train.jsonl'))); dev=list(read_pairs(Path('data/corpus_a3/splits/dev.jsonl')))
    assert len(train)==842 and len(dev)==181
    shards=sorted(cache_dir.rglob('features_*.jsonl'))
    assert len(shards)==8 and len({hashlib.sha256(p.read_bytes()).hexdigest() for p in shards})==8, 'Need eight distinct shards'
    cache=load_feature_cache([str(p) for p in shards],counting=True,strict=True); set_feature_cache(cache)
    missing=[]
    for p in train+dev:
        key=feature_cache_key(p['answer'],p['context'],'hf',runtime['model_name'],chunking=True,max_length=512)
        if key not in cache: missing.append(p['id'])
    write_json(out/'cache_coverage.json',{'keys':len(cache),'pairs':1023,'missing':missing})
    if missing: raise ValueError('Missing compatible cache; fresh inference forbidden')
    v=make_verifier(baseline,runtime,cache)
    states=[]
    for p in train:
        doc=p.get('meta',{}).get('doc_id')
        if not doc: raise ValueError('Missing real doc ID')
        tokens=tokenize_with_offsets(p['answer']); f=v._features(p['answer'],p['context'],tokens)
        assert all(len(getattr(f,n))==len(tokens) for n in FEATURE_NAMES)
        truth=[(int(a),int(b)) for a,b,y in p['labels'] if int(y)==1]
        states.append({'id':p['id'],'doc':doc,'answer':p['answer'],'context':p['context'],'truth':truth,
            'offsets':[(t.start,t.end) for t in tokens], 'scored':[is_scored_token(t.text) for t in tokens],
            'gold':[int(any(t.start<b and t.end>a for a,b in truth)) for t in tokens],
            'X':[extract_token_features(i,len(tokens),t.text,{n:getattr(f,n)[i] for n in FEATURE_NAMES}) for i,t in enumerate(tokens)]})
    docs=sorted({s['doc'] for s in states}); receipts=[]
    for fold,(fi,hi) in enumerate(KFold(5,shuffle=True,random_state=42).split(docs)):
        fit={docs[i] for i in fi}; held={docs[i] for i in hi}; assert not fit & held
        model,counts=fit_documents(states,fit)
        receipts.append({'fold':fold,'fit_documents':sorted(fit),'held_documents':sorted(held),'fit_sha256':digest(sorted(fit)),'held_sha256':digest(sorted(held)),'shared_documents':0,'model_sha256':digest(model.to_dict()),**counts})
        for s in states:
            if s['doc'] not in held: continue
            assert 'fold' not in s
            raw=model.predict_many(s['X'])
            s['smoothed']=smooth_token_risks(raw,s['scored']); s['fold']=fold
        write_json(out/'fold_receipts.json',receipts)
        print('Completed fold',fold,counts,flush=True)
    assert all('fold' in s for s in states)
    write_jsonl(out/'oof_states.jsonl',[{k:v for k,v in s.items() if k!='X'} for s in states])
    risks=[r for s in states for r,m in zip(s['smoothed'],s['scored']) if m]
    candidates=sorted(set([0.0,float(np.nextafter(max(risks),np.inf)),*map(float,np.quantile(risks,np.linspace(0,1,500)))]))
    p2runtime={**runtime,'span_expansion':'mask'}
    selector=make_verifier(baseline,p2runtime,cache)
    points=[]
    for i,tau in enumerate(candidates):
        points.append(selection(selector,states,tau))
        if i%50==0: print('Tau candidate',i,'/',len(candidates),flush=True)
    feasible=[p for p in points if p['tokens']['fpr']<=0.30]
    chosen=max(feasible,key=lambda p:(p['objective'],-p['tokens']['fpr'],-p['tau']))
    write_json(out/'tau_selection.json',{'source':'train-only document OOF; selection is not independent acceptance','criterion':'max min(token F1, strict span F1), token FPR <= .30; tie lower FPR','selected':chosen,'candidates':points})
    model,counts=fit_documents(states,set(docs)); model.save(out/'p2_token_head.json')
    candidate=replace(baseline,token_head=model.to_dict(),token_threshold=chosen['tau'])
    assert baseline.to_dict()==original
    candidate.save(out/'combined_bundle.json'); write_json(out/'runtime_config.json',p2runtime)
    pv=make_verifier(candidate,p2runtime,cache); reloadv=make_verifier(WeightsBundle.load(out/'combined_bundle.json'),p2runtime,cache)
    before=collect_public_rows(pv,dev); after=collect_public_rows(reloadv,dev)
    reference=read_jsonl(BASE/'public_replay.jsonl')
    parity={'reference':compare_records(before,reference),'reload':compare_records(after,before)}
    write_jsonl(out/'dev_answers.jsonl',before); write_json(out/'p1_invariance.json',parity)
    answer=metrics(before); assert [answer[k] for k in ('tp','fp','fn','tn')]==[49,14,47,71]
    assert parity['reference']['matched'] and parity['reload']['matched'], 'P1 drift blocks candidate'
    evidence=[]
    for p in dev:
        result=pv.verify(p['answer'],p['context'],with_tokens=True)
        rr=reloadv.verify(p['answer'],p['context'],with_tokens=True)
        assert result.tokens==rr.tokens and result.spans==rr.spans
        for span in result.spans: assert span.text==p['answer'][span.start:span.end]
        evidence.append({'id':p['id'],'gold':p['labels'],'tokens':result.tokens,'spans':[asdict(s) for s in result.spans]})
    write_jsonl(out/'dev_evidence.jsonl',evidence)
    allmetrics={'baseline':v.evaluate(dev),'p2':pv.evaluate(dev)}
    write_json(out/'all_family_metrics.json',allmetrics)
    assert cache.misses==0
    write_json(out/'manifest.json',{'run_id':os.environ.get('GITHUB_RUN_ID'),'code_sha':os.environ.get('GITHUB_SHA'),'cache_source_run':39060670054,'baseline':str(BASE),'baseline_sha256':digest(original),'runtime':p2runtime,'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'cache_hits':cache.hits,'cache_misses':cache.misses,'train_pairs':842,'dev_pairs':181,'fit':counts,'elapsed_seconds':time.perf_counter()-started,'sizes':{p.name:p.stat().st_size for p in out.iterdir() if p.is_file()},'fresh_inference':False,'official_test':False})
    print(json.dumps({'tau':chosen,'metrics':allmetrics,'parity':parity},ensure_ascii=False),flush=True)

if __name__=='__main__':
    import argparse, traceback
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    try: run(args.cache,args.output)
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
