#!/usr/bin/env python3
"""
Phase B: deterministic pair generation for CORPUS-REAL-R1
8 codes exactly from КОРПУС_v1.md, without LLM, seed-fixed, span positions deterministic from sha256(id)
"""
import json, hashlib, re, random
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/user/Smarty_Chumakov")
MANIFEST = ROOT / "reports/science/real/CORPUS_MANIFEST.json"
OUT_DIR = ROOT / "reports/science/real"
SEED = 42  # fixed before generation
random.seed(SEED)

# definitions verbatim from КОРПУС_v1.md table
DEFINITIONS = {
    "faithful": "без искажений (чистые пары, ≥ 40 %) | faithful",
    "contradiction": "значение заменено на другое | Contradiction",
    "partial": "отброшено условие, смысл искажён | Partial",
    "number_attribution": "число взято из факта другого объекта | Contradiction (подтип)",
    "unconfirmed": "добавлено утверждение, которого нет в документе | Unconfirmed",
    "missing": "обязательная деталь пропущена | Missing",
    "oversight": "важное значение потеряно при пересказе | Oversight",
    "excess": "добавлена лишняя деталь | Excess",
}

CODES = list(DEFINITIONS.keys()) # 8
# quotas: faithful 40%+, rest each >=5% of injected and >=40 pairs
# for 1500 total: faithful 600, each error 128-129
TOTAL = 1500
DEV_N = 300
TEST_N = 1200
FAITHFUL_N = 600
ERROR_N_TOTAL = TOTAL - FAITHFUL_N # 900
ERROR_EACH = ERROR_N_TOTAL // 7 # 128
REMAINDER = ERROR_N_TOTAL % 7 # 4 -> distribute to first 4 error codes
# build mode assignment list deterministic
error_codes = [c for c in CODES if c != "faithful"]
mode_list = ["faithful"]*FAITHFUL_N
for i,code in enumerate(error_codes):
    cnt = ERROR_EACH + (1 if i < REMAINDER else 0)
    mode_list.extend([code]*cnt)
assert len(mode_list)==TOTAL

# shuffle mode assignment deterministically by doc id hash? For reproducibility, we will assign modes in order of sorted doc ids after shuffling docs deterministically
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
entries = manifest["entries"]
# entries are 7710 docs, need 1500 distinct docs for pairs
# sort by id for determinism, then hash shuffle with seed
def hash_shuffle(lst, seed_str):
    # deterministic shuffle: sort by hash(seed+id)
    def h(x):
        return hashlib.sha256(f"{seed_str}:{x['id']}".encode()).hexdigest()
    return sorted(lst, key=h)

shuffled_docs = hash_shuffle(entries, str(SEED))
selected_docs = shuffled_docs[:TOTAL] # 1500 unique docs
# verify we have enough distinct docs (7710 >1500)
print(f"Selected {len(selected_docs)} docs from {len(entries)}")

# Now assign modes: we also want deterministic mode assignment order: sort selected_docs by id, then assign mode_list in order of hash shuffle?
# Approach: selected_docs already hash-shuffled, so their order is deterministic. Assign mode_list in that order sequentially (first 600 faithful etc). This satisfies quotas.
# But to avoid bias where early docs all faithful, we should intermix? Spec says "детерминизм: выбор позиции — от sha256(id пары)" for spans, but not for mode. For mode, we should ensure distribution is deterministic but not necessarily alphabetical. Our current approach assigns first 600 of hash-shuffled docs as faithful, which is random enough due to shuffle.
pairs = []
for idx, doc in enumerate(selected_docs):
    mode = mode_list[idx]  # deterministic assignment after hash shuffle
    doc_id = doc["id"]
    doc_text = None
    # need to get text from corpus_raw.jsonl.gz
    # We'll load raw later
    pairs.append((doc, mode))

# Load corpus_raw mapping id->text
raw_path = OUT_DIR / "corpus_raw.jsonl"
# if gz exists, use gz
import gzip
text_map = {}
if raw_path.exists():
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text_map[rec["id"]] = rec["text"]
else:
    gz_path = OUT_DIR / "corpus_raw.jsonl.gz"
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text_map[rec["id"]] = rec["text"]
print(f"text_map {len(text_map)}")

# helpers
def split_sentences(text):
    # simple split by .!? with regex preserving delimiters
    # Use regex for Russian sentence boundaries: [.!?]+ followed by space or end
    # Keep sentences as strings with offsets
    sents=[]
    # Find sentences via regex
    pattern = re.compile(r'[^.!?]+[.!?]+(?:\s|$)', re.UNICODE)
    pos=0
    for m in pattern.finditer(text):
        s=m.group().strip()
        if s:
            start=text.find(s, pos)
            end=start+len(s)
            sents.append((s, start, end))
            pos=end
    # if no sentences found, treat whole as one
    if not sents and text.strip():
        sents=[(text.strip(),0,len(text))]
    return sents

def extract_numbers(text):
    # numbers like 10, 152-ФЗ, dates, etc
    # regex for numbers
    return [m.group() for m in re.finditer(r'\d+(?:[-\s]*[А-Яа-яA-Za-z]+)?|\d+[\.,/]\d+', text)]

def deterministic_choice(options, key, seed=SEED):
    if not options:
        return None
    h=hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    idx=int(h,16) % len(options)
    return options[idx]

# generation per mode
generated=[]
for doc, mode in pairs:
    doc_id=doc["id"]
    doc_text=text_map.get(doc_id, "")
    if not doc_text:
        print(f"WARNING no text for {doc_id}")
        continue
    sents=split_sentences(doc_text)
    if not sents:
        sents=[(doc_text,0,len(doc_text))]
    # pick 1-2 sentences for faithful base deterministically
    # Need to pick faithful base answer: 1-2 consecutive sentences
    # Choose start index deterministic from sha
    h=hashlib.sha256(f"{SEED}:{doc_id}:base".encode()).hexdigest()
    start_idx=int(h,16) % max(1, len(sents))
    # decide length 1 or 2
    h2=hashlib.sha256(f"{SEED}:{doc_id}:len".encode()).hexdigest()
    n_sents=1 + (int(h2,16) % 2) # 1 or 2
    # ensure not out of bounds
    base_sents=sents[start_idx:start_idx+n_sents]
    if not base_sents:
        base_sents=sents[:1]
    faithful_answer=" ".join(s for s,_,_ in base_sents)
    # remove excessive whitespace
    faithful_answer=re.sub(r'\s+', ' ', faithful_answer).strip()
    # Generate answer per mode
    answer=""
    spans=[]
    reason=mode
    if mode=="faithful":
        answer=faithful_answer
        spans=[]
    elif mode=="contradiction":
        # find numbers in faithful answer, replace first deterministic number with altered
        nums=re.findall(r'\d+', faithful_answer)
        if nums:
            target=deterministic_choice(nums, f"{doc_id}:contradiction")
            # alter: increment or change
            try:
                new_num=str(int(target)+1) if target.isdigit() else target+"-modified"
            except:
                new_num="999"
            # replace first occurrence deterministically
            # find position of target in answer
            pos=faithful_answer.find(target)
            if pos!=-1:
                answer=faithful_answer[:pos]+new_num+faithful_answer[pos+len(target):]
                # span is the new number
                spans=[{"start":pos, "end":pos+len(new_num), "text":new_num, "reason":reason, "type":"contradiction"}]
            else:
                answer=faithful_answer+" Неверное число."
                pos=len(answer)-len("Неверное число.")
                spans=[{"start":pos,"end":len(answer),"text":"Неверное число.","reason":reason}]
        else:
            # if no numbers, inject generic contradiction
            answer=faithful_answer+" При этом срок составляет 10 лет, что противоречит документу."
            extra="10 лет, что противоречит документу"
            pos=answer.find(extra)
            spans=[{"start":pos,"end":pos+len(extra),"text":extra,"reason":reason}]
    elif mode=="partial":
        # drop condition after "если", "при условии" etc
        # find conditional clause
        m=re.search(r'если[^,.]+|при условии[^,.]+|в случае[^,.]+', faithful_answer, flags=re.IGNORECASE)
        if m:
            # remove clause
            answer=faithful_answer[:m.start()]+faithful_answer[m.end():]
            answer=re.sub(r'\s+',' ',answer).strip()
            # span is remaining part that lost condition? For partial, the span is the truncated answer where condition missing? But we need a span that indicates distorted meaning — we can mark the area near removal as span
            # Choose span as the sentence where condition was removed (approx)
            # Find nearest sentence boundary around removal
            # For simplicity, mark the whole answer as span where info missing? But spec says spans are where answer is false. For partial, the answer is incomplete — we can mark the place where condition was as span with text that is now misleading
            # We'll mark the next 20 chars after removal as span
            pos=m.start()
            end=min(len(answer), pos+30)
            span_text=answer[pos:end]
            spans=[{"start":pos,"end":end,"text":span_text,"reason":reason}]
        else:
            # fallback: truncate answer by removing last clause
            if len(faithful_answer)>30:
                cut=len(faithful_answer)//2
                answer=faithful_answer[:cut].strip()+"…"
                spans=[{"start":cut,"end":len(answer),"text":answer[cut:],"reason":reason}]
            else:
                answer=faithful_answer+" (условие опущено)"
                pos=answer.find("(условие опущено)")
                spans=[{"start":pos,"end":len(answer),"text":"(условие опущено)","reason":reason}]
    elif mode=="number_attribution":
        # need two numbers from document, swap attribution
        doc_nums=re.findall(r'\d+\s*[а-яА-Яa-zA-Z%]+|\d+[-–]\d+', doc_text)
        # pick two distinct numbers
        if len(doc_nums)>=2:
            # deterministic pick
            h=hashlib.sha256(f"{SEED}:{doc_id}:num_attr".encode()).hexdigest()
            idx1=int(h[16:32],16)%len(doc_nums)
            idx2=int(h[32:48],16)%len(doc_nums)
            if idx1==idx2:
                idx2=(idx2+1)%len(doc_nums)
            n1=doc_nums[idx1]
            n2=doc_nums[idx2]
            # replace n1 in faithful answer with n2 (if present) else append
            if n1 in faithful_answer:
                pos=faithful_answer.find(n1)
                answer=faithful_answer[:pos]+n2+faithful_answer[pos+len(n1):]
                spans=[{"start":pos,"end":pos+len(n2),"text":n2,"reason":reason}]
            else:
                # inject n2 as attribution error
                answer=faithful_answer+f" Срок {n2}."
                pos=answer.rfind(n2)
                spans=[{"start":pos,"end":pos+len(n2),"text":n2,"reason":reason}]
        else:
            # fallback generic
            answer=faithful_answer+" Число взято из другого факта."
            pos=answer.find("Число взято")
            spans=[{"start":pos,"end":len(answer),"text":answer[pos:],"reason":reason}]
    elif mode=="unconfirmed":
        # add statement not in document
        extra=" Документ также регулирует вопросы международной торговли, что не подтверждается источником."
        answer=faithful_answer+extra
        pos=answer.find(extra.strip())
        spans=[{"start":pos,"end":len(answer),"text":answer[pos:].strip(),"reason":reason}]
    elif mode=="missing":
        # omit mandatory detail: remove a sentence (if faithful had 2 sents, keep 1)
        if len(base_sents)>1:
            # keep only first
            answer=base_sents[0][0]
            # span is where detail missing? We can mark the whole remaining answer as missing detail? But spec says spans are where answer is false/incomplete — for missing, the fact that detail missing is not directly visible, but we can mark the answer as span where omission occurs (e.g., the point where detail should be)
            # Mark the end of answer as missing
            pos=len(answer)
            # add ellipsis marker
            answer=answer+" […]"
            spans=[{"start":pos,"end":len(answer),"text":"[…]", "reason":reason}]
        else:
            # remove part of sentence
            words=faithful_answer.split()
            if len(words)>5:
                cut=len(words)//2
                answer=" ".join(words[:cut])+" …"
                spans=[{"start":len(" ".join(words[:cut])),"end":len(answer),"text":"…","reason":reason}]
            else:
                answer=faithful_answer
                spans=[{"start":0,"end":len(answer),"text":answer,"reason":reason}]
    elif mode=="oversight":
        # important value lost during retelling: remove number
        nums=re.findall(r'\d+', faithful_answer)
        if nums:
            target=deterministic_choice(nums, f"{doc_id}:oversight")
            pos=faithful_answer.find(target)
            answer=faithful_answer[:pos]+faithful_answer[pos+len(target):]
            answer=re.sub(r'\s+',' ',answer).strip()
            # span is gap where value lost — mark surrounding 20 chars (ensure >0)
            end=min(len(answer), pos+20)
            start=max(0, pos-5)
            if end<=start:
                end=min(len(answer), start+5)
                if end<=start:
                    start=0; end=min(3,len(answer))
            span_text=answer[start:end]
            if not span_text.strip():
                span_text=answer[start:end] or answer[:3]
                end=start+len(span_text)
            spans=[{"start":start,"end":end,"text":span_text,"reason":reason}]
            if not span_text.strip():
                spans=[{"start":pos,"end":pos+3,"text":answer[pos:pos+3],"reason":reason}]
        else:
            # fallback: remove last word
            words=faithful_answer.split()
            answer=" ".join(words[:-1])+" …"
            spans=[{"start":len(answer)-1,"end":len(answer),"text":"…","reason":reason}]
    elif mode=="excess":
        extra=" Дополнительная деталь, не относящаяся к документу, добавлена."
        answer=faithful_answer+extra
        pos=answer.find(extra.strip())
        spans=[{"start":pos,"end":len(answer),"text":answer[pos:].strip(),"reason":reason}]
    else:
        answer=faithful_answer

    # Ensure spans coordinates are correct UTF-8 after normalization (we already normalized answer via re.sub, but need to verify)
    # Normalize answer: NFC? Keep as is, but ensure we use python string length (UTF-8 codepoints = chars, but spec says symbols UTF-8 — for python, len==codepoints, which for Russian equals symbols)
    # Verify span text equals answer[start:end]
    verified_spans=[]
    for sp in spans:
        s,e=sp["start"],sp["end"]
        # clamp
        s=max(0,min(s,len(answer)))
        e=max(s,min(e,len(answer)))
        if e<=s:
            e=min(len(answer), s+3)
            if e<=s:
                s=0; e=min(3,len(answer))
        txt=answer[s:e]
        # if mismatch, adjust to actual span_text if needed
        if txt != sp["text"]:
            # try to find sp["text"] in answer
            pos2=answer.find(sp["text"])
            if pos2!=-1:
                s=pos2
                e=pos2+len(sp["text"])
                txt=sp["text"]
            else:
                # keep as is but set text to actual slice
                sp["text"]=txt
                if not txt:
                    s=0; e=min(1,len(answer)); txt=answer[s:e]; sp["text"]=txt
        if not txt:
            s=0; e=min(1,len(answer)); txt=answer[s:e]
        verified_spans.append({"start":s,"end":e,"text":txt,"reason":sp["reason"],"type":mode})
    # For faithful, spans empty
    # Create pair record
    pair_id=f"real-{mode}-{doc_id}"
    # deterministic pair id from doc_id+mode
    # Use hash to make unique but stable
    pair_hash=hashlib.sha256(f"{doc_id}:{mode}:{SEED}".encode()).hexdigest()[:8]
    pair_id=f"corpus-real-{pair_hash}"
    generated.append({
        "id": pair_id,
        "doc_id": doc_id,
        "context": doc_text,
        "answer": answer,
        "labels": [[sp["start"], sp["end"], 1] for sp in verified_spans], # for compatibility with corpus_a format? but we need spans
        "spans": verified_spans,
        "meta": {
            "group": doc_id,
            "mode": mode,
            "taxonomy": mode,
            "reason": reason,
            "definition": DEFINITIONS[mode],
            "clean": mode=="faithful",
            "seed": SEED,
            "doc_sha256": doc["sha256"],
            "pair_sha256": hashlib.sha256(f"{doc_id}:{answer}".encode()).hexdigest()
        }
    })

print(f"Generated {len(generated)} pairs")
# count per mode
from collections import Counter
cnt=Counter(g["meta"]["mode"] for g in generated)
print(cnt)
# assert quotas
assert cnt["faithful"]>=0.4*TOTAL, f"faithful {cnt['faithful']} <40%"
for code in error_codes:
    assert cnt[code]>=40, f"{code} {cnt[code]} <40"
    assert cnt[code]>=0.05*ERROR_N_TOTAL, f"{code} below 5% of injected"
print("Quotas OK")
# verify spans text == answer slice for 100%
for g in generated:
    ans=g["answer"]
    for sp in g["spans"]:
        assert ans[sp["start"]:sp["end"]]==sp["text"], f"mismatch {g['id']} {sp}"
print("All spans verified 100%")

# Now split by document: need real_dev >=300, real_test >=1200, zero overlap by doc/parent id
# We have 1500 pairs each with unique doc_id (since we used distinct docs), so split by doc is easy
# Determine split deterministic: hash shuffle of generated list by doc_id
def hash_key(g):
    return hashlib.sha256(f"{SEED}:split:{g['doc_id']}".encode()).hexdigest()
sorted_pairs=sorted(generated, key=hash_key)
dev=sorted_pairs[:DEV_N]
test=sorted_pairs[DEV_N:DEV_N+TEST_N]
assert len(dev)==DEV_N and len(test)==TEST_N
# check no overlap
dev_ids=set(g["doc_id"] for g in dev)
test_ids=set(g["doc_id"] for g in test)
assert not dev_ids & test_ids, "overlap"
print(f"Split dev {len(dev)} test {len(test)} no overlap OK")

# Save
OUT_DIR.mkdir(parents=True, exist_ok=True)
# pairs.jsonl format: id, context, answer, labels, meta
def save_pairs(pairs_list, path):
    with open(path, "w", encoding="utf-8") as f:
        for g in pairs_list:
            rec={"id": g["id"], "context": g["context"], "answer": g["answer"], "labels": g["labels"], "meta": g["meta"]}
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
def save_spans(pairs_list, path):
    # spans.jsonl: maybe for compatibility? We'll create answers.jsonl style
    with open(path, "w", encoding="utf-8") as f:
        for g in pairs_list:
            rec={"id": g["id"], "spans": g["spans"]}
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")

save_pairs(dev, OUT_DIR / "real_dev_pairs.jsonl")
save_pairs(test, OUT_DIR / "real_test_pairs.jsonl")
save_pairs(generated, OUT_DIR / "real_all_pairs.jsonl")
# also splits dir like corpus_b expectation?
# Create splits directory
(OUT_DIR / "splits").mkdir(exist_ok=True)
save_pairs(dev, OUT_DIR / "splits" / "dev.jsonl")
save_pairs(test, OUT_DIR / "splits" / "test.jsonl")
# spans
save_spans(dev, OUT_DIR / "real_dev_spans.jsonl")
save_spans(test, OUT_DIR / "real_test_spans.jsonl")

# Compute sha256 for all files
import hashlib as hl
files_to_hash = [OUT_DIR / "real_dev_pairs.jsonl", OUT_DIR / "real_test_pairs.jsonl", OUT_DIR / "real_all_pairs.jsonl", OUT_DIR / "real_dev_spans.jsonl", OUT_DIR / "real_test_spans.jsonl", OUT_DIR / "splits/dev.jsonl", OUT_DIR / "splits/test.jsonl"]
sha_map={}
for p in files_to_hash:
    if p.exists():
        sha_map[str(p.relative_to(ROOT))] = hl.sha256(p.read_bytes()).hexdigest()

# span statistics
all_spans=[]
for g in generated:
    for sp in g["spans"]:
        all_spans.append(sp)
# stats
if all_spans:
    lens=[len(sp["text"]) for sp in all_spans]
    lens_sorted=sorted(lens)
    n=len(lens)
    min_l=min(lens)
    max_l=max(lens)
    median=lens_sorted[n//2]
    # share of answer: sum span lengths / sum answer lengths for those with spans
    total_span_len=sum(lens)
    total_answer_len=sum(len(g["answer"]) for g in generated if g["spans"])
    share=total_span_len/total_answer_len if total_answer_len else 0
else:
    n=0; min_l=max_l=median=share=0

# CORPUS_FROZEN.json
frozen={
    "kind": "corpus_frozen",
    "created_at": datetime.utcnow().isoformat()+"Z",
    "created_by": "CORPUS-REAL-R1",
    "source": "LawCodex/ru pin 5806f92",
    "manifest_sha256": json.loads((OUT_DIR / "CORPUS_MANIFEST.json").read_text(encoding="utf-8")).get("manifest_sha256"),
    "seed": SEED,
    "date": datetime.utcnow().date().isoformat(),
    "author": "deterministic generation (no LLM)",
    "determinism": "seed фиксирован до генерации, выбор позиции детерминирован от sha256(id пары)",
    "split": {
        "by": "document (parent id)",
        "rule": "deterministic hash shuffle by sha256(seed:split:doc_id), first 300 dev, next 1200 test, zero overlap",
        "real_dev": DEV_N,
        "real_test": TEST_N,
        "total": TOTAL,
        "dev_ids": sorted(dev_ids)[:5], # sample
        "test_ids": sorted(test_ids)[:5],
        "overlap": 0
    },
    "quotas": {
        "faithful": cnt["faithful"],
        "per_error_min": min(cnt[c] for c in error_codes),
        "per_error": {k: cnt[k] for k in CODES},
        "faithful_share": cnt["faithful"]/TOTAL,
        "clean_40pct": cnt["faithful"]>=0.4*TOTAL
    },
    "definitions": DEFINITIONS, # verbatim fragments from КОРПУС_v1.md
    "files": {k: {"sha256": v, "path": k} for k,v in sha_map.items()},
    "span_stats": {
        "n_spans": n,
        "min_len": min_l,
        "median_len": median,
        "max_len": max_l,
        "share_of_answer": share,
        "avg_spans_per_injected": n/len([g for g in generated if g["spans"]]) if n else 0
    },
    "asserts": {
        "spans_text_equals_slice": "100% verified",
        "clean_share_40pct": True,
        "per_type_min_40": True,
        "split_no_overlap": True
    }
}
(OUT_DIR / "CORPUS_FROZEN.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(f"Wrote CORPUS_FROZEN n_spans {n} min {min_l} median {median} max {max_l} share {share:.3f}")
print(f"Files sha: {sha_map}")

# spotcheck 5% deterministic from manifest sha
manifest_sha=json.loads((OUT_DIR / "CORPUS_MANIFEST.json").read_text(encoding="utf-8"))["manifest_sha256"]
# pick 5% of doc ids = 0.05*7710≈385 but we need 5% of pairs? Spec says 5% пар из-под просмотра куратора (коммит списка doc_id)
# We'll pick 5% of generated pairs' doc_ids = 75 (5% of 1500)
spot_n=int(0.05*TOTAL)
# deterministic selection from manifest sha
# sort doc_ids of generated
all_doc_ids=sorted([g["doc_id"] for g in generated])
def h2(did):
    return hashlib.sha256(f"{manifest_sha}:{did}".encode()).hexdigest()
sorted_by_hash=sorted(all_doc_ids, key=h2)
spot_ids=sorted_by_hash[:spot_n]
(OUT_DIR / "SPOTCHECK_DOCIDS.txt").write_text("\n".join(spot_ids)+"\n", encoding="utf-8")
print(f"Spotcheck {spot_n} doc_ids: {spot_ids[:3]}...")

