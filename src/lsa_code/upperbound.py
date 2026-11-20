import json, time, urllib.request, sys
B="http://127.0.0.1:30000"; M="/mnt/sda/qyj1044/dspine_train/model_view"
OUT="/mnt/sda/qyj1044/lsa_code/upperbound.out"
open(OUT,"w").close()
def log(s):
    print(s, flush=True)
    with open(OUT,"a") as f: f.write(s+"\n")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
P = json.load(open("/mnt/sda/qyj1044/dspine_train/prompts.json"))

def ask(prompt, n=512):
    body=json.dumps({"model":M,"messages":[{"role":"user","content":prompt}],
                     "max_tokens":n,"temperature":0,
                     "chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=json.loads(urllib.request.urlopen(urllib.request.Request(B+"/v1/chat/completions",data=body,
        headers={"Content-Type":"application/json"}),timeout=1800).read())
    return r["choices"][0]["message"]["content"], r["usage"]

def match_len(seq, p, w):
    """最右匹配：用 seq[p-w:p] 在 seq[:p-w] 中找，返回后续连续吻合的 token 数"""
    if p < w + 1: return 0
    q_tokens = seq[p-w:p]
    best = 0
    limit = p - w
    start = limit - 1
    while start >= 0:
        if seq[start:start+w] == q_tokens:
            k = 0
            while (start+w+k < p) and (p+k < len(seq)) and seq[start+w+k] == seq[p+k]:
                k += 1
            if k > best: best = k
            if best >= 16: break
        start -= 1
    return best

summary={}
for name, prompt in P.items():
    txt, usage = ask(prompt)
    ptok = tok(prompt)["input_ids"]; ctok = tok(txt)["input_ids"]
    seq = ptok + ctok
    base = len(ptok)
    stats={}
    for w in (2,3,4):
        lens=[match_len(seq, p, w) for p in range(base, len(seq))]
        if not lens: continue
        n=len(lens)
        stats[w]=dict(n=n,
                      mean=sum(lens)/n,
                      ge1=sum(1 for x in lens if x>=1)/n,
                      ge4=sum(1 for x in lens if x>=4)/n,
                      ge8=sum(1 for x in lens if x>=8)/n,
                      mean_cap16=sum(min(x,16) for x in lens)/n)
    summary[name]=dict(completion_tokens=len(ctok), prompt_tokens=len(ptok), by_w=stats)
    log(f"[{name}] prompt_tok={len(ptok)} gen_tok={len(ctok)}")
    for w,s in stats.items():
        log(f"    w={w}: mean_match={s['mean']:.2f} P(>=1)={s['ge1']:.2f} "
            f"P(>=4)={s['ge4']:.2f} P(>=8)={s['ge8']:.2f} mean_cap16={s['mean_cap16']:.2f}")
json.dump(summary, open("/mnt/sda/qyj1044/lsa_code/upperbound.json","w"), indent=2)
log("DONE")