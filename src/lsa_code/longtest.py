import json, time, urllib.request, sys, os
B="http://127.0.0.1:30000"; M="/mnt/sda/models/Qwen3.8-27B"
NEEDLE="The internal project codename is BLUEHERON-7742."
F=("The quarterly logistics review covers warehouse throughput, inbound freight scheduling, and inventory turnover "
   "across the regional distribution centres. Each centre reports its dock utilisation and cold-chain compliance.")
def build(target):
    n=max(4,target//40)
    body="\n\n".join(f"Section {i+1}. "+F for i in range(n))
    cut=body.rfind("\n\n",0,len(body)//2); body=body[:cut]+"\n\n"+NEEDLE+"\n\n"+body[cut:]
    return ("You are given an internal operations dossier.\n\n"+body+
            "\n\nQuestion: What is the internal project codename stated in the dossier? Answer with the codename only.")
out=[]
def log(s):
    print(s, flush=True)
    open("/mnt/sda/qyj1044/lsa_code/longtest.out","a").write(s+"\n")
open("/mnt/sda/qyj1044/lsa_code/longtest.out","w").close()
for target in (8000, 16000, 32000, 64000, 128000):
    p=build(target)
    payload=json.dumps({"model":M,"messages":[{"role":"user","content":p}],"max_tokens":24,
                        "temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    try:
        t0=time.time()
        r=json.loads(urllib.request.urlopen(urllib.request.Request(B+"/v1/chat/completions",data=payload,
            headers={"Content-Type":"application/json"}),timeout=2400).read())
        dt=time.time()-t0
        u=r["usage"]; c=r["choices"][0]["message"]["content"]
        log(f"[target={target:6d} body={len(payload):7d}B] prompt_tok={u['prompt_tokens']:6d} wall={dt:6.2f}s "
            f"prefill_rate={u['prompt_tokens']/dt:7.1f} tok/s needle={'BLUEHERON-7742' in c} out={c.strip()[:32]!r}")
    except Exception as e:
        log(f"[target={target:6d} body={len(payload):7d}B] FAILED {type(e).__name__}: {str(e)[:120]}")
log("DONE")