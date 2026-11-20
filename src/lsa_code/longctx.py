import json, time, urllib.request, random
B="http://127.0.0.1:30000"; M="/mnt/sda/models/Qwen3.8-27B"
random.seed(0)
NEEDLE = "The internal project codename is BLUEHERON-7742."
filler_paras = [
 "The quarterly logistics review covers warehouse throughput, inbound freight scheduling, and inventory turnover "
 "across the regional distribution centres. Each centre reports its dock utilisation and cold-chain compliance.",
 "Supplier audits are conducted twice per year. Findings are logged in the compliance register and tracked to "
 "closure with an owner and a due date. Repeat findings escalate to the regional operations committee.",
 "The maintenance backlog is prioritised by safety impact first, then by production criticality, then by cost. "
 "Work orders older than ninety days are reviewed individually by the plant engineering lead.",
 "Demand planning uses a rolling thirteen-week horizon. Forecast error is tracked weekly and the bias is reviewed "
 "monthly so that safety stock targets can be adjusted before the seasonal peak.",
]
def build(approx_tokens):
    words_per_para = len(filler_paras[0].split())
    n = max(4, approx_tokens // words_per_para)
    parts = []
    for i in range(n):
        parts.append(f"Section {i+1}. " + filler_paras[i % len(filler_paras)])
    body = "\n\n".join(parts)
    cut = len(body)//2
    cut = body.rfind("\n\n", 0, cut)
    body = body[:cut] + "\n\n" + NEEDLE + "\n\n" + body[cut:]
    return ("You are given an internal operations dossier.\n\n" + body +
            "\n\nQuestion: What is the internal project codename stated in the dossier? Answer with the codename only."), n

for target in (8000, 30000):
    prompt, npara = build(target)
    body=json.dumps({"model":M,"messages":[{"role":"user","content":prompt}],"max_tokens":24,
                     "temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(B+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    resp=urllib.request.urlopen(req,timeout=1800)
    t0=time.time(); first=None; chunks=[]
    raw=b""
    for line in resp:
        raw+=line
    dt=time.time()-t0
    r=json.loads(raw)
    u=r["usage"]; c=r["choices"][0]["message"]["content"]
    print(f"[~{target} tok target] prompt_tokens={u['prompt_tokens']} completion={u['completion_tokens']} "
          f"total_wall={dt:.2f}s out={c.strip()[:60]!r}")
    print(f"    needle found: {'BLUEHERON-7742' in c}")