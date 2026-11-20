import json
NEEDLE="The internal project codename is BLUEHERON-7742."
f=[("The quarterly logistics review covers warehouse throughput, inbound freight scheduling, and inventory turnover "
 "across the regional distribution centres. Each centre reports its dock utilisation and cold-chain compliance.")]
n=16000//40; body="\n\n".join(f"Section {i+1}. "+f[0] for i in range(n))
cut=body.rfind("\n\n",0,len(body)//2); body=body[:cut]+"\n\n"+NEEDLE+"\n\n"+body[cut:]
p=("You are given an internal operations dossier.\n\n"+body+
   "\n\nWrite a detailed risk assessment of the operations described in the dossier above. "
   "Cover supplier concentration, dock capacity, maintenance backlog and forecast error. "
   "Use continuous prose without lists. Aim for 600 words.")
json.dump({"model":"/mnt/sda/models/Qwen3.8-27B","messages":[{"role":"user","content":p}],
           "max_tokens":512,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}},
          open("/mnt/sda/qyj1044/lsa_code/gen16k.json","w"))
print("bytes", len(json.dumps({"x":p})))