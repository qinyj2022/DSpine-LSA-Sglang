import json
NEEDLE="The internal project codename is BLUEHERON-7742."
filler=[
 "The quarterly logistics review covers warehouse throughput, inbound freight scheduling, and inventory turnover "
 "across the regional distribution centres. Each centre reports its dock utilisation and cold-chain compliance.",
 "Supplier audits are conducted twice per year. Findings are logged in the compliance register and tracked to "
 "closure with an owner and a due date. Repeat findings escalate to the regional operations committee.",
 "The maintenance backlog is prioritised by safety impact first, then by production criticality, then by cost. "
 "Work orders older than ninety days are reviewed individually by the plant engineering lead.",
 "Demand planning uses a rolling thirteen-week horizon. Forecast error is tracked weekly and the bias is reviewed "
 "monthly so that safety stock targets can be adjusted before the seasonal peak.",
]
def build(target):
    n=max(4, target//45); parts=[f"Section {i+1}. "+filler[i%4] for i in range(n)]
    body="\n\n".join(parts); cut=body.rfind("\n\n",0,len(body)//2)
    body=body[:cut]+"\n\n"+NEEDLE+"\n\n"+body[cut:]
    return ("You are given an internal operations dossier.\n\n"+body+
            "\n\nQuestion: What is the internal project codename stated in the dossier? Answer with the codename only.")
for name,t in (("p8k",8000),("p30k",30000)):
    p=build(t)
    json.dump({"model":"/mnt/sda/models/Qwen3.8-27B",
               "messages":[{"role":"user","content":p}],"max_tokens":24,"temperature":0,
               "chat_template_kwargs":{"enable_thinking":False}},
              open("/mnt/sda/qyj1044/lsa_code/"+name+".json","w"))
    print(name,"chars",len(p))