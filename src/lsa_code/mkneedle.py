import json,sys
NEEDLE="The internal project codename is BLUEHERON-7742."
f=[("The quarterly logistics review covers warehouse throughput, inbound freight scheduling, and inventory turnover "
 "across the regional distribution centres. Each centre reports its dock utilisation and cold-chain compliance.")]
def build(target):
    n=max(4,target//40); body="\n\n".join(f"Section {i+1}. "+f[0] for i in range(n))
    cut=body.rfind("\n\n",0,len(body)//2); body=body[:cut]+"\n\n"+NEEDLE+"\n\n"+body[cut:]
    return ("You are given an internal operations dossier.\n\n"+body+
            "\n\nQuestion: What is the internal project codename stated in the dossier? Answer with the codename only.")
for name,t in (("n8k",8000),("n16k",16000)):
    p=build(t)
    json.dump({"model":"/mnt/sda/models/Qwen3.8-27B","messages":[{"role":"user","content":p}],
               "max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}},
              open("/mnt/sda/qyj1044/lsa_code/"+name+".json","w"))
    print(name,"bytes",len(json.dumps({"x":p})))