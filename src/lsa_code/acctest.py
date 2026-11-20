import json, time, sys, os
def main():
    MODE = sys.argv[1]; MAXLEN = int(sys.argv[2])
    MAP="/mnt/sda/qyj1044/lsa_code/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt"
    OUT=f"/mnt/sda/qyj1044/lsa_code/acc_{MODE}.out"; open(OUT,"w").close()
    def log(s):
        print(s,flush=True)
        with open(OUT,"a") as f: f.write(s+"\n")
    import sglang as sgl
    kw=dict(model_path="/mnt/sda/models/Qwen3.8-27B",tp_size=1,attention_backend="triton",
            linear_attn_backend="triton",page_size=1,disable_prefill_cuda_graph=True,
            mem_fraction_static=0.80,context_length=MAXLEN,max_running_requests=4,trust_remote_code=True)
    if MODE=="lsa": kw["json_model_override_args"]=json.dumps({"lsa":{"projections":MAP,"query_heads":"all"}})
    eng=sgl.Engine(**kw); log(f"[{MODE}] engine up")
    NEEDLE="The internal project codename is BLUEHERON-7742."
    F=("The quarterly logistics review covers warehouse throughput, inbound freight scheduling, and inventory turnover "
       "across the regional distribution centres. Each centre reports its dock utilisation and cold-chain compliance.")
    def build(t):
        n=max(4,t//40); body="\n\n".join(f"Section {i+1}. "+F for i in range(n))
        c=body.rfind("\n\n",0,len(body)//2); body=body[:c]+"\n\n"+NEEDLE+"\n\n"+body[c:]
        return ("You are given an internal operations dossier.\n\n"+body+
                "\n\nQuestion: What is the internal project codename stated in the dossier? "
                "Answer with the codename only, no explanation.")
    SP={"max_new_tokens":400,"temperature":0,"skip_special_tokens":True}
    for t in (8000,64000,128000):
        p=build(t)
        try:
            t0=time.time(); res=eng.generate(prompt=p,sampling_params=SP); dt=time.time()-t0
            if isinstance(res,dict):
                t=res.get("text"); txt=t[0] if isinstance(t,list) else (t or "")
                m=res.get("meta_info"); meta=m[0] if isinstance(m,list) else (m or {})
            else:
                txt=res[0]["text"]; meta=res[0].get("meta_info",{})
            pt=meta.get("prompt_tokens"); ct=meta.get("completion_tokens")
            ok="BLUEHERON-7742" in str(txt)
            tail=str(txt).strip()[-40:]
            log(f"[{MODE} ctx={t:6d}] ptok={pt} gen={ct} wall={dt:6.2f}s needle={ok} tail={tail!r}")
        except Exception as e:
            log(f"[{MODE} ctx={t}] FAILED {type(e).__name__}: {str(e)[:140]}")
    log(f"[{MODE}] DONE")
    try: eng.shutdown()
    except Exception: pass
    os._exit(0)
if __name__=="__main__": main()