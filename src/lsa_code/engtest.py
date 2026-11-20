import json, time, sys, os
def main():
    MODE = sys.argv[1]; MAXLEN = int(sys.argv[2])
    MAP = "/mnt/sda/qyj1044/lsa_code/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt"
    OUT = f"/mnt/sda/qyj1044/lsa_code/eng_{MODE}.out"
    open(OUT,"w").close()
    def log(s):
        print(s, flush=True)
        with open(OUT,"a") as f: f.write(s+"\n")
    import sglang as sgl
    kw = dict(model_path="/mnt/sda/models/Qwen3.8-27B", tp_size=1,
              attention_backend="triton", linear_attn_backend="triton",
              page_size=1, disable_prefill_cuda_graph=True,
              mem_fraction_static=0.80, context_length=MAXLEN,
              max_running_requests=8, trust_remote_code=True)
    if MODE == "lsa":
        kw["json_model_override_args"] = json.dumps({"lsa": {"projections": MAP, "query_heads": "all"}})
    t0=time.time(); eng = sgl.Engine(**kw)
    log(f"[{MODE}] engine up in {time.time()-t0:.1f}s (context={MAXLEN})")
    NEEDLE="The internal project codename is BLUEHERON-7742."
    F=("The quarterly logistics review covers warehouse throughput, inbound freight scheduling, and inventory turnover "
       "across the regional distribution centres. Each centre reports its dock utilisation and cold-chain compliance.")
    def build(target):
        n=max(4,target//40)
        body="\n\n".join(f"Section {i+1}. "+F for i in range(n))
        cut=body.rfind("\n\n",0,len(body)//2); body=body[:cut]+"\n\n"+NEEDLE+"\n\n"+body[cut:]
        return ("You are given an internal operations dossier.\n\n"+body+
                "\n\nQuestion: What is the internal project codename stated in the dossier? Answer with the codename only.")
    SP={"max_new_tokens":24,"temperature":0,"skip_special_tokens":True}
    for target in (8000,16000,32000,64000,128000):
        if target > MAXLEN-3000:
            log(f"[{MODE} target={target:6d}] skipped (context {MAXLEN})"); continue
        p=build(target); t0=time.time()
        try:
            res=eng.generate(prompt=p, sampling_params=SP); dt=time.time()-t0
            if isinstance(res, dict):
                txt=res["text"][0] if isinstance(res.get("text"), list) else res.get("text","")
                meta=res["meta_info"][0] if isinstance(res.get("meta_info"), list) else res.get("meta_info",{})
            else:
                txt=res[0]["text"]; meta=res[0].get("meta_info",{})
            pt=meta.get("prompt_tokens"); ct=meta.get("completion_tokens")
            log(f"[{MODE} target={target:6d}] ptok={pt} wall={dt:6.2f}s "
                f"prefill_rate={(pt/dt if pt else 0):8.1f} tok/s needle={'BLUEHERON-7742' in str(txt)} out={str(txt).strip()[:30]!r}")
        except Exception as e:
            log(f"[{MODE} target={target}] FAILED {type(e).__name__}: {str(e)[:160]}")
    log(f"[{MODE}] DONE")
    try: eng.shutdown()
    except Exception: pass
    os._exit(0)
if __name__ == "__main__":
    main()