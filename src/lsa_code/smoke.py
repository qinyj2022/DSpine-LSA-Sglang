import json, time, urllib.request
B="http://127.0.0.1:30000"; M="/mnt/sda/models/Qwen3.8-27B"
def ask(prompt, n=128):
    body=json.dumps({"model":M,"messages":[{"role":"user","content":prompt}],"max_tokens":n,
                     "temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    t0=time.time()
    r=json.loads(urllib.request.urlopen(urllib.request.Request(B+"/v1/chat/completions",data=body,
        headers={"Content-Type":"application/json"}),timeout=900).read())
    dt=time.time()-t0
    return r["choices"][0]["message"]["content"], r["usage"], dt
print("=== 冒烟 1: 短问答 ===")
c,u,dt=ask("The capital of France is", 16)
print("  out:", repr(c[:120]))
print(f"  usage={u} {dt:.2f}s")
print("=== 冒烟 2: 简单推理 ===")
c,u,dt=ask("Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each. How many total? Answer with the number.", 96)
print("  out:", repr(c[:200]))
print(f"  tokens={u['completion_tokens']} {dt:.2f}s -> {u['completion_tokens']/dt:.1f} tok/s")
print("=== 冒烟 3: 长生成吞吐 ===")
c,u,dt=ask("Write a detailed technical explanation of how a transformer decoder works.", 256)
print(f"  tokens={u['completion_tokens']} {dt:.2f}s -> {u['completion_tokens']/dt:.1f} tok/s")
print("  out head:", repr(c[:150]))