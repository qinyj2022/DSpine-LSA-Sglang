#!/bin/bash
cd /mnt/sda/qyj1044/lsa_code
LOG=/mnt/sda/qyj1044/logs/lsa.log
for P in p8k p30k; do
  L0=$(wc -l < $LOG)
  echo "--- $P ---"
  T0=$(date +%s.%N)
  curl -sS --max-time 900 -o /tmp/$P.out -X POST http://127.0.0.1:30000/v1/chat/completions \
       -H 'Content-Type: application/json' --data @$P.json
  T1=$(date +%s.%N)
  /mnt/nvme1n1/sglang522/bin/python - "$P" "$T0" "$T1" <<'PYEOF'
import json,sys
p,t0,t1=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
r=json.load(open("/tmp/"+p+".out"))
u=r["usage"]; c=r["choices"][0]["message"]["content"]
print(f"  wall={t1-t0:.2f}s prompt_tokens={u['prompt_tokens']} completion={u['completion_tokens']}")
print(f"  needle_found={'BLUEHERON-7742' in c}  out={c.strip()[:50]!r}")
PYEOF
  tail -n +$((L0+1)) $LOG | grep -oE "(Prefill|Decode) batch.*" | tail -6 | sed 's/^/  /'
done