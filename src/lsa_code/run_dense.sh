#!/bin/bash
for i in $(seq 1 40); do
  if grep -qE "ready to roll|Traceback" /mnt/sda/qyj1044/logs/dense.log 2>/dev/null; then break; fi
  sleep 10
done
grep -E "ready to roll" /mnt/sda/qyj1044/logs/dense.log | tail -1
grep -c "lsa" /mnt/sda/qyj1044/logs/dense.log | sed 's/^/  lsa mentions: /'
cd /mnt/sda/qyj1044/lsa_code
LOG=/mnt/sda/qyj1044/logs/dense.log
for P in n8k n16k; do
  L0=$(wc -l < $LOG); T0=$(date +%s.%N)
  code=$(curl -sS --max-time 900 -o /tmp/d_$P.out -w '%{http_code}' -X POST \
    http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' --data-binary @$P.json 2>&1)
  T1=$(date +%s.%N)
  /mnt/nvme1n1/sglang522/bin/python - "$P" "$T0" "$T1" "$code" <<'PYEOF'
import json,sys
p,t0,t1,code=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
try:
    r=json.load(open("/tmp/d_"+p+".out")); u=r["usage"]; c=r["choices"][0]["message"]["content"]
    print(f"[dense {p}] http={code} wall={t1-t0:.2f}s prompt_tok={u['prompt_tokens']} needle={'BLUEHERON-7742' in c} out={c.strip()[:40]!r}")
except Exception as e:
    print(f"[dense {p}] http={code} parse_failed {e}")
PYEOF
  tail -n +$((L0+1)) $LOG | grep -oE "Prefill batch.*" | tail -2 | sed 's/^/    /'
done