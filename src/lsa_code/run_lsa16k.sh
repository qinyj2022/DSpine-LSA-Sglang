#!/bin/bash
cd /mnt/sda/qyj1044
cp logs/dense.log logs/dense_run1.log 2>/dev/null
pkill -9 -f "launch_server" 2>/dev/null; sleep 8
truncate -s 0 logs/lsa_run2.log
setsid nohup bash /mnt/sda/qyj1044/lsa_serve.sh > /mnt/sda/qyj1044/logs/lsa_run2.log 2>&1 < /dev/null &
for i in $(seq 1 40); do
  if grep -qE "ready to roll|Traceback" /mnt/sda/qyj1044/logs/lsa_run2.log 2>/dev/null; then break; fi
  sleep 10
done
grep -E "lsa: enabled|ready to roll" /mnt/sda/qyj1044/logs/lsa_run2.log | head -2
cd /mnt/sda/qyj1044/lsa_code
LOG=/mnt/sda/qyj1044/logs/lsa_run2.log
for R in 1 2; do
  L0=$(wc -l < $LOG); T0=$(date +%s.%N)
  curl -sS --max-time 1200 -o /tmp/lsa_g${R}.out -w "  r$R http=%{http_code}\n" -X POST \
    http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' --data-binary @gen16k.json
  T1=$(date +%s.%N)
  /mnt/nvme1n1/sglang522/bin/python - "$T0" "$T1" "$R" <<'PYEOF'
import json,sys
t0,t1,r=float(sys.argv[1]),float(sys.argv[2]),sys.argv[3]
d=json.load(open(f"/tmp/lsa_g{r}.out")); u=d["usage"]; gen=u["completion_tokens"]; wall=t1-t0
print(f"[LSA run{r}] prompt_tok={u['prompt_tokens']} gen={gen} wall={wall:.2f}s -> {gen/wall:.2f} tok/s")
print("   head:", repr(d["choices"][0]["message"]["content"][:100]))
PYEOF
  echo "    decode lines:"
  tail -n +$((L0+1)) $LOG | grep -oE "gen throughput \(token/s\): [0-9.]+" | tail -3 | sed 's/^/      /'
done