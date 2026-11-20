#!/bin/bash
cd /mnt/sda/qyj1044/lsa_code
LOG=$1; TAG=$2
L0=$(wc -l < $LOG); T0=$(date +%s.%N)
curl -sS --max-time 900 -o /tmp/gen_$TAG.out -w "  http=%{http_code}\n" -X POST \
  http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' --data-binary @n16k_gen.json
T1=$(date +%s.%N)
/mnt/nvme1n1/sglang522/bin/python - "$TAG" "$T0" "$T1" <<'PYEOF'
import json,sys
tag,t0,t1=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
r=json.load(open("/tmp/gen_"+tag+".out")); u=r["usage"]
gen=u["completion_tokens"]; wall=t1-t0
print(f"[{tag}] prompt_tok={u['prompt_tokens']} gen={gen} wall={wall:.2f}s  e2e_gen_rate={gen/wall:.2f} tok/s")
PYEOF
echo "  --- 该请求的 Decode 行 ---"
tail -n +$((L0+1)) $LOG | grep -oE "Decode batch.*gen throughput \(token/s\): [0-9.]+" | sed 's/.*\(gen throughput.*\)/    \1/' | tail -5
tail -n +$((L0+1)) $LOG | grep -c "Decode batch" | sed 's/^/    decode steps: /'