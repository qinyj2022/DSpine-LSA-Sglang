#!/bin/bash
cd /mnt/sda/qyj1044/lsa_code
LOG=$1; TAG=$2
for R in 1 2; do
  L0=$(wc -l < $LOG); T0=$(date +%s.%N)
  curl -sS --max-time 1200 -o /tmp/g$R_$TAG.out -w "  r$R http=%{http_code}\n" -X POST \
    http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' --data-binary @gen16k.json
  T1=$(date +%s.%N)
  /mnt/nvme1n1/sglang522/bin/python - "$TAG" "$T0" "$T1" "$R" <<'PYEOF'
import json,sys
tag,t0,t1,r=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
d=json.load(open(f"/tmp/g{r}_{tag}.out")); u=d["usage"]; gen=u["completion_tokens"]; wall=t1-t0
print(f"[{tag} run{r}] prompt_tok={u['prompt_tokens']} gen={gen} wall={wall:.2f}s -> {gen/wall:.2f} tok/s")
PYEOF
  echo "    decode lines:"
  tail -n +$((L0+1)) $LOG | grep -oE "gen throughput \(token/s\): [0-9.]+" | tail -3 | sed 's/^/      /'
done