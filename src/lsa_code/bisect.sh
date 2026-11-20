#!/bin/bash
cd /mnt/sda/qyj1044/lsa_code
mk() {
  /mnt/nvme1n1/sglang522/bin/python - "$1" "$2" <<'PYEOF'
import json,sys
name,target=sys.argv[1],int(sys.argv[2])
filler=[("The quarterly logistics review covers warehouse throughput, inbound freight scheduling, and inventory turnover "
 "across the regional distribution centres. Each centre reports its dock utilisation and cold-chain compliance.")]
n=max(4,target//40); body="\n\n".join(f"Section {i+1}. "+filler[0] for i in range(n))
p=("Dossier:\n\n"+body+"\n\nQuestion: How many sections are there? Answer with the number only.")
json.dump({"model":"/mnt/sda/models/Qwen3.8-27B","messages":[{"role":"user","content":p}],
           "max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}},
          open(name+".json","w"))
PYEOF
}
for t in 8000 12000 16000 20000 30000; do
  mk s$t $t
  SZ=$(stat -c %s s$t.json)
  R=$(curl -sS --max-time 600 -o /tmp/s$t.out -w '%{http_code}' -X POST http://127.0.0.1:30000/v1/chat/completions \
      -H 'Content-Type: application/json' --data-binary @s$t.json 2>&1)
  echo "target=$t bytes=$SZ curl_result=$R"
done