#!/bin/bash
cd /mnt/sda/qyj1044/lsa_code
echo "=== A) curl 关闭 Expect ==="
curl -sS --max-time 600 -o /tmp/a.out -w 'http=%{http_code}\n' -X POST \
  -H 'Content-Type: application/json' -H 'Expect:' --data-binary @s20000.json \
  http://127.0.0.1:30000/v1/chat/completions 2>&1 | tail -2
echo "=== B) curl --http1.0 ==="
curl -sS --http1.0 --max-time 600 -o /tmp/b.out -w 'http=%{http_code}\n' -X POST \
  -H 'Content-Type: application/json' --data-binary @s20000.json \
  http://127.0.0.1:30000/v1/chat/completions 2>&1 | tail -2
echo "=== C) python http.client 分块发送 ==="
/mnt/nvme1n1/sglang522/bin/python - <<'PYEOF'
import http.client, json
body=open('/mnt/sda/qyj1044/lsa_code/s20000.json','rb').read()
print("  bytes",len(body))
try:
    c=http.client.HTTPConnection("127.0.0.1",30000,timeout=900)
    c.putrequest("POST","/v1/chat/completions")
    c.putheader("Content-Type","application/json")
    c.putheader("Content-Length",str(len(body)))
    c.endheaders()
    step=8192
    for i in range(0,len(body),step):
        c.send(body[i:i+step])
    r=c.getresponse()
    d=r.read()
    print("  http",r.status,"len",len(d))
    print("  out",json.loads(d)["choices"][0]["message"]["content"][:60])
except Exception as e:
    print("  FAILED:",type(e).__name__,e)
PYEOF