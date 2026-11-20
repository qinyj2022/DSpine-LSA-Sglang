#!/bin/bash
cd /mnt/sda/qyj1044
cp logs/lsa.log logs/lsa_run1.log 2>/dev/null
grep -v -- '--json-model-override-args' lsa_serve.sh | sed 's/ --trust-remote-code \\/ --trust-remote-code \\/' > dense_serve.sh
sed -i 's/^export SGLANG_PLUGINS=lsa$/export SGLANG_PLUGINS=/' dense_serve.sh
chmod +x dense_serve.sh
echo "=== dense_serve.sh 差异确认 ==="
grep -nE "SGLANG_PLUGINS|json-model-override|attention-backend|page-size" dense_serve.sh
pkill -9 -f "launch_server" 2>/dev/null; sleep 8
truncate -s 0 logs/dense.log
setsid nohup bash /mnt/sda/qyj1044/dense_serve.sh > /mnt/sda/qyj1044/logs/dense.log 2>&1 < /dev/null &
echo DENSE_LAUNCHED