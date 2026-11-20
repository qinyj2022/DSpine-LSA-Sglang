#!/bin/bash
L=/mnt/sda/qyj1044/logs/lsa_run2.log
echo "=== 证明 LSA 索引器实际运行的日志 ==="
grep -oE "lsa: (memory|indexing|enabled)[^|]{0,180}" $L | sort -u | head -12
echo ""
echo "=== 请求期间的选择/索引记录条数 ==="
grep -c "lsa: memory - a request" $L | sed 's/^/  request-index records: /'
grep -oE "lsa: memory - a request of [0-9]+ keys[^|]{0,120}" $L | tail -3
echo ""
echo "=== 显存与端口 ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
ss -ltn | grep -q 30000 && echo "  port 30000 UP" || echo "  port DOWN"
echo ""
echo "=== 生成的 launcher 文件 ==="
ls -la /mnt/sda/qyj1044/lsa_serve.sh /mnt/sda/qyj1044/dense_serve.sh 2>/dev/null | awk '{print "  "$9, $5"B"}'