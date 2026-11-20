#!/bin/bash
# 用「之前 FP4 部署」的原始参数复跑，只把 `sglang serve` 换成 UDS 启动器（我们的隧道需要 UDS）。
#   fp4_prev_ar_uds.sh    = verify_ar.sh 的完整参数（无投机），仅换启动入口
# 目的：验证 FP4 在既有参数下输出是否正常，从而判断之前隔离实验里的乱码是不是我自己
#       强加 --kv-cache-dtype bfloat16 / --page-size 1 造成的。
set -eu
cd /mnt/sda/qyj1044

python3 - <<'PY'
src = "verify_ar.sh"
dst = "fp4_prev_ar_uds.sh"
text = open(src, encoding="utf-8").read()
old = "exec /mnt/nvme1n1/sglang522/bin/sglang serve"
new = "exec /mnt/nvme1n1/sglang522/bin/python /mnt/sda/qyj1044/lsa_code/sglang_uds.py"
if old not in text:
    raise SystemExit("未找到 sglang serve 启动行")
open(dst, "w", encoding="utf-8").write(text.replace(old, new))
print("written:", dst)
PY

chmod +x fp4_prev_ar_uds.sh
echo "=== fp4_prev_ar_uds.sh 参数块 ==="
sed -n '/^exec/,$p' fp4_prev_ar_uds.sh | grep -vE '^\s*$' | cut -c1-160
echo
echo "=== 与 verify_ar.sh 的差异（应只有启动入口一行）==="
diff verify_ar.sh fp4_prev_ar_uds.sh || true
