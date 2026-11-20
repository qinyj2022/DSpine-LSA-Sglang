#!/bin/bash
# 逐项定位：在「之前部署能正常跑」的 FP4 AR 参数上，单独加 LSA 所要求的一个参数。
#   fp4_prev_pgsz1.sh        只加 --page-size 1
#   fp4_prev_bf16kv.sh       只加 --kv-cache-dtype bfloat16
#   fp4_prev_triton.sh       只把 attention 换成 triton（LSA 需要 triton 后端）
set -eu
cd /mnt/sda/qyj1044

python3 - <<'PY'
base = open("fp4_prev_ar_uds.sh", encoding="utf-8").read()

# 只加 page-size 1
pgsz = base.replace("--attention-backend flashinfer",
                    "--page-size 1 --attention-backend flashinfer")
open("fp4_prev_pgsz1.sh", "w", encoding="utf-8").write(pgsz)

# 只加 bf16 KV cache
kv = base.replace("--attention-backend flashinfer",
                  "--kv-cache-dtype bfloat16 --attention-backend flashinfer")
open("fp4_prev_bf16kv.sh", "w", encoding="utf-8").write(kv)

# 只换 triton 后端（+ page-size 1，因为 triton 在 LSA 里要求单 slot；先只换后端看是否够）
triton = base.replace("--attention-backend flashinfer",
                      "--attention-backend triton")
open("fp4_prev_triton.sh", "w", encoding="utf-8").write(triton)
print("written 3 probes")
PY

chmod +x fp4_prev_pgsz1.sh fp4_prev_bf16kv.sh fp4_prev_triton.sh
for f in fp4_prev_pgsz1.sh fp4_prev_bf16kv.sh fp4_prev_triton.sh; do
    echo "=== $f ==="
    grep -E 'attention-backend|page-size|kv-cache-dtype' "$f" | cut -c1-120
done
