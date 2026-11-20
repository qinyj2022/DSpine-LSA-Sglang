#!/bin/bash
# 生成「FP4 + LSA（不投机）」脚本：模型与注意力保持原样（KV 走 auto → fp8），
# 由 patch v10 让 LSA 自己在读索引键时反量化成 bf16。
#   fp4_lsa_fp8kv.sh  = verify_ar.sh 的参数 + LSA 所需项（triton / page-size 1 / 关 prefill 图）
set -eu
cd /mnt/sda/qyj1044

python3 - <<'PY'
import json

text = open("fp4_prev_ar_uds.sh", encoding="utf-8").read()

# LSA 需要 triton 后端、单 slot 分页、以及关掉 prefill 图（选择在 host 侧）
text = text.replace("--attention-backend flashinfer",
                    "--page-size 1 --disable-prefill-cuda-graph --attention-backend triton")

lsa_map = "/mnt/sda/qyj1044/lsa_code/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt"
block = {"lsa": {"projections": lsa_map, "query_heads": "all"}}
override = '--json-model-override-args "' + json.dumps(block).replace('"', '\\"') + '" \\'
text = text.replace("  --trust-remote-code \\", f"  {override}\n  --trust-remote-code \\")

open("fp4_lsa_fp8kv.sh", "w", encoding="utf-8").write(text)
print("written fp4_lsa_fp8kv.sh")
PY

chmod +x fp4_lsa_fp8kv.sh
echo "=== fp4_lsa_fp8kv.sh 关键行（注意：无 --kv-cache-dtype，KV 保持 auto/fp8）==="
grep -nE 'model-path|attention-backend|page-size|kv-cache-dtype|cuda-graph|json-model-override' fp4_lsa_fp8kv.sh | cut -c1-150
