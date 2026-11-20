#!/bin/bash
# 生成「FP4 权重 + KV 回 bf16」的 LSA 启动脚本。
#   fp4_lsa_kvbf16.sh = fp4_lsa_fp8kv.sh 的模型路径换成 kvbf16 变体，
#                       显式 --kv-cache-dtype bfloat16，并压低 mem-fraction 避免 LSA 索引表 OOM。
set -eu
cd /mnt/sda/qyj1044

python3 - <<'PY'
text = open("fp4_lsa_fp8kv.sh", encoding="utf-8").read()
text = text.replace("/mnt/sda/models/Qwen3.8-27B-NVFP4",
                    "/mnt/sda/models/Qwen3.8-27B-NVFP4-kvbf16")
# 显式 bf16 KV（变体已无 kv_cache_scheme，auto 也会给 bf16；写明更可控）
text = text.replace("--page-size 1", "--kv-cache-dtype bfloat16 --page-size 1")
# LSA 索引结构随池规模增长，FP4 权重省下的显存别全给 KV 池
import re
text = re.sub(r"--mem-fraction-static [0-9.]+", "--mem-fraction-static 0.70", text)
open("fp4_lsa_kvbf16.sh", "w", encoding="utf-8").write(text)
print("written fp4_lsa_kvbf16.sh")
PY

chmod +x fp4_lsa_kvbf16.sh
echo "=== 关键行 ==="
grep -nE 'model-path|attention-backend|page-size|kv-cache-dtype|mem-fraction|json-model-override' fp4_lsa_kvbf16.sh | cut -c1-150
