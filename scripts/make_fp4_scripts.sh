#!/bin/bash
# 生成 FP4（权重）相关的两组启动脚本，用于回答「LSA 在 FP4 上是否仍有收益」：
#   fp4_lsa.sh    目标模型换 FQ 量化权重 + LSA + DSpine
#   fp4_dense.sh  同一量化权重 + DSpine，但 attention 用 flashinfer 且不开 LSA（dense 基线）
# KV cache 保持 bf16（--kv-cache-dtype 不设 = auto），满足 LSA 的 key_buffer() 硬门。
set -eu
cd /mnt/sda/qyj1044

FP4=/mnt/sda/models/Qwen3.8-27B-NVFP4
BASE=graph_lsa.sh

# --- fp4_lsa.sh：只换模型路径 ---
sed "s#--model-path /mnt/sda/qyj1044/dspine_train/model_view#--model-path $FP4#" "$BASE" > fp4_lsa.sh
chmod +x fp4_lsa.sh

# --- fp4_dense.sh：换模型 + 去掉 LSA 的 json override + attention 换 flashinfer ---
python3 - "$BASE" fp4_dense.sh "$FP4" <<'PY'
import json
import re
import sys

base, out, fp4 = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(base, encoding="utf-8").read()
text = text.replace("--model-path /mnt/sda/qyj1044/dspine_train/model_view", f"--model-path {fp4}")
# attention 后端：triton -> flashinfer
text = text.replace("--attention-backend triton", "--attention-backend flashinfer")
# 去掉整个 --json-model-override-args 行（含它上面的转义续行）
text = re.sub(r'\s*--json-model-override-args "[^\n]*"\s*\\\n', "\n", text)
open(out, "w", encoding="utf-8").write(text)
print("written:", out)
PY
chmod +x fp4_dense.sh

echo "=== fp4_lsa.sh 关键行 ==="
grep -n 'model-path\|attention-backend\|page-size\|cuda-graph\|kv-cache-dtype\|json-model-override' fp4_lsa.sh | cut -c1-150
echo
echo "=== fp4_dense.sh 关键行 ==="
grep -n 'model-path\|attention-backend\|page-size\|cuda-graph\|kv-cache-dtype\|json-model-override' fp4_dense.sh | cut -c1-150
