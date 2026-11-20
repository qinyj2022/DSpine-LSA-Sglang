#!/bin/bash
# 生成「FP4 权重 + 不投机」的两组对照脚本，用于回答：LSA 在 FP4 上能否有正常收益
# （FP4 不能和 DSpine 同用，所以只能在非投机配置下测。）
#   fp4_lsa_alone.sh    FP4 权重 + LSA（attention triton, page-size 1）
#   fp4_dense_alone.sh  FP4 权重 + dense（attention flashinfer，去掉 LSA override）
# 两者都显式 --kv-cache-dtype bfloat16，否则 FP8 量化 checkpoint 会把 KV 解析成 fp8，LSA 直接拒。
set -eu
cd /mnt/sda/qyj1044

FP4=/mnt/sda/models/Qwen3.8-27B-NVFP4
BASE=/mnt/sda/qyj1044/lsa_uds.sh
OLD_MODEL=/mnt/sda/qyj1044/dspine_train/model_view

python3 - "$BASE" "$FP4" "$OLD_MODEL" <<'PY'
import re
import sys

base, fp4, old_model = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(base, encoding="utf-8").read()

lsa = text.replace(old_model, fp4).replace(
    "--page-size 1", "--kv-cache-dtype bfloat16 --page-size 1"
)
open("fp4_lsa_alone.sh", "w", encoding="utf-8").write(lsa)

dense = text.replace(old_model, fp4).replace(
    "--page-size 1", "--kv-cache-dtype bfloat16 --page-size 1"
).replace("--attention-backend triton", "--attention-backend flashinfer")
# 去掉 LSA 的 --json-model-override-args 及其续行
dense = re.sub(r'\n\s*--json-model-override-args "[^\n]*" \\', "", dense)
open("fp4_dense_alone.sh", "w", encoding="utf-8").write(dense)
print("written fp4_lsa_alone.sh / fp4_dense_alone.sh")
PY

chmod +x fp4_lsa_alone.sh fp4_dense_alone.sh
echo "=== fp4_lsa_alone.sh ==="
grep -n 'model-path\|attention-backend\|kv-cache-dtype\|json-model-override\|page-size' fp4_lsa_alone.sh | cut -c1-140
echo "=== fp4_dense_alone.sh ==="
grep -n 'model-path\|attention-backend\|kv-cache-dtype\|json-model-override\|page-size' fp4_dense_alone.sh | cut -c1-140
