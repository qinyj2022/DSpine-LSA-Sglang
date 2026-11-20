#!/bin/bash
# 生成纯 dense 基线脚本：bf16 权重、flashinfer、不开 LSA、不开投机，其余与线上一致。
#   dense_base.sh   = graph_lsa.sh 去掉 LSA override + 去掉 DSpine，attention 换 flashinfer
set -eu
cd /mnt/sda/qyj1044

python3 - <<'PY'
import re

text = open("graph_lsa.sh", encoding="utf-8").read()

# 1) 去掉 LSA 的 --json-model-override-args 行
text = re.sub(r'\n\s*--json-model-override-args "[^\n]*" \\', "", text)
# 2) 去掉 DSpine 的投机参数（保留 mamba/linear 等无关项）
for pattern in (
    r'\n\s*--speculative-algorithm DSPINE \\',
    r'\n\s*--speculative-draft-model-path [^\n]*\\',
    r'\n\s*--speculative-num-draft-tokens \d+ \\',
    r'\n\s*--speculative-draft-attention-backend [^\n]*\\',
    r'\n\s*--enable-linear-replayssm-spec \\',
):
    text = re.sub(pattern, "", text)
# 3) attention 换 flashinfer（dense 最快路径），triton 专用于 LSA
text = text.replace("--attention-backend triton", "--attention-backend flashinfer")
# 4) page-size 1 只对 LSA 有意义，dense 用默认
text = text.replace("--page-size 1 \\\n", "")
# 5) prefill 图对 dense 可以开
text = text.replace("--disable-prefill-cuda-graph \\\n", "")

open("dense_base.sh", "w", encoding="utf-8").write(text)
print("written dense_base.sh")
PY

chmod +x dense_base.sh
echo "=== 关键行 ==="
grep -nE 'model-path|speculative|attention-backend|page-size|cuda-graph|mem-fraction|context-length|json-model' dense_base.sh | cut -c1-120
