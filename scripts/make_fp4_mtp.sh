#!/bin/bash
# 生成 FP4 权重 + LSA + MTP(NEXTN) 的启动脚本。
#   fp4_lsa_mtp.sh        FP4 + LSA(triton, page-size 1) + NEXTN 16
#   fp4_lsa_mtp_nograph.sh 同上但禁用 CUDA graph（若 verify 图捕获失败时的退路）
# 关键点：
#   - 必须显式 --kv-cache-dtype bfloat16：FP8 量化 checkpoint 下 auto 会把 KV 解析成 fp8，
#     LSA 的 key_buffer() 硬性要求 bf16，否则 NotImplementedError。
#   - LSA 需要 triton 后端 + page-size 1 + 关 prefill 图（prefill 选择在 host 侧）。
#   - MTP 权重在 FP4 checkpoint 里存在（mtp.fc / mtp.layers.0.*），且被量化 ignore 列表排除（高精度）。
set -eu
cd /mnt/sda/qyj1044

FP4=/mnt/sda/models/Qwen3.8-27B-NVFP4
MAP=/mnt/sda/qyj1044/lsa_code/maps/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt

python3 - "$FP4" "$MAP" <<'PY'
import json
import sys

fp4, lsa_map = sys.argv[1], sys.argv[2]
text = open("nextn16_uds.sh", encoding="utf-8").read()

# 1) 换模型 + 显式 bf16 KV cache
text = text.replace("--model-path /mnt/sda/models/Qwen3.8-27B",
                    f"--model-path {fp4}")
text = text.replace("--tp-size 1 --dtype bfloat16",
                    "--kv-cache-dtype bfloat16 --tp-size 1 --dtype bfloat16")

# 2) LSA 要求 triton 后端 + 单 slot 分页 + 关 prefill 图
text = text.replace("--attention-backend flashinfer",
                    "--attention-backend triton")
text = text.replace("--linear-attn-backend triton",
                    "--page-size 1 \\\n  --disable-prefill-cuda-graph \\\n  --linear-attn-backend triton")

# 3) 挂上 LSA 设计
block = {"lsa": {"projections": lsa_map, "query_heads": "all"}}
override = "--json-model-override-args \"" + json.dumps(block).replace('"', '\\"') + "\" \\"
text = text.replace("  --trust-remote-code \\", f"  {override}\n  --trust-remote-code \\")

open("fp4_lsa_mtp.sh", "w", encoding="utf-8").write(text)
nograph = text.replace("  --cuda-graph-max-bs-decode 32 \\",
                       "  --cuda-graph-max-bs-decode 32 \\\n  --disable-cuda-graph \\")
open("fp4_lsa_mtp_nograph.sh", "w", encoding="utf-8").write(nograph)
print("written fp4_lsa_mtp.sh / fp4_lsa_mtp_nograph.sh")
PY

chmod +x fp4_lsa_mtp.sh fp4_lsa_mtp_nograph.sh
echo "=== fp4_lsa_mtp.sh 关键行 ==="
grep -n 'model-path\|speculative\|attention-backend\|page-size\|kv-cache-dtype\|cuda-graph\|json-model-override\|context-length' fp4_lsa_mtp.sh | cut -c1-150
