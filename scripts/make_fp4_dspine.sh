#!/bin/bash
# 生成「FP4 量化权重 + KV bf16 + LSA + DSpine」启动脚本。
#   fp4_lsa_dspine_kvbf16.sh         正确性/吞吐用（无每步日志）
#   fp4_lsa_dspine_kvbf16_log.sh     同上 + --decode-log-interval 1，用于统计 accept
#
# 前提（已由前面的实验确立）：
#   - KV 必须 bf16 才能满足 LSA 的 key_buffer() 守卫；直接给 fp8 包加 --kv-cache-dtype bfloat16
#     会毒化注意力（checkpoint 的 kv_cache_scheme 已按 fp8 校准 k_scale/v_scale）。
#     所以模型用 -kvbf16 变体（config.json 摘掉 kv_cache_scheme，权重不动）。
#   - DSpine 只拒绝 FP4 的 KV cache dtype；bf16 KV 不触发该拒绝。
set -eu
cd /mnt/sda/qyj1044

python3 - <<'PY'
text = open("fp4_lsa_kvbf16.sh", encoding="utf-8").read()

# DSpine 的投机参数与混合 SSM 要求
dspine = (
    "  --speculative-algorithm DSPINE \\\n"
    "  --speculative-draft-model-path /mnt/sda/qyj1044/dspine_draft \\\n"
    "  --speculative-num-draft-tokens 16 \\\n"
    "  --speculative-draft-attention-backend flashinfer \\\n"
    "  --enable-linear-replayssm-spec \\\n"
)
anchor = "  --tp-size 1"
if anchor not in text:
    raise SystemExit("未找到 --tp-size 锚点")
text = text.replace(anchor, dspine + anchor, 1)

open("fp4_lsa_dspine_kvbf16.sh", "w", encoding="utf-8").write(text)
open("fp4_lsa_dspine_kvbf16_log.sh", "w", encoding="utf-8").write(
    text.replace("--host 0.0.0.0", "--decode-log-interval 1 --host 0.0.0.0")
)
print("written fp4_lsa_dspine_kvbf16.sh / _log.sh")
PY

chmod +x fp4_lsa_dspine_kvbf16.sh fp4_lsa_dspine_kvbf16_log.sh
echo "=== fp4_lsa_dspine_kvbf16.sh 关键行 ==="
grep -nE 'model-path|speculative|attention-backend|page-size|kv-cache-dtype|mem-fraction|mamba|json-model-override' fp4_lsa_dspine_kvbf16.sh | cut -c1-150
