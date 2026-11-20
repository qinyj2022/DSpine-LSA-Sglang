#!/bin/bash
# 1) 列出这代 sglang 支持的 --speculative-algorithm 取值（为 MTP / Engram 选型）
# 2) 生成 FP4 + LSA + MTP(NEXTN) 脚本：
#    fp4_lsa_mtp_kvbf16.sh      正确性/吞吐
#    fp4_lsa_mtp_kvbf16_log.sh  + --decode-log-interval 1，统计 accept
set -eu
cd /mnt/sda/qyj1044
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/sglang/srt

echo "=== --speculative-algorithm 的取值 ==="
grep -rn 'SpeculativeAlgorithm' $SP/arg_groups/fields/spec.py | head -5
grep -rn 'choices=\|_ALGO' $SP/arg_groups/fields/spec.py | head -8
echo "--- 枚举类 ---"
grep -rn 'class SpeculativeAlgorithm' -A 40 $SP/speculative/*.py 2>/dev/null | grep -E '^\S+[-:]\s+[A-Z_]+ =' | head -20

echo
echo "=== 是否有 ENGRAM ==="
grep -rin 'engram' --include=*.py $SP/ 2>/dev/null | head -8 || echo "（srt 内未见 engram）"

echo
python3 - <<'PY'
text = open("fp4_lsa_kvbf16.sh", encoding="utf-8").read()
# MTP / NEXTN：用模型自带的 mtp 头（checkpoint 里存在且未量化）
mtp = (
    "  --speculative-algorithm NEXTN \\\n"
    "  --speculative-num-steps 16 \\\n"
    "  --speculative-eagle-topk 1 \\\n"
    "  --speculative-num-draft-tokens 16 \\\n"
    "  --enable-linear-replayssm-spec \\\n"
)
anchor = "  --tp-size 1"
text = text.replace(anchor, mtp + anchor, 1)
open("fp4_lsa_mtp_kvbf16.sh", "w", encoding="utf-8").write(text)
open("fp4_lsa_mtp_kvbf16_log.sh", "w", encoding="utf-8").write(
    text.replace("--host 0.0.0.0", "--decode-log-interval 1 --host 0.0.0.0")
)
print("written fp4_lsa_mtp_kvbf16.sh / _log.sh")
PY

chmod +x fp4_lsa_mtp_kvbf16.sh fp4_lsa_mtp_kvbf16_log.sh
echo
echo "=== fp4_lsa_mtp_kvbf16.sh 关键行 ==="
grep -nE 'model-path|speculative|attention-backend|page-size|kv-cache-dtype|mem-fraction|json-model' fp4_lsa_mtp_kvbf16.sh | cut -c1-140
