#!/bin/bash
# 1) 打印既有 NGRAM 脚本的参数（prompt lookup 草稿）
# 2) 生成三份待测脚本（都在 FP4 + LSA + kvbf16 底座上）：
#      fp4_lsa_mtp4_kvbf16.sh     MTP, speculative-num-steps 4
#      fp4_lsa_mtp8_kvbf16.sh     MTP, speculative-num-steps 8
#      fp4_lsa_ngram_kvbf16.sh    NGRAM（prompt/n-gram lookup 草稿）
set -eu
cd /mnt/sda/qyj1044

echo "=== verify_ngram16b10.sh 参数 ==="
grep -nE 'model-path|speculative|attention-backend|mem-fraction|context-length' verify_ngram16b10.sh 2>/dev/null | cut -c1-150 || echo "（不存在）"

python3 - <<'PY'
import re

base = open("fp4_lsa_mtp_kvbf16.sh", encoding="utf-8").read()

# MTP 调参：num-steps 4 / 8（eagle_topk=1 时 draft tokens 自动 = steps+1）
for steps in (4, 8):
    text = re.sub(r"--speculative-num-steps \\d+", f"--speculative-num-steps {steps}", base)
    text = re.sub(r"--speculative-num-draft-tokens \\d+",
                  f"--speculative-num-draft-tokens {steps + 1}", text)
    open(f"fp4_lsa_mtp{steps}_kvbf16.sh", "w", encoding="utf-8").write(text)
    print(f"written fp4_lsa_mtp{steps}_kvbf16.sh  (steps={steps}, draft_tokens={steps+1})")

# NGRAM：把 MTP 的投机块整体替换成 prompt lookup
ngram = (
    "  --speculative-algorithm NGRAM \\\n"
    "  --speculative-num-draft-tokens 16 \\\n"
    "  --speculative-ngram-match-type BFS \\\n"
    "  --speculative-ngram-max-trie-depth 16 \\\n"
    "  --speculative-ngram-max-bfs-breadth 10 \\\n"
    "  --enable-linear-replayssm-spec \\\n"
)
old = (
    "  --speculative-algorithm NEXTN \\\n"
    "  --speculative-num-steps 16 \\\n"
    "  --speculative-eagle-topk 1 \\\n"
    "  --speculative-num-draft-tokens 16 \\\n"
    "  --enable-linear-replayssm-spec \\\n"
)
if old not in base:
    raise SystemExit("MTP 投机块未找到")
text = base.replace(old, ngram, 1)
open("fp4_lsa_ngram_kvbf16.sh", "w", encoding="utf-8").write(text)
open("fp4_lsa_ngram_kvbf16_log.sh", "w", encoding="utf-8").write(
    text.replace("--host 0.0.0.0", "--decode-log-interval 1 --host 0.0.0.0")
)
print("written fp4_lsa_ngram_kvbf16.sh / _log.sh")
PY

chmod +x fp4_lsa_mtp4_kvbf16.sh fp4_lsa_mtp8_kvbf16.sh fp4_lsa_ngram_kvbf16.sh fp4_lsa_ngram_kvbf16_log.sh
echo
echo "=== 各脚本的投机行 ==="
for f in fp4_lsa_mtp4_kvbf16.sh fp4_lsa_mtp8_kvbf16.sh fp4_lsa_ngram_kvbf16.sh; do
    echo "--- $f"
    grep -E 'speculative' "$f" | cut -c1-90
done
