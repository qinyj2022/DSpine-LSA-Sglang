#!/bin/bash
# 生成 bf16 + LSA + MTP 脚本：按行替换 DSpine 的投机参数（它们并不相邻）。
# 动机：DSpine 的 sampled（非贪心）验证路径会抛
#   RuntimeError: DSPINE sampled verify reached without this round's sampled draft chain
# 并拖垮整个 scheduler；MTP 是 sglang 原生路径，能正常服务采样请求。
set -eu
cd /mnt/sda/qyj1044

python3 - <<'PY'
lines = open("graph_lsa.sh", encoding="utf-8").read().splitlines(keepends=True)
subs = [
    ("--speculative-algorithm DSPINE", "--speculative-algorithm NEXTN"),
    ("--speculative-draft-model-path", "--speculative-num-steps 8"),
    ("--speculative-num-draft-tokens 16", "--speculative-eagle-topk 1"),
    ("--speculative-draft-attention-backend", "--speculative-num-draft-tokens 9"),
]
done = {old: False for old, _ in subs}
out = []
for line in lines:
    for old, new in subs:
        if old in line and not done[old]:
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}{new} \\\n"
            done[old] = True
            break
    out.append(line)
missing = [old for old, hit in done.items() if not hit]
if missing:
    raise SystemExit("未找到: " + ", ".join(missing))
open("graph_lsa_mtp.sh", "w", encoding="utf-8").write("".join(out))
print("written graph_lsa_mtp.sh")
PY

chmod +x graph_lsa_mtp.sh
echo "=== 关键行 ==="
grep -nE "model-path|speculative|attention-backend|page-size|cuda-graph" graph_lsa_mtp.sh | cut -c1-110
