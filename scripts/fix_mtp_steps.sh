#!/bin/bash
# 修正 MTP 调参脚本里的 num-steps / num-draft-tokens（上一步 Python 正则转义没生效）。
set -eu
cd /mnt/sda/qyj1044

python3 - <<'PY'
import re

for steps, path in ((4, "fp4_lsa_mtp4_kvbf16.sh"), (8, "fp4_lsa_mtp8_kvbf16.sh")):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"--speculative-num-steps \d+", f"--speculative-num-steps {steps}", text)
    text = re.sub(
        r"--speculative-num-draft-tokens \d+",
        f"--speculative-num-draft-tokens {steps + 1}",
        text,
    )
    open(path, "w", encoding="utf-8").write(text)
    lines = [l.strip() for l in text.splitlines() if "num-steps" in l or "num-draft-tokens" in l]
    print(path, "->", lines)
PY
