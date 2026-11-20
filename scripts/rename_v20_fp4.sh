#!/bin/bash
# 把 v2.0 快照改名为 V2.0-FP4：目录、tarball、内部 VERSION 一并改，重算校验和。
set -eu
cd /mnt/sda/qyj1044/releases

OLD=v2.0-qwen38-lsa-dspine
NEW=v2.0-fp4-qwen38-lsa-dspine

if [ -d "$NEW" ]; then
    echo "目标目录已存在，先删除: $NEW"
    rm -rf "$NEW"
fi
mv "$OLD" "$NEW"

cd "$NEW"
python3 - <<'PY'
import re

path = "VERSION"
text = open(path, encoding="utf-8").read()
text = text.replace("version: v2.0\n", "version: v2.0-fp4\n", 1)
text = text.replace("tag: v2.0-qwen38-lsa-dspine\n",
                    "tag: v2.0-fp4-qwen38-lsa-dspine\n", 1)
text = text.replace("name: Qwen3.8-27B LSA + DSpine 部署\n",
                    "name: Qwen3.8-27B LSA + DSpine 部署 — V2.0-FP4\n", 1)
open(path, "w", encoding="utf-8").write(text)
print("VERSION 已更新")
for line in text.splitlines()[:4]:
    print("  " + line)
PY

echo "=== 重算校验和 ==="
( cd source && find . -type f ! -name MANIFEST.sha256 ! -name ORIGIN.md | sort | xargs sha256sum > MANIFEST.sha256 )
find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
echo "  top    : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)"
( cd source && echo "  source : $(sha256sum -c MANIFEST.sha256 2>&1 | grep -c ': OK$') / $(wc -l < MANIFEST.sha256)" )

echo "=== 重打 tarball ==="
cd /mnt/sda/qyj1044/releases
rm -f "$OLD.tar.gz" "$NEW.tar.gz"
tar -czf "$NEW.tar.gz" "$NEW"
ls -lh "$NEW.tar.gz" | awk '{print "  size", $5}'
sha256sum "$NEW.tar.gz"
echo "=== releases 目录 ==="
ls -1 | sed 's/^/  /'
