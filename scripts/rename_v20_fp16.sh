#!/bin/bash
# 把快照名从 v2.0-fp4 改为 v2.0-fp16：目录、tarball、内部 VERSION 一并改，重算校验和。
set -eu
cd /mnt/sda/qyj1044/releases

OLD=v2.0-fp4-qwen38-lsa-dspine
NEW=v2.0-fp16-qwen38-lsa-dspine

if [ -d "$NEW" ]; then rm -rf "$NEW"; fi
mv "$OLD" "$NEW"
cd "$NEW"

python3 - <<'PY'
path = "VERSION"
text = open(path, encoding="utf-8").read()
text = text.replace("version: v2.0-fp4\n", "version: v2.0-fp16\n", 1)
text = text.replace("tag: v2.0-fp4-qwen38-lsa-dspine\n",
                    "tag: v2.0-fp16-qwen38-lsa-dspine\n", 1)
text = text.replace("name: Qwen3.8-27B LSA + DSpine 部署 — V2.0-FP4\n",
                    "name: Qwen3.8-27B LSA + DSpine 部署 — V2.0-FP16（bf16 权重）\n", 1)
# 在「已知限制」之前补一句精度说明，避免误解为 fp16 而非 bf16
marker = "# 已知限制\n"
note = ("# 精度说明\n"
        "#   本版线上权重与 KV cache 均为 bfloat16（16-bit）。标签 FP16 指 16-bit 精度档，\n"
        "#   非 IEEE fp16；FP4/FP8 混合精度路线见 launch/fp4_*.sh（accept 会下降，未采用）。\n\n")
if marker in text and "精度说明" not in text:
    text = text.replace(marker, note + marker, 1)
open(path, "w", encoding="utf-8").write(text)
print("VERSION 已更新")
for line in text.splitlines()[:3]:
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
echo "=== releases ==="
ls -1 | sed 's/^/  /'
