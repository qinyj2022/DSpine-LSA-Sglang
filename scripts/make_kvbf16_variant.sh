#!/bin/bash
# 造一个「权重仍量化、但 KV cache 回到 bf16」的 checkpoint 变体。
#
# 原理（sglang 源码依据）：
#   compressed_tensors.py 的 kv_cache_quant_algo 属性在 kv_cache_scheme 存在时返回 "FP8"，
#   configure_kv_cache_dtype 据此把 --kv-cache-dtype auto 解析成 fp8；
#   注释原话：“loaded scales need the fp8 pool they calibrate, never bf16”。
#   所以只要从 config.json 摘掉 kv_cache_scheme，KV 就会解析成 bf16，
#   而量化权重（FP8 attention proj / NVFP4 MLP）完全不动。
#
# 用法: bash make_kvbf16_variant.sh
set -eu

SRC=/mnt/sda/models/Qwen3.8-27B-NVFP4
DST=/mnt/sda/models/Qwen3.8-27B-NVFP4-kvbf16

rm -rf "$DST"
mkdir -p "$DST"

# 大文件软链接，其余小文件复制
for f in "$SRC"/*; do
    name=$(basename "$f")
    if [ "$name" = "config.json" ]; then
        continue
    fi
    ln -s "$f" "$DST/$name"
done

python3 - "$SRC/config.json" "$DST/config.json" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
config = json.load(open(src, encoding="utf-8"))
quant = config.get("quantization_config") or {}
had = "kv_cache_scheme" in quant
quant.pop("kv_cache_scheme", None)
# 显式声明为不做 KV 量化，让 configure_kv_cache_dtype 不再报告 FP8
config["quantization_config"] = quant
json.dump(config, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"kv_cache_scheme removed: {had}")
print("quantization_config keys:", sorted(quant.keys()))
PY

echo "=== 变体目录 ==="
ls -l "$DST" | head -20
echo
echo "=== 确认 kv_cache_scheme 不存在 ==="
grep -c 'kv_cache_scheme' "$DST/config.json" || echo "0 (good)"
