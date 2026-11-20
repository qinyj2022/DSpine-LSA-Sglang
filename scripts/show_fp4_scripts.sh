#!/bin/bash
# 打印远端既有的 FP4 启动脚本里与运行参数相关的行，用于复现"之前能跑"的配置。
set -u
cd /mnt/sda/qyj1044

FILES="run_dspine.sh verify_ar.sh verify_dspine.sh verify_dspine_mine.sh verify_ngram.sh"
PATTERN='model-path|quantization|kv-cache|attention-backend|--dtype|page-size|speculative|cuda-graph|mem-fraction|context-length|json-model|host|port|linear-attn|mamba'

for f in $FILES; do
    echo "################ $f ################"
    if [ -f "$f" ]; then
        echo "--- 修改时间: $(date -r "$f" '+%Y-%m-%d %H:%M')  行数: $(wc -l < "$f")"
        grep -nE "$PATTERN" "$f" | cut -c1-170
    else
        echo "（不存在）"
    fi
    echo
done
