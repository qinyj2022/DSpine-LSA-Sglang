#!/bin/bash
# 1) 打印其余 FP4 脚本的参数块（不含 export 前缀）
# 2) 找出历史上以 NVFP4 启动过的日志，列出时间/大小，并检查是否有报错
set -u
cd /mnt/sda/qyj1044

echo "############ verify_dspine_mine.sh ############"
date -r verify_dspine_mine.sh '+  修改时间 %Y-%m-%d %H:%M' 2>/dev/null
sed -n '/^exec/,$p' verify_dspine_mine.sh 2>/dev/null | grep -vE '^\s*$' | cut -c1-160

echo
echo "############ verify_ngram.sh ############"
date -r verify_ngram.sh '+  修改时间 %Y-%m-%d %H:%M' 2>/dev/null
sed -n '/^exec/,$p' verify_ngram.sh 2>/dev/null | grep -vE '^\s*$' | cut -c1-160

echo
echo "############ 历史 FP4 日志 ############"
for log in logs/*.log; do
    if grep -q 'NVFP4' "$log" 2>/dev/null; then
        ready=$(grep -c 'ready to roll' "$log" 2>/dev/null || true)
        errs=$(grep -ac 'Traceback\|Error' "$log" 2>/dev/null || true)
        printf '  %-34s %s  %6s  ready=%s errors=%s\n' \
            "$(basename "$log")" "$(date -r "$log" '+%m-%d %H:%M')" \
            "$(du -h "$log" | cut -f1)" "$ready" "$errs"
    fi
done
