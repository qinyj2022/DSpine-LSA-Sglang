#!/bin/bash
# 复验用：确认 SPL 补丁在运行进程里是惰性的（未设置 SPL_ENABLE）。
set -u
pid=$(pgrep -f sglang_uds | head -1)
if [ -z "$pid" ]; then
    echo "no sglang process"
    exit 0
fi
echo "pid=$pid"
n=$(tr '\0' '\n' < "/proc/$pid/environ" | grep -c '^SPL_ENABLE')
echo "SPL_ENABLE entries: $n    (0 = inert)"
echo "--- SPL_/LSA_ env ---"
tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(SPL_|LSA_)' | head -5
echo "--- 启动脚本 ---"
ps -eo cmd | grep '[s]glang_uds' | head -1 | grep -oE 'restart_any|graph_lsa\.sh|spl_[a-z0-9_]+\.sh|fp4_[a-z0-9_]+\.sh' | head -3
