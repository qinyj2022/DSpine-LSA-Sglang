#!/bin/bash
# 从 SGLang 日志里统计投机接受率。
# 用法: bash accept_stats.sh <日志文件> [起始行号]
# 依赖 "Decode batch ... accept len: X.XX, accept rate: Y" 这类行（需 --decode-log-interval 1）。
set -u
LOG="${1:?log path required}"
FROM="${2:-1}"
tr '\r' '\n' < "$LOG" | tail -n +"$FROM" | grep -oE 'accept len: [0-9.]+' \
  | sed 's/accept len: //' | /usr/bin/python3 -c '
import sys
values = [float(line) for line in sys.stdin if line.strip()]
if not values:
    print("no samples")
    raise SystemExit
values.sort()
count = len(values)
mean = sum(values) / count
block = 16.0
print(
    "steps=%d mean_accept=%.3f median=%.1f rate=%.3f "
    "p10=%.1f p90=%.1f max=%.0f full=%d"
    % (
        count,
        mean,
        values[count // 2],
        max(0.0, (mean - 1.0) / (block - 1.0)),
        values[count // 10],
        values[count * 9 // 10],
        values[-1],
        sum(1 for v in values if v >= block),
    )
)
'
