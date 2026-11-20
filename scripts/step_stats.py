#!/usr/bin/env python3
"""Summarize per-step time from sglang decode log lines.

step_ms = 1000 * accept_len / gen_throughput, which removes the accept-length
content effect and leaves only the config-attributable per-step cost.

用法：
    python step_stats.py logs/acclog.log 12
"""

from __future__ import annotations

import re
import sys

LINE = re.compile(
    r"#full token: (?P<ctx>\d+).*?accept len: (?P<acc>[0-9.]+)"
    r".*?gen throughput \(token/s\): (?P<tp>[0-9.]+)"
)


def main() -> int:
    """Print ctx/accept/throughput/step_ms for the trailing decode lines."""
    path = sys.argv[1]
    take = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LINE.search(line)
            if not match:
                continue
            ctx = int(match["ctx"])
            acc = float(match["acc"])
            tp = float(match["tp"])
            if ctx == 0 or tp < 1.0:
                continue
            rows.append((ctx, acc, tp, 1000.0 * acc / tp))
    for ctx, acc, tp, step in rows[-take:]:
        print(f"ctx={ctx:7d} accept={acc:5.2f} thpt={tp:7.2f} step_ms={step:6.1f}")
    steady = [row for row in rows[-take:] if row[0] > 0]
    if steady:
        mean_step = sum(row[3] for row in steady) / len(steady)
        mean_acc = sum(row[1] for row in steady) / len(steady)
        print(f"mean over last {len(steady)}: accept={mean_acc:.2f} step_ms={mean_step:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
