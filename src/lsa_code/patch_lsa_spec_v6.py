#!/usr/bin/env python3
"""Patch 6 (diagnostic only) - print what the verify read returns, before it is consumed.

`[PATCH spec v4]` feeds a verify block through `selector.decode`. The capture pass reports
bs=12, tokens=192, k=16 - so the read is asked for 192 rows - yet `triton_sparse_read` is
handed 13. This prints the returned shape and its input row count so the discrepancy is
visible in the log rather than inferred.

Diagnostic only: to be reverted or turned into the fix once the number is explained.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/"
    "qwen3_5/backend.py"
)
BACKUP = TARGET + ".pre-v6"
MARK = "[PATCH spec v6]"

OLD = '''            return index.selector.decode(index_queries, positions, _rows.repeat_interleave(_k))
'''

NEW = '''            _lists = index.selector.decode(
                index_queries, positions, _rows.repeat_interleave(_k)
            )
            # [PATCH spec v6] what the read actually returned, against what it was asked for
            dbg(
                "verify read -> lists %s from %s queries, %s rows, positions %s, counts %s"
                % (
                    tuple(_lists.shape),
                    int(index_queries.shape[0]),
                    int(_rows.numel()),
                    tuple(positions.shape),
                    tuple(index.selector.cache.decode_counts(_rows.repeat_interleave(_k)).shape),
                )
            )
            return _lists
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"
    old = OLD.replace("\n", newline)
    count = source.count(old)
    if count != 1:
        print(f"FAILED: {count} matches")
        return 1
    shutil.copyfile(TARGET, BACKUP)
    print(f"backup: {BACKUP}")
    with open(TARGET, "w", newline="") as handle:
        handle.write(source.replace(old, NEW.replace("\n", newline)))
    print(f"patched: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
