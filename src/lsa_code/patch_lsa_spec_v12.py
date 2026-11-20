#!/usr/bin/env python3
"""Patch 12 - keep LSA's maintenance to the real requests of a padded batch.

`logs/final_best2.log`, first concurrent run at 12 in flight:

    ValueError: zip() argument 2 is longer than argument 1

`registry.rows_of()` returns every row of `forward_batch.req_pool_indices`, but a verify batch
replayed from a CUDA graph is padded past its real requests, while `forward_batch.rids` holds
only the real ones. `Selector.maintain` then zips `(rids, rows, ends)` with `strict=True` and
raises. LSA+DSpine had only ever run at `bs == 1`, where padding never shows up.

Fix: cut the pool rows to the number of real request ids, so `lsa_rows`, `ends` and `rids` all
describe the same requests - and the padded rows are left alone, exactly as sglang does for a
padded decode batch.

Idempotent: refuses to patch twice, and writes `registry.py.pre-v12` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/index/registry.py"
BACKUP = TARGET + ".pre-v12"
MARK = "[PATCH spec v12]"

OLD = '''    rows = forward_batch.req_pool_indices.tolist()
    ROWS.update(zip(rids, rows, strict=False))
'''

NEW = '''    rows = forward_batch.req_pool_indices.tolist()
    # [PATCH spec v12] a captured verify batch is padded past its real requests, and `rids` holds
    # only the real ones - keep the rows aligned with them, as a padded decode batch is left alone
    if len(rows) > len(rids):
        rows = rows[: len(rids)]
    ROWS.update(zip(rids, rows, strict=False))
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
