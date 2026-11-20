#!/usr/bin/env python3
"""Patch 7 - size the decode listing buffers for a verify block, not one row a request.

Evidence (`logs/exp_dbg2.log`, capture pass, bs=12, k=16):

    [LSA-DBG2] verify q=(192, 6144) key=(192, 4, 256) bs=12 tokens=192 k=16
    [LSA-DBG2] verify read -> lists (13, 4612) from 192 queries, 12 rows, positions (192,), counts (192,)
    RuntimeError: shape '[13, 24, 256]' is invalid for input of size 1179648

192 rows went in and 13 came out. 13 is the request table's own row count
(`max_running_requests` 12 + 1). The cause is `Selector.__init__`:

    self.decode_top_k = TopK(top_p, request_count, device, exact=True)

`TopK` with `rows=` holds its listing in buffers sized once and slices them per call
(`self.held[0][:rows, :width]`, in `TopK.listed`), so a call that asks for more rows than
the buffer holds is silently cut to the buffer's length.

Under speculative decoding the decode-shaped read is handed a verify block - one query row
per drafted token, `k` of them to a request - so the buffers must hold `request_count * k`
rows. `request_count` already covers the graph's padded batch (its buckets stay within
`max_running_requests`, and the table carries one row more than that).

Idempotent: refuses to patch twice, and writes `selector.py.pre-v7` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/index/selector.py"
)
BACKUP = TARGET + ".pre-v7"
MARK = "[PATCH spec v7]"

# --- 1. the helper, after the imports ---------------------------------------

HELPER_OLD = "from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings\n"

HELPER_NEW = '''from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings


# [PATCH spec v7] query rows one request hands a decode-shaped read - its draft block, or one token
def draft_rows() -> int:
    """Return how many query rows a request contributes to one decode-shaped read.

    A plain decode step reads one row a request. Under speculative decoding the same read
    serves a verify block, which carries one row per drafted token, so a captured graph's
    listing buffers must hold that many rows for a request.

    Returns:
        The draft block's token count, or 1 when the server does not speculate.
    """
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1
    return max(1, int(spec.speculative_num_draft_tokens or 1))
'''

# --- 2. the buffer sizing ---------------------------------------------------

SIZING_OLD = '''        # A check reads the listing a captured decode step wrote, so it lives in buffers sized once. A decode step's
        # few rows are listed exactly, which is faster for them than a superset
        self.decode_top_k = TopK(top_p, request_count, device, exact=True)
'''

SIZING_NEW = '''        # A check reads the listing a captured decode step wrote, so it lives in buffers sized once. A decode step's
        # few rows are listed exactly, which is faster for them than a superset. A verify block hands this read one
        # row per drafted token, so the buffers hold a whole block for every request the table can hold.
        self.decode_top_k = TopK(top_p, request_count * draft_rows(), device, exact=True)
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"
    pairs = [("draft_rows helper", HELPER_OLD, HELPER_NEW), ("decode listing buffers", SIZING_OLD, SIZING_NEW)]
    patched = source
    failures: list[str] = []
    for name, old, new in pairs:
        old_nl = old.replace("\n", newline)
        count = patched.count(old_nl)
        if count != 1:
            failures.append(f"{name}: {count} matches")
            continue
        patched = patched.replace(old_nl, new.replace("\n", newline))
        print(f"ok: {name}")
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    shutil.copyfile(TARGET, BACKUP)
    print(f"backup: {BACKUP}")
    with open(TARGET, "w", newline="") as handle:
        handle.write(patched)
    print(f"patched: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
