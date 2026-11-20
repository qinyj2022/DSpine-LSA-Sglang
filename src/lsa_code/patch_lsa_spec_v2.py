#!/usr/bin/env python3
"""Patch LSA's Qwen3.5 backend for a speculative (verify) batch - the second round.

Round 1 (the `[PATCH spec]` markers) got a verify batch through the extend-shaped
selection but left the row count wrong: `triton_sparse_read` built a view of
`lists.shape[0]` rows over a query of `T` rows, and a verify batch has `k` query
rows per request.

This round
  1. derives the per-request token count from the batch's own new-token count
     (`out_cache_loc`, one KV slot per new token) instead of `positions.numel()`,
     so the selection has exactly one row per query row;
  2. prints what the speculative path actually sees, behind `LSA_SPEC_DEBUG=1`,
     including where each request's index has got to.

Idempotent: refuses to patch twice, and writes `backend.py.pre-v2` once.
Run with the sglang python:
    /mnt/nvme1n1/sglang522/bin/python patch_lsa_spec_v2.py
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/"
    "qwen3_5/backend.py"
)
BACKUP = TARGET + ".pre-v2"
MARK = "[PATCH spec v2]"

# --- 1. the module-level diagnostic helpers, after the INDEXERS constant -------

HELPERS_OLD = 'INDEXERS: str = "lsa_lsa_indexers"\n'

HELPERS_NEW = '''INDEXERS: str = "lsa_lsa_indexers"

# [PATCH spec v2] one line of the speculative path's diagnostics, printed when LSA_SPEC_DEBUG=1
def dbg(message: str) -> None:
    """Print one diagnostic line of the speculative path, when `LSA_SPEC_DEBUG` asks for it.

    Args:
        message: What to print.
    """
    import os

    if os.environ.get("LSA_SPEC_DEBUG"):
        print(f"[LSA-DBG2] {message}", flush=True)


# [PATCH spec v2] where each request's index has got to, for the diagnostics - never raises
def cache_ends(index: Any, rows: Any) -> list[int]:
    """Return the last key each request row holds in the index, or nothing when it cannot be read.

    Args:
        index: The layer's `LayerIndex`.
        rows: The request table rows.
    """
    import contextlib

    with contextlib.suppress(Exception):
        ends = index.selector.cache.end
        return [int(ends[row]) for row in rows]
    return []
'''

# --- 2. the verify branch of `select()` --------------------------------------

VERIFY_OLD = '''            if _fm.is_target_verify() or _fm.is_draft_extend_v2():
                # [PATCH spec] synthesize the extend lengths a verify batch omits
                _bs = len(forward_batch.lsa_rows)
                _k = int(forward_batch.positions.numel()) // max(1, _bs)
                _lens = [_k] * _bs
                _prefix = [int(s) - _k for s in forward_batch.seq_lens_cpu.tolist()[:_bs]]
'''

VERIFY_NEW = '''            if _fm.is_target_verify() or _fm.is_draft_extend_v2():
                # [PATCH spec v2] synthesize the extend lengths a verify batch omits.
                # A selection needs one row per *query row*, and a verify batch carries k query
                # rows per request, so the per-request token count comes from the batch's own
                # new-token count - one KV slot per new token - and never from `positions`, which
                # a verify batch may carry for fewer rows than the queries it is asked about.
                _bs = len(forward_batch.lsa_rows)
                _tokens = int(slots.numel())
                if _bs and _tokens % _bs == 0:
                    _k = _tokens // _bs
                else:
                    _k = max(1, int(forward_batch.positions.numel()) // max(1, _bs))
                _lens = [_k] * _bs
                _prefix = [int(s) - _k for s in forward_batch.seq_lens_cpu.tolist()[:_bs]]
                dbg(
                    "verify mode=%s bs=%s q=%s key=%s positions=%s(%s) slots=%s seq_lens=%s"
                    " lens=%s prefix=%s rows=%s ends=%s"
                    % (
                        _fm,
                        _bs,
                        tuple(q.shape),
                        tuple(k.shape),
                        int(forward_batch.positions.numel()),
                        forward_batch.positions.tolist()[:8],
                        _tokens,
                        [int(s) for s in forward_batch.seq_lens_cpu.tolist()],
                        list(_lens),
                        list(_prefix),
                        list(forward_batch.lsa_rows),
                        cache_ends(index, forward_batch.lsa_rows),
                    )
                )
'''

# --- 3. keep the selection, to report its shape ------------------------------

EXTEND_OLD = '''            return index.extend(
                inputs,
                positions,
                slots,
                forward_batch.lsa_rows,
                _lens,
                _prefix,
            )
'''

EXTEND_NEW = '''            _lists = index.extend(
                inputs,
                positions,
                slots,
                forward_batch.lsa_rows,
                _lens,
                _prefix,
            )
            if _fm.is_target_verify() or _fm.is_draft_extend_v2():
                dbg("verify selected lists=%s of %s query rows" % (tuple(_lists.shape), int(q.numel())))
            return _lists
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"
    pairs = [
        ("helpers", HELPERS_OLD, HELPERS_NEW),
        ("verify branch", VERIFY_OLD, VERIFY_NEW),
        ("extend call", EXTEND_OLD, EXTEND_NEW),
    ]
    patched = source
    failures: list[str] = []
    for name, old, new in pairs:
        old_nl, new_nl = old.replace("\n", newline), new.replace("\n", newline)
        count = patched.count(old_nl)
        if count != 1:
            failures.append(f"{name}: {count} matches")
            continue
        patched = patched.replace(old_nl, new_nl)
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
