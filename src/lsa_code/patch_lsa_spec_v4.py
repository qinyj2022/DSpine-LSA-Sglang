#!/usr/bin/env python3
"""Patch 4 - read a speculative verify block through the decode path, so its CUDA graph can be captured.

Why
---
Round 3 (`[PATCH spec v3]`) served a verify batch by a per-request host loop over
`selector.read_segment`, which is correct but not capturable: it needs `lsa_rows`
(a Python list), `seq_lens_cpu.tolist()` and one kernel launch per request, and the
maintenance fallback it relies on raises on a capture batch that carries no rids:

    Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=16
    RuntimeError: LSA maintenance needs request ids on the forward batch

But a verify block is semantically a decode step over a *committed* index: every one
of its `k` queries reads the request's committed prefix, and the read's dense
`window_size` tail already covers the block's own earlier tokens. So it can use
`decode`'s read - device-side, fixed-shape, table row per query - which is the very
path LSA already runs inside a captured decode graph when it serves alone.

What changes
------------
1. `select()`: the verify branch becomes

       rows = req_pool_indices.long().repeat_interleave(k)
       return selector.decode(index_queries, positions, rows)

   `k` query rows per request share the request's table row, so `decode_flat` reads
   `closed_keys_device[row]` (the committed bucket count) for all of them and folds the
   per-query window/sink tail in by position. Nothing touches the host, and the
   maintenance fallback is skipped while a graph is being captured or compiled - the
   out-of-graph metadata hook has already advanced the index before a replay.
2. `select_and_read()`: the capture refusal now applies to a real prefill only
   (`is_extend_without_speculative()`), not to a verify batch.

Idempotent: refuses to patch twice, and writes `backend.py.pre-v4` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/"
    "qwen3_5/backend.py"
)
BACKUP = TARGET + ".pre-v4"
MARK = "[PATCH spec v4]"

VERIFY_OLD = '''        if _fm.is_target_verify() or _fm.is_draft_extend_v2():
            # [PATCH spec v3] a verify batch's keys are speculative: the block is attended as it
            # stands - the read folds in a dense `window_size` tail and the sinks by position - but
            # it must not enter the index, because only its accepted prefix ever becomes real. The
            # index stays at the committed prefix, which this batch's own maintenance brought it
            # to, and the accepted positions enter through the next forward's maintenance, which
            # reads their keys back out of the pool.
            if getattr(forward_batch, "lsa_rows", None) is None:
                from lsa.sglang.index import registry as _registry
                _registry.maintain(forward_batch)
            _bs = len(forward_batch.lsa_rows)
            _tokens = int(slots.numel())
            if _bs and _tokens % _bs == 0:
                _k = _tokens // _bs
            else:
                _k = max(1, int(forward_batch.positions.numel()) // max(1, _bs))
            dbg(
                "verify mode=%s bs=%s q=%s key=%s positions=%s slots=%s seq_lens=%s k=%s"
                " rows=%s ends=%s"
                % (
                    _fm,
                    _bs,
                    tuple(q.shape),
                    tuple(k.shape),
                    int(forward_batch.positions.numel()),
                    _tokens,
                    [int(s) for s in forward_batch.seq_lens_cpu.tolist()],
                    _k,
                    list(forward_batch.lsa_rows),
                    cache_ends(index, forward_batch.lsa_rows),
                )
            )
            index_queries = index.write(inputs, slots)
            lists = []
            _start = 0
            for _row, _count in zip(forward_batch.lsa_rows, [_k] * _bs, strict=True):
                _stop = _start + _count
                lists.append(
                    index.selector.read_segment(
                        _row,
                        index_queries[_start:_stop],
                        positions[_start:_stop],
                    )
                )
                _start = _stop
            if _start != _tokens:
                raise RuntimeError(
                    f"a verify batch of {_bs} requests covers {_start} of its {_tokens} queries under LSA"
                )
            _lists = torch.cat(lists)
            dbg("verify lists=%s" % (tuple(_lists.shape),))
            return _lists
'''

VERIFY_NEW = '''        if _fm.is_target_verify() or _fm.is_draft_extend_v2():
            # [PATCH spec v4] a verify block is read exactly as a decode step is: one row per query
            # over the *committed* index. Its keys stay speculative and never enter the index - only
            # the accepted prefix does, through the next forward's out-of-graph maintenance - and
            # because the read is `decode`'s it is device-side and fixed-shape, so the target-verify
            # CUDA graph can be captured over it.
            if getattr(forward_batch, "lsa_rows", None) is None and not (
                torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()
            ):
                from lsa.sglang.index import registry as _registry
                _registry.maintain(forward_batch)
            _rows = forward_batch.req_pool_indices.long()
            _bs = int(_rows.numel())
            _tokens = int(positions.numel())
            if not _bs or _tokens % _bs:
                raise RuntimeError(
                    f"a verify batch of {_bs} requests carries {_tokens} queries under LSA"
                )
            _k = _tokens // _bs
            index_queries = index.write(inputs, slots)
            # Each query keeps its request's table row, so all k of them read the committed prefix
            # of that request, and the read's own dense window covers the block they belong to.
            dbg(
                "verify q=%s key=%s bs=%s tokens=%s k=%s"
                % (tuple(q.shape), tuple(k.shape), _bs, _tokens, _k)
            )
            return index.selector.decode(index_queries, positions, _rows.repeat_interleave(_k))
'''

GUARD_OLD = '''    mode = forward_batch.forward_mode
    if mode.is_extend() and (torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()):
        raise RuntimeError("LSA selects a prefill on the host - serve with --disable-prefill-cuda-graph")
'''

GUARD_NEW = '''    mode = forward_batch.forward_mode
    # [PATCH spec v4] only a real prefill selects on the host - a verify batch reads device-side
    if mode.is_extend_without_speculative() and (
        torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()
    ):
        raise RuntimeError("LSA selects a prefill on the host - serve with --disable-prefill-cuda-graph")
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"
    pairs = [
        ("verify branch -> decode read", VERIFY_OLD, VERIFY_NEW),
        ("capture guard -> prefill only", GUARD_OLD, GUARD_NEW),
    ]
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
