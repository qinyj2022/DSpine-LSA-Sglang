#!/usr/bin/env python3
"""Patch LSA's Qwen3.5 backend for a speculative (verify) batch - the third round.

Round 2 measured the failure and changed the diagnosis.  A verify block is
`k` query rows per request over KV that is *provisional*: the target model writes
all `k` drafted tokens into the pool, the sampler accepts a prefix of them, and
the rejected tail is rewritten by the next block.  LSA's `extend` appends the
whole block to the index and advances the request's `end` past it, so on the next
block `end` sits ahead of the block's own first position:

    block 1: seq_lens=[138] positions 122..137  index end=122  -> 16 of 16 rows
    block 2: seq_lens=[140] positions 124..139  index end=138  ->  2 of 16 rows
                                                 (14 rows dropped by cache.end, not by `positions`)

The fix is not a row-count fudge but a semantic one: a verify block must never
enter the index.  Only its accepted prefix is real, and that prefix reaches the
index by itself through the ordinary maintenance path (`advance` reads the keys
back out of the KV pool at the accepted positions), which is the design the
handoff already called for.

Reading without indexing is sound because the read is position-based anyway: the
loader resolves every bucket member and every window position through the
request's slot table, and the dense tail is a full `window_size` (512) positions
before each query, so a block of 16 drafted tokens is inside it.  The buckets
that route the read only cover closed cluster blocks, all of which are committed.

Idempotent: refuses to patch twice, and writes `backend.py.pre-v3` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/"
    "qwen3_5/backend.py"
)
BACKUP = TARGET + ".pre-v3"
MARK = "[PATCH spec v3]"

OLD = '''        if _fm.is_extend() or _fm.is_target_verify() or _fm.is_draft_extend_v2():
            # [PATCH spec] a verify batch reaches here without the metadata hook
            # having run, so put the request rows on it and advance the index now.
            if getattr(forward_batch, "lsa_rows", None) is None:
                from lsa.sglang.index import registry as _registry
                _registry.maintain(forward_batch)
            if _fm.is_target_verify() or _fm.is_draft_extend_v2():
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
            else:
                _lens = forward_batch.extend_seq_lens_cpu
                _prefix = forward_batch.extend_prefix_lens_cpu
            _lists = index.extend(
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

NEW = '''        if _fm.is_target_verify() or _fm.is_draft_extend_v2():
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
        if _fm.is_extend():
            # [PATCH spec] a verify batch reaches here without the metadata hook
            # having run, so put the request rows on it and advance the index now.
            if getattr(forward_batch, "lsa_rows", None) is None:
                from lsa.sglang.index import registry as _registry
                _registry.maintain(forward_batch)
            return index.extend(
                inputs,
                positions,
                slots,
                forward_batch.lsa_rows,
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_prefix_lens_cpu,
            )
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"
    old, new = OLD.replace("\n", newline), NEW.replace("\n", newline)
    count = source.count(old)
    if count != 1:
        print(f"FAILED: {count} matches for the select() block")
        return 1
    shutil.copyfile(TARGET, BACKUP)
    print(f"backup: {BACKUP}")
    with open(TARGET, "w", newline="") as handle:
        handle.write(source.replace(old, new))
    print(f"patched: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
