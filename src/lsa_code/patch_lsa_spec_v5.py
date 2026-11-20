#!/usr/bin/env python3
"""Patch 5 - drop the verify branch's maintenance fallback; the metadata hooks own maintenance.

Evidence this is the right shape (`logs/exp_graph.log`):

    Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=16
    File ".../lsa/sglang/models/qwen3_5/backend.py", line 270, in select
        _registry.maintain(forward_batch)
    File ".../lsa/sglang/index/registry.py", line 102, in maintain
        raise RuntimeError("LSA maintenance needs request ids on the forward batch")

The failing pass carries no request ids, so it is a graph pass - and it is not the pass
that `torch.cuda.is_current_stream_capturing()` reports (sglang warms a graph up before
capturing it), which is why a capture flag inside `select()` cannot guard it. The batch
does not need a guard at all: LSA already maintains through two metadata hooks, the eager
one (`init_forward_metadata`) and the out-of-graph one (`init_forward_metadata_out_graph`,
which skips while capturing and runs before a replay). Round 4's read no longer needs
`lsa_rows`, so the fallback has nothing left to protect.

Idempotent: refuses to patch twice, and writes `backend.py.pre-v5` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/"
    "qwen3_5/backend.py"
)
BACKUP = TARGET + ".pre-v5"
MARK = "[PATCH spec v5]"

OLD = '''            if getattr(forward_batch, "lsa_rows", None) is None and not (
                torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()
            ):
                from lsa.sglang.index import registry as _registry
                _registry.maintain(forward_batch)
            _rows = forward_batch.req_pool_indices.long()
'''

NEW = '''            # [PATCH spec v5] No maintenance here. LSA maintains through its two metadata hooks -
            # the eager one before a forward, and the out-of-graph one before a captured forward
            # replays - and a capture pass carries no request ids to maintain with, so a fallback
            # in this branch only fires on that pass and raises.
            _rows = forward_batch.req_pool_indices.long()
'''

OLD_EXTEND = '''        if _fm.is_extend():
            # [PATCH spec] a verify batch reaches here without the metadata hook
            # having run, so put the request rows on it and advance the index now.
            if getattr(forward_batch, "lsa_rows", None) is None:
                from lsa.sglang.index import registry as _registry
                _registry.maintain(forward_batch)
'''

NEW_EXTEND = '''        if _fm.is_extend():
            # [PATCH spec] a prefill reaches here without the metadata hook having run only on a
            # batch that carries no request ids, which is a graph pass and has nothing to advance.
            if getattr(forward_batch, "lsa_rows", None) is None and (
                getattr(forward_batch, "rids", None) is not None
            ):
                from lsa.sglang.index import registry as _registry
                _registry.maintain(forward_batch)
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"
    pairs = [("verify fallback removed", OLD, NEW), ("extend fallback guarded", OLD_EXTEND, NEW_EXTEND)]
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
