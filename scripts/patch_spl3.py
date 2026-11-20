#!/usr/bin/env python3
"""Patch spl3 - B'-chain: append prompt-lookup tokens to the drafter's chain, one verify covers all.

Round 12's SPL replaced the chain's tail, which measured neutral (accept 3.913 -> 3.927): it can
only swap tokens, never add. This patch appends instead, so the verify width grows by `extra` and
every model token is kept:

    chain   = [model's m tokens] ++ [lookup continuation of length e]     (width m + e)
    bonus   = 1                                                            (verify width m+e+1)

`build_chain_tree` infers the chain length from the tensor it is handed, so a longer chain needs
no tree encoding work - only `num_draft_tokens`, the chain-path gate and the (preallocated) mask
buffer have to follow the new width.

Knobs:
    SPL_EXTEND=e     tokens to append (0 = off, keeps the old replace behaviour)
    SPL_MAXL=3       n-gram length used to find the earlier occurrence
    SPL_WINDOW=4096  tokens of context scanned for the match

Idempotent: refuses to patch twice, and writes `eagle_worker_common.py.pre-spl3` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/sglang/srt/speculative/"
    "eagle_worker_common.py"
)
BACKUP = TARGET + ".pre-spl3"
MARK = "[PATCH spl3]"

HELPER = '''# [PATCH spl3] append lookup continuations to a linear draft chain - the B'-chain fusion
def _spl_append_chain(batch, draft_tokens: torch.Tensor, num_steps: int, extra: int):
    """Extend each request's draft chain with prompt-lookup tokens taken from its own context.

    Searches [recent context + the drafted chain] for the last earlier occurrence of its tail and
    appends what followed it, so the continuation stays coherent with the tokens before it. A
    request with no earlier occurrence keeps the drafter's chain alone.

    Args:
        batch: The schedule batch, whose `reqs` carry token ids.
        draft_tokens: The drafter's chain [bs, num_steps].
        num_steps: Columns of that chain.
        extra: Tokens to append at most.

    Returns:
        (chain, appended) - the chain widened to num_steps + appended columns, all rows rectangular.
    """
    import os

    reqs = getattr(batch, "reqs", None)
    if not reqs or extra <= 0:
        return draft_tokens, 0
    length = max(1, int(os.environ.get("SPL_MAXL", "3")))
    window = max(64, int(os.environ.get("SPL_WINDOW", "4096")))

    rows = draft_tokens.tolist()
    appended = 0
    for index, req in enumerate(reqs[: len(rows)]):
        history = list(req.origin_input_ids) + list(req.output_ids)
        seq = history[-window:] + [int(token) for token in rows[index]]
        if len(seq) < length + 2:
            continue
        grams: dict[tuple[int, ...], int] = {}
        for at in range(len(seq) - length - 1):
            grams[tuple(seq[at : at + length])] = at
        found = grams.get(tuple(seq[-length:]))
        if found is None:
            continue
        start = found + length
        continuation = seq[start : start + extra]
        if not continuation:
            continue
        rows[index].extend(int(token) for token in continuation)
        appended = max(appended, len(continuation))

    if not appended:
        return draft_tokens, 0
    width = max(len(row) for row in rows)
    for row in rows:
        row.extend([row[-1]] * (width - len(row)))
    chain = torch.tensor(rows, dtype=draft_tokens.dtype, device=draft_tokens.device)
    return chain, width - num_steps


'''

CALL_OLD = '''    # [PATCH spl] serial drafts, one verify: prompt lookup continues the drafter's chain
    if topk == 1:
        draft_tokens = _spl_extend_chain(batch, draft_tokens, num_steps)
'''

CALL_NEW = '''    # [PATCH spl3] B'-chain: append lookup continuations to the drafter's chain, then verify once
    _spl_extra = 0
    if topk == 1:
        import os as _os

        _want = int(_os.environ.get("SPL_EXTEND", "0") or 0)
        if _want > 0:
            draft_tokens, _spl_extra = _spl_append_chain(batch, draft_tokens, num_steps, _want)
            if _spl_extra:
                num_draft_tokens = num_draft_tokens + _spl_extra
                if _os.environ.get("SPL_DEBUG"):
                    print(
                        "[SPL3] chain widened to %d (extra %d), num_draft_tokens=%d"
                        % (draft_tokens.shape[1], _spl_extra, num_draft_tokens),
                        flush=True,
                    )
        else:
            draft_tokens = _spl_extend_chain(batch, draft_tokens, num_steps)
'''

MASK_OLD = '''        tree_mask_buf = verify_mask.buffer if verify_mask.fits(bs) else None
'''

MASK_NEW = '''        # [PATCH spl3] a widened chain no longer fits the buffer sized for the configured width
        tree_mask_buf = (
            verify_mask.buffer if (verify_mask.fits(bs) and not _spl_extra) else None
        )
'''

GATE_OLD = '''        and num_draft_tokens == num_steps + 1
        and draft_tokens.shape == (bs, num_steps)
'''

GATE_NEW = '''        and num_draft_tokens == num_steps + 1 + _spl_extra
        and draft_tokens.shape == (bs, num_steps + _spl_extra)
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"
    pairs = [
        ("append helper", "def build_eagle_verify_input(\n", HELPER + "def build_eagle_verify_input(\n"),
        ("call site", CALL_OLD, CALL_NEW),
        ("mask buffer", MASK_OLD, MASK_NEW),
        ("chain gate", GATE_OLD, GATE_NEW),
    ]
    patched = source
    failures: list[str] = []
    for name, old, new in pairs:
        old_nl, new_nl = old.replace("\n", newline), new.replace("\n", newline)
        count = patched.count(old_nl)
        if count != 1:
            failures.append(f"{name}: {count} matches")
            continue
        patched = patched.replace(old_nl, new_nl, 1)
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
