#!/usr/bin/env python3
"""Patch 13 - make the prompt-lookup continuation follow the *drafted prefix*, not the raw context.

Measured with patch 12 (`SPL_KEEP=4`, FP4 + LSA + MTP steps=8):

    baseline (no SPL): mean 94.1 tok/s, accept 3.913
    SPL keep=4      : mean 85.2 tok/s, accept 3.558      <- worse

The continuation was matched against the real context alone, but it is written into the columns
that follow the drafter's own `keep` tokens. Those tokens were conditioned on the drafted
prefix, so a continuation that only follows the real context is incoherent with its predecessor
and gets rejected. Fix: search the *synthetic* sequence [recent context + drafted prefix], so the
continuation continues exactly what precedes it.

Idempotent: refuses to patch twice, and writes `eagle_worker_common.py.pre-spl2` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/sglang/srt/speculative/"
    "eagle_worker_common.py"
)
BACKUP = TARGET + ".pre-spl2"
MARK = "[PATCH spl2]"

OLD = '''        win = history[-window:]
        # n-gram -> last position, never including the final n-gram itself
        grams: dict[tuple[int, ...], int] = {}
        for at in range(len(win) - length - 1):
            grams[tuple(win[at : at + length])] = at
        suffix = tuple(win[-length:])
        found = grams.get(suffix)
        if found is None:
            continue
        start = found + length
        continuation = win[start : start + want]
        if not continuation:
            continue
'''

NEW = '''        # [PATCH spl2] search the synthetic sequence [recent context + drafted prefix], so the
        # continuation continues the tokens it is written after instead of the raw context
        seq = history[-window:] + [int(token) for token in rows[index][:keep]]
        # n-gram -> last position, never including the final n-gram itself
        grams: dict[tuple[int, ...], int] = {}
        for at in range(len(seq) - length - 1):
            grams[tuple(seq[at : at + length])] = at
        suffix = tuple(seq[-length:])
        found = grams.get(suffix)
        if found is None:
            continue
        start = found + length
        continuation = seq[start : start + want]
        if not continuation:
            continue
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
