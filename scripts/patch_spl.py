#!/usr/bin/env python3
"""Patch 12 - serial drafts, single verify: let prompt lookup fill the tail of a linear draft chain.

Design (option B, agreed): the primary drafter proposes a chain, then prompt lookup (sglang's
NGRAM idea, done inline) continues it from the real context, and *one* target verify covers the
whole chain. No second verify pass - on this GPU a step is dominated by streaming the weights
once, so a second verify would nearly double the step.

Where it goes: `build_eagle_verify_input` (speculative/eagle_worker_common.py) is the shared tail
of every eagle-style drafter - MTP/NEXTN, EAGLE, and DSPine all reach it - and for `topk == 1`
`draft_tokens` is exactly the `[bs, num_steps]` linear chain that `build_chain_tree` verifies.
Replacing tail columns leaves parents, tree mask, graph shapes and the accept machinery untouched.

Knobs (environment, so they can be swept without a rebuild):
    SPL_ENABLE=1        turn the extension on (default off: the patch is inert)
    SPL_KEEP=4          how many leading draft tokens to keep from the primary drafter
    SPL_MAXL=3          n-gram length used to find the earlier occurrence
    SPL_WINDOW=4096     tokens of context scanned for the match
    SPL_EVERY=1          extra periodic gate (kept for quick on/off A/B)

Idempotent: refuses to patch twice, and writes `eagle_worker_common.py.pre-spl` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/sglang/srt/speculative/"
    "eagle_worker_common.py"
)
BACKUP = TARGET + ".pre-spl"
MARK = "[PATCH spl]"

HELPER_ANCHOR = "def build_eagle_verify_input(\n"

HELPER = '''# [PATCH spl] prompt lookup fills a linear draft chain's tail, so one verify covers serial drafts
def _spl_extend_chain(batch, draft_tokens: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Continue a linear draft chain with prompt-lookup tokens taken from the real context.

    The primary drafter's leading `SPL_KEEP` tokens are kept; the remaining columns are filled
    with what followed the last earlier occurrence of the context's tail. A request with no
    earlier occurrence keeps the drafter's own tokens, so the extension is never worse than
    doing nothing but the cost of one dict build.

    Args:
        batch: The schedule batch, whose `reqs` carry each request's token ids.
        draft_tokens: The drafter's chain [bs, num_steps] int64, one row a request.
        num_steps: Columns of the chain, which is what the verify will cover.

    Returns:
        The chain, with tail columns possibly replaced - the same shape and dtype.
    """
    import os

    if not os.environ.get("SPL_ENABLE") or num_steps < 2:
        return draft_tokens
    reqs = getattr(batch, "reqs", None)
    if not reqs:
        return draft_tokens
    keep = max(0, min(num_steps, int(os.environ.get("SPL_KEEP", "4"))))
    if keep >= num_steps:
        return draft_tokens
    length = max(1, int(os.environ.get("SPL_MAXL", "3")))
    window = max(64, int(os.environ.get("SPL_WINDOW", "4096")))
    want = num_steps - keep

    rows = draft_tokens.tolist()
    extended = 0
    for index, req in enumerate(reqs[: len(rows)]):
        history = list(req.origin_input_ids) + list(req.output_ids)
        if len(history) < length + 2:
            continue
        win = history[-window:]
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
        for offset, token in enumerate(continuation):
            rows[index][keep + offset] = int(token)
        extended += 1

    if not extended:
        return draft_tokens
    return torch.tensor(rows, dtype=draft_tokens.dtype, device=draft_tokens.device)


'''

CALL_ANCHOR = '''    if batch.forward_mode.is_idle():
        return EagleVerifyInput.create_idle_input(
            topk,
            num_steps,
            num_draft_tokens,
            device,
        )
'''

CALL_NEW = '''    if batch.forward_mode.is_idle():
        return EagleVerifyInput.create_idle_input(
            topk,
            num_steps,
            num_draft_tokens,
            device,
        )

    # [PATCH spl] serial drafts, one verify: prompt lookup continues the drafter's chain
    if topk == 1:
        draft_tokens = _spl_extend_chain(batch, draft_tokens, num_steps)
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"

    if source.count(HELPER_ANCHOR.replace("\n", newline)) != 1:
        print("FAILED: build_eagle_verify_input anchor not unique")
        return 1
    if source.count(CALL_ANCHOR.replace("\n", newline)) != 1:
        print("FAILED: idle-input anchor not unique")
        return 1

    patched = source.replace(
        HELPER_ANCHOR.replace("\n", newline),
        HELPER.replace("\n", newline) + HELPER_ANCHOR.replace("\n", newline),
        1,
    )
    patched = patched.replace(
        CALL_ANCHOR.replace("\n", newline), CALL_NEW.replace("\n", newline), 1
    )

    shutil.copyfile(TARGET, BACKUP)
    print(f"backup: {BACKUP}")
    with open(TARGET, "w", newline="") as handle:
        handle.write(patched)
    print(f"patched: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
