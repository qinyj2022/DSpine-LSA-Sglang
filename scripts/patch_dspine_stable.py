#!/usr/bin/env python3
"""Patch DSpine - degrade instead of killing the scheduler when a sampled round's chain is missing.

Reproduced twice on the live server (`logs/alive.log`, `logs/mtp_live.log` predecessor):

    DSPINE: first decode round with a sampled request
    RuntimeError: DSPINE sampled verify reached without this round's sampled draft chain
      at sglang_dspine/runtime/worker.py:919 _accept_sampled_round
    -> SIGQUIT, whole scheduler process group torn down

Any client that sends `temperature > 0` (a browser, an SDK with defaults) hits it, so the service
died within a minute of serving such a request.

`sampler.stage_sampling_params()` opens a round with `_round_sampled = False`; only
`sampler.sample(bs)` sets it True, and some draft paths do not reach it. DSpine's own comment
calls that case out ("a draft graph replayed by another runner class"). The verification seam is
the plugin's AROUND hook, which still holds `original` - the stock DFlash sampled verify - so the
right behaviour is to fall back to it for that round instead of raising.

What this costs: for a round whose sampled chain is missing, acceptance follows stock DFlash
sampling verification rather than DSpine's own q-based rejection sampling. The tokens stay valid
and the service stays up; the fast path is untouched whenever the chain did run.

Idempotent: refuses to patch twice, and writes `plugin.py.pre-stable` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = "/mnt/sda/qyj1044/dspine_code/DSpine-main/sglang_dspine/plugin.py"
BACKUP = TARGET + ".pre-stable"
MARK = "[PATCH dspine-stable]"

OLD = '''def around_dflash_sampling_verify(original, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec

    if is_dspine(get_spec().speculative_algorithm or ""):
        from sglang_dspine.runtime.worker import accept_dspine_sampled_round

        return accept_dspine_sampled_round(*args, **kwargs)
    return original(*args, **kwargs)
'''

NEW = '''# [PATCH dspine-stable] count of rounds that had to fall back, for the log line below
_DSPINE_FALLBACK_ROUNDS = [0]


def around_dflash_sampling_verify(original, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec

    if is_dspine(get_spec().speculative_algorithm or ""):
        from sglang_dspine.runtime import worker as _dspine_worker
        from sglang_dspine.runtime.worker import accept_dspine_sampled_round

        # [PATCH dspine-stable] A round's sampled chain is produced by `sampler.sample(bs)`, and
        # some draft paths do not reach it. DSpine raises there, which tears down the whole
        # scheduler and takes every in-flight request with it. Fall back to the stock DFlash
        # sampled verification for that round instead - valid tokens, service stays up.
        holder = getattr(_dspine_worker, "_active_worker", None)
        sampler = getattr(holder, "_dspine_sampler", None) if holder is not None else None
        if sampler is not None and not getattr(sampler, "round_sampled", False):
            _DSPINE_FALLBACK_ROUNDS[0] += 1
            if _DSPINE_FALLBACK_ROUNDS[0] == 1:
                print(
                    "[PATCH dspine-stable] sampled round without its sampled draft chain - "
                    "verifying with the stock DFlash sampler instead of raising",
                    flush=True,
                )
            return original(*args, **kwargs)
        return accept_dspine_sampled_round(*args, **kwargs)
    return original(*args, **kwargs)
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
