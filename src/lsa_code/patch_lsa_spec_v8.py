#!/usr/bin/env python3
"""Patch 8 (diagnostic) - time LSA's selection and its paged read, per forward, on the GPU clock.

Two cost hypotheses are already refuted by measurement: the bucket scan (step time is flat
from 16k, 2000 buckets, to 120k, 14870 buckets) and the sparse read's width (top_k 4096 ->
1024, list 4612 -> 1540 slots, did not move step time). So the ~29 ms a step costs once the
context passes the dense window has to be attributed by measurement, not by inference.

This wraps the two LSA regions of `select_and_read` in CUDA events, gated by `LSA_TIME=1`,
and prints their running means every 100 forwards:

    [LSA-TIME] n=100 select=3.21 ms read=18.44 ms (per forward, GPU clock)

Per-call synchronisation serialises the stream, so use the numbers as shares, not as a
throughput score, and run it on the eager (no-graph) configuration.

Diagnostic only: inert without the env var, and to be removed once attributed.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/"
    "qwen3_5/backend.py"
)
BACKUP = TARGET + ".pre-v8"
MARK = "[PATCH spec v8]"

HELPER_OLD = "# [PATCH spec v2] one line of the speculative path's diagnostics, printed when LSA_SPEC_DEBUG=1\n"

HELPER_NEW = '''# [PATCH spec v8] GPU-clock attribution of the selection and the read, when LSA_TIME=1
LSA_TIME_TOTALS: dict[str, float] = {"select": 0.0, "read": 0.0}
LSA_TIME_COUNT: list[int] = [0]


def timed(label: str, call: Any) -> Any:
    """Run one region under CUDA events and print the running mean every hundred forwards.

    Args:
        label: "select" or "read".
        call: A zero-argument callable running the region.

    Returns:
        Whatever the region returned.
    """
    import os

    if not os.environ.get("LSA_TIME"):
        return call()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = call()
    end.record()
    end.synchronize()
    LSA_TIME_TOTALS[label] += start.elapsed_time(end)
    if (LSA_TIME_TOTALS["select"] or LSA_TIME_TOTALS["read"]) != 0.0:
        # Both regions land in the same forward, so the read is what marks a forward complete
        pass
    if label == "read":
        LSA_TIME_COUNT[0] += 1
        if LSA_TIME_COUNT[0] % 100 == 0:
            count = LSA_TIME_COUNT[0]
            print(
                "[LSA-TIME] n=%d select=%.2f ms read=%.2f ms (per forward, GPU clock)"
                % (count, LSA_TIME_TOTALS["select"] / count, LSA_TIME_TOTALS["read"] / count),
                flush=True,
            )
            LSA_TIME_TOTALS["select"] = 0.0
            LSA_TIME_TOTALS["read"] = 0.0
            LSA_TIME_COUNT[0] = 0
    return out


# [PATCH spec v2] one line of the speculative path's diagnostics, printed when LSA_SPEC_DEBUG=1
'''

CALL_OLD = '''    if shard.size == 1:
        return sparse_read(self, q, layer, indexer.select(q, k, forward_batch))
'''

CALL_NEW = '''    if shard.size == 1:
        # [PATCH spec v8] the two LSA regions, on the GPU clock when LSA_TIME=1
        lists = timed("select", lambda: indexer.select(q, k, forward_batch))
        return timed("read", lambda: sparse_read(self, q, layer, lists))
'''


def main() -> int:
    with open(TARGET, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {TARGET}")
        return 0
    newline = "\r\n" if "\r\n" in source else "\n"
    pairs = [("timer helper", HELPER_OLD, HELPER_NEW), ("select/read timing", CALL_OLD, CALL_NEW)]
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
