#!/usr/bin/env python3
"""Patch 9 - read a replay view's sequence lengths off the device when it carries no host copy.

NEXTN/MTP replay fails on LSA's maintenance hook (`logs/fp4_mtp.log`):

    File ".../sglang/srt/speculative/eagle_utils.py", line 624, in eagle_prepare_for_verify
        target_worker.model_runner.decode_cuda_graph_runner.load_batch(
    File ".../sglang/srt/model_executor/runner/decode_cuda_graph_runner.py", line 1575, in load_batch
        attn_backend.init_forward_metadata_out_graph(fb_view)
    File ".../lsa/sglang/models/qwen3_5/backend.py", line 489, in maintain_out_graph
        registry.maintain(kwargs.get("forward_batch") or args[1])
    File ".../lsa/sglang/index/registry.py", line 77, in ends_of
        seq_lens = forward_batch.seq_lens_cpu.tolist()[:request_count]
    AttributeError: 'NoneType' object has no attribute 'tolist'

`load_batch` builds the view a captured graph replays from device tensors only, so
`seq_lens_cpu` is None on it - DSpine's worker happened to hand one over, NEXTN does not.
The hook runs *outside* capture, where a device sync is legal, so the count can be read off
`forward_batch.seq_lens` instead.

Idempotent: refuses to patch twice, and writes `registry.py.pre-v9` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/index/registry.py"
BACKUP = TARGET + ".pre-v9"
MARK = "[PATCH spec v9]"

OLD = '''    seq_lens = forward_batch.seq_lens_cpu.tolist()[:request_count]
'''

NEW = '''    # [PATCH spec v9] a replay view built by `decode_cuda_graph_runner.load_batch` carries the
    # lengths on the device only; this runs outside capture, so reading them back is legal
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens
    seq_lens = seq_lens_cpu.tolist()[:request_count]
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
