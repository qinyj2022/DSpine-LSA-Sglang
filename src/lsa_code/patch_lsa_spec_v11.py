#!/usr/bin/env python3
"""Patch 11 - when the quantised cache carries no per-layer scale, treat the scale as 1.0.

`logs/fp4_lsa_fp8kv.log`:

    NotImplementedError: the key and query input reads a bf16 KV cache - this one is
    torch.float8_e4m3fn and the layer carries no k_scale_float to descale it with

`RadixAttention.__init__` does define `k_scale` / `k_scale_float` (radix_attention.py:136-139),
so the attribute exists but is None here - this server casts the cache to fp8 without a scale,
which is also why sglang's own triton backend takes its unscaled branch
(`if layer.k_scale is not None and layer.v_scale is not None:`). So the descale factor is 1.0.

Idempotent: refuses to patch twice, and writes `backend.py.pre-v11` once.
"""

from __future__ import annotations

import shutil
import sys

TARGET = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/"
    "qwen3_5/backend.py"
)
BACKUP = TARGET + ".pre-v11"
MARK = "[PATCH spec v11]"

OLD = '''        scale = getattr(self.layer, "k_scale_float", None)
        if scale is None:
            raise NotImplementedError(
                f"the key and query input reads a bf16 KV cache - this one is {buffer.dtype} "
                "and the layer carries no k_scale_float to descale it with"
            )
        return (keys.to(torch.float32) * scale.to(keys.device)).to(torch.bfloat16)
'''

NEW = '''        scale = getattr(self.layer, "k_scale_float", None)
        # [PATCH spec v11] a cache cast without a scale descales by 1.0 - sglang's own triton
        # backend takes the same unscaled branch when `layer.k_scale` is None
        keys = keys.to(torch.float32)
        if scale is not None:
            keys = keys * scale.to(keys.device)
        return keys.to(torch.bfloat16)
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
