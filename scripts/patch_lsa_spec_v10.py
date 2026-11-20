#!/usr/bin/env python3
"""Patch 10 - let LSA read an fp8 KV cache by dequantising on the fly, leaving the cache itself alone.

The model's attention keeps whatever cache it wants (`--kv-cache-dtype auto` on the FP8/NVFP4
checkpoint resolves to `float8_e4m3fn`), and only LSA's own view of the keys becomes bf16 -
the dtype its index maps were distilled on.

Two surgical edits:

1. `GroupedQueryIndexer.key_buffer()` no longer refuses a non-bf16 cache. It returns the keys
   dequantised to bf16 (`buffer * layer.k_scale_float`, the same descale sglang's triton
   backend uses per `triton_backend.py`). The pool is untouched - it is a copy for the index.

2. `KeySlot.write()` skips a non-bf16 pool. LSA's pre-write of the position-free key dims
   exists so the pool holds them before the index reads, and the attention's own `save()`
   runs first in `select_and_read`, so under a quantised cache the attention is the writer
   and LSA only reads. Writing a bf16 view back would silently corrupt the fp8 cache.

Idempotent: refuses to patch twice, and writes `backend.py.pre-v10` / `layer_index.py.pre-v10`.
"""

from __future__ import annotations

import shutil
import sys

BACKEND = (
    "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/models/"
    "qwen3_5/backend.py"
)
LAYERS = "/mnt/nvme1n1/sglang522/lib/python3.12/site-packages/lsa/sglang/index/layer_index.py"
MARK = "[PATCH spec v10]"

KEY_BUFFER_OLD = '''        buffer: torch.Tensor = self.backend.token_to_kv_pool.get_key_buffer(self.layer)
        if buffer.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"the key and query input reads a bf16 KV cache - this one is {buffer.dtype}, serve kv_cache_dtype auto"
            )
        return buffer.view(-1, self.geometry.kv_head_count, self.geometry.head_dim)
'''

KEY_BUFFER_NEW = '''        buffer: torch.Tensor = self.backend.token_to_kv_pool.get_key_buffer(self.layer)
        keys = buffer.view(-1, self.geometry.kv_head_count, self.geometry.head_dim)
        if buffer.dtype == torch.bfloat16:
            return keys
        # [PATCH spec v10] a quantised cache stays quantised - the attention keeps its own pool.
        # The index takes a bf16 copy of the keys, the dtype its maps were distilled on, descaled
        # the way sglang's triton backend descales a fp8 cache (`buffer * k_scale_float`).
        scale = getattr(self.layer, "k_scale_float", None)
        if scale is None:
            raise NotImplementedError(
                f"the key and query input reads a bf16 KV cache - this one is {buffer.dtype} "
                "and the layer carries no k_scale_float to descale it with"
            )
        return (keys.to(torch.float32) * scale.to(keys.device)).to(torch.bfloat16)
'''

WRITE_OLD = '''        buffer[rows, :, self.offset : self.offset + self.width] = key.view(len(rows), self.heads, self.width)
'''

WRITE_NEW = '''        # [PATCH spec v10] a quantised cache is the attention's to write, and this view of it is a
        # dequantised copy - writing back would either be lost or corrupt the stored fp8 keys
        if buffer.dtype != torch.bfloat16:
            return
        buffer[rows, :, self.offset : self.offset + self.width] = key.view(len(rows), self.heads, self.width)
'''


def patch(path: str, old: str, new: str, backup: str) -> bool:
    """Replace one literal block exactly once, keeping a backup."""
    with open(path, newline="") as handle:
        source = handle.read()
    if MARK in source:
        print(f"already patched: {path}")
        return True
    newline = "\r\n" if "\r\n" in source else "\n"
    old_nl = old.replace("\n", newline)
    count = source.count(old_nl)
    if count != 1:
        print(f"FAILED {path}: {count} matches")
        return False
    shutil.copyfile(path, backup)
    print(f"backup: {backup}")
    with open(path, "w", newline="") as handle:
        handle.write(source.replace(old_nl, new.replace("\n", newline)))
    print(f"patched: {path}")
    return True


def main() -> int:
    if not patch(BACKEND, KEY_BUFFER_OLD, KEY_BUFFER_NEW, BACKEND + ".pre-v10"):
        return 1
    if not patch(LAYERS, WRITE_OLD, WRITE_NEW, LAYERS + ".pre-v10"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
