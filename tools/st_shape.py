#!/usr/bin/env python3
"""Dump a single-file safetensors checkpoint skeleton without loading tensors.

用法：python st_shape.py <model.safetensors>
"""

from __future__ import annotations

import collections
import json
import struct
import sys


def main() -> int:
    """Print per-layer module inventory plus parameter totals by name prefix."""
    path = sys.argv[1]
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)

    groups: dict[str, list[tuple[str, list[int]]]] = collections.OrderedDict()
    total_params = 0
    for key, info in header.items():
        shape = info.get("shape") or []
        count = 1
        for dim in shape:
            count *= dim
        total_params += count
        parts = key.split(".")
        if len(parts) > 2 and parts[1] == "layers":
            group = f"layers.{parts[2]}"
            rest = ".".join(parts[3:])
        else:
            group = "top"
            rest = key
        groups.setdefault(group, []).append((rest, shape))

    print(f"tensors: {len(header)}   params: {total_params/1e9:.3f} B "
          f"({total_params*2/2**30:.2f} GiB bf16)")
    for group, entries in groups.items():
        if group.startswith("layers.") and group not in ("layers.0", "layers.1"):
            continue
        print(f"\n[{group}] ({len(entries)})")
        for rest, shape in sorted(entries):
            print(f"   {rest:52s} {shape}")
    layer_keys = [g for g in groups if g.startswith("layers.")]
    print(f"\nlayer groups: {layer_keys}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
