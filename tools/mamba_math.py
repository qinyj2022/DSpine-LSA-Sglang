#!/usr/bin/env python3
"""Account for the mamba/SSM cache footprint of this checkpoint.

用法：python mamba_math.py <config.json> <max_mamba_cache_size>
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    """Print the per-slot and total SSM/conv state footprint."""
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    text = cfg.get("text_config") or cfg

    print("--- linear_attention / ssm fields ---")
    for key in sorted(text):
        lowered = key.lower()
        if any(tag in lowered for tag in ("linear", "ssm", "mamba", "conv", "state")):
            print(f"{key}: {text[key]}")

    layer_types = text.get("layer_types") or []
    linear = sum(1 for kind in layer_types if "linear" in str(kind))
    full = len(layer_types) - linear
    print(f"\nlayers: {len(layer_types)}  linear={linear}  full_attn={full}")

    heads_v = text.get("linear_num_value_heads") or text.get("num_value_heads")
    heads_k = text.get("linear_num_key_heads") or text.get("num_key_heads")
    dim_k = text.get("linear_key_head_dim") or text.get("key_head_dim")
    dim_v = text.get("linear_value_head_dim") or text.get("value_head_dim")
    kernel = text.get("linear_conv_kernel_dim") or text.get("conv_kernel_dim")
    hidden = text.get("hidden_size")
    print(f"heads_v={heads_v} heads_k={heads_k} dim_k={dim_k} dim_v={dim_v} conv_k={kernel} hidden={hidden}")

    if heads_v and dim_k and dim_v and kernel:
        # Recurrent SSM state per layer: value heads x (key dim x value dim), fp32.
        ssm_bytes = heads_v * dim_k * dim_v * 4
        # Conv state per layer: depthwise kernel over (2*heads_k*dim_k + hidden) channels.
        conv_channels = 2 * heads_k * dim_k + hidden
        conv_bytes = conv_channels * kernel * 4
        print(f"\nper layer: ssm={ssm_bytes / 2**20:.2f} MiB  conv={conv_bytes / 2**20:.2f} MiB")
        print(f"x{linear} linear layers = ssm {ssm_bytes * linear / 2**30:.2f} GiB, "
              f"conv {conv_bytes * linear / 2**30:.3f} GiB per slot")
        print(f"x{slots} slots = ssm {ssm_bytes * linear * slots / 2**30:.2f} GiB, "
              f"conv {conv_bytes * linear * slots / 2**30:.3f} GiB")
        # What one KV token costs, for scale.
        kv_heads = text.get("num_key_value_heads")
        head_dim = text.get("head_dim")
        if kv_heads and head_dim and full:
            per_token = full * kv_heads * head_dim * 2 * 2  # K and V, bf16
            print(f"\nKV for {full} full-attn layers: {per_token / 1024:.0f} KiB/token")
            print(f"the {ssm_bytes * linear * slots / 2**30:.2f} GiB of SSM state buys "
                  f"{ssm_bytes * linear * slots / per_token / 1024:.0f}K tokens worth of KV")
    return 0


if __name__ == "__main__":
    sys.exit(main())
