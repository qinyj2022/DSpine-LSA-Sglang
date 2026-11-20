#!/usr/bin/env python3
"""Dump the fields that decide whether a draft checkpoint is a DSpine draft.

用法：python draft_probe.py <draft config.json>
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    """Print architecture, geometry and the DSpine bridge marker."""
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    text = cfg.get("text_config") or cfg
    print("architectures:", cfg.get("architectures"))
    print("model_type:", cfg.get("model_type"))
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads",
                "num_key_value_heads", "head_dim", "max_position_embeddings"):
        print(f"{key}: {text.get(key)}")
    dflash = cfg.get("dflash_config") or text.get("dflash_config") or {}
    print("--- dflash_config ---")
    for key, value in dflash.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key}: {value}")
        else:
            print(f"{key}: <{type(value).__name__} len={len(value) if hasattr(value, '__len__') else '?'}>")
    bridge = dflash.get("selected_prefix_code_bridge")
    print("\nis DSpine draft (bridge present):", bool(bridge))
    return 0


if __name__ == "__main__":
    sys.exit(main())
