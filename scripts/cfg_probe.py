#!/usr/bin/env python3
"""Print the facts needed to size a 200K-context boot.

用法：python cfg_probe.py <config.json>
"""

from __future__ import annotations

import json
import sys

KEYS = (
    "max_position_embeddings",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_size",
    "rope_scaling",
    "rope_theta",
    "layer_types",
)


def main() -> int:
    """Dump the geometry fields and the layer-type histogram."""
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key in KEYS:
        value = cfg.get(key)
        if key == "layer_types" and isinstance(value, list):
            from collections import Counter

            print(f"{key}: {dict(Counter(value))}")
        else:
            print(f"{key}: {value}")
    text = cfg.get("text_config") or {}
    if text:
        print("--- text_config ---")
        for key in KEYS:
            if key in text:
                print(f"text.{key}: {text[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
