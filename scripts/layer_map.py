#!/usr/bin/env python3
"""Which target layers does this DSpine draft tap, and are they recurrent ones?

用法：python layer_map.py <draft config.json> <target config.json>
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    """Cross-check target_layer_ids against the target's layer_types."""
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        draft = json.load(handle)
    dflash = draft.get("dflash_config") or {}
    layer_ids = dflash.get("target_layer_ids") or []
    feature_layers = dflash.get("selected_prefix_feature_layers") or []
    print("draft target_layer_ids:", layer_ids)
    print("draft selected_prefix_feature_layers:", feature_layers)

    if len(sys.argv) > 2:
        with open(sys.argv[2], "r", encoding="utf-8") as handle:
            target = json.load(handle)
        text = target.get("text_config") or target
        types = text.get("layer_types") or []
        print(f"\ntarget has {len(types)} layers")
        for name, ids in (("target_layer_ids", layer_ids),
                          ("feature_layers", feature_layers)):
            kinds = []
            for index in ids:
                if isinstance(index, int) and 0 <= index < len(types):
                    kinds.append(f"{index}:{types[index]}")
                else:
                    kinds.append(f"{index}:OOR")
            print(f"{name} -> {kinds}")
        recurrent = [index for index in layer_ids
                     if isinstance(index, int) and 0 <= index < len(types)
                     and "linear" in str(types[index])]
        print(f"\nrecurrent (linear) taps: {recurrent if recurrent else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
