#!/usr/bin/env python3
"""Does this target expose fill-in-the-middle / mask tokens a drafter could reuse?

并行草稿要求模型具备“一次填多个空”的能力；只有被 FIM 目标训练过的模型才自带这种 token。

用法：python fim_probe.py <model_dir>
"""

from __future__ import annotations

import json
import os
import sys

TAGS = ("fim", "mask", "fill", "prefix", "middle", "suffix")


def scan(name: str, data: object, hits: list[str]) -> None:
    """Collect token-ish strings containing any tag."""
    if isinstance(data, dict):
        for key, value in data.items():
            lowered = str(key).lower()
            if any(tag in lowered for tag in TAGS) and not isinstance(value, (dict, list)):
                hits.append(f"{name}.{key} = {value!r}")
            scan(f"{name}.{key}", value, hits)
    elif isinstance(data, list):
        for index, value in enumerate(data):
            if isinstance(value, str) and any(f"<|{tag}" in value.lower() or f"<{tag}" in value.lower() for tag in TAGS):
                hits.append(f"{name}[{index}] = {value!r}")
            else:
                scan(f"{name}[{index}]", value, hits)


def main() -> int:
    """Report FIM/mask markers in the model dir."""
    root = sys.argv[1]
    hits: list[str] = []
    for filename in ("config.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"):
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            try:
                scan(filename, json.load(handle), hits)
            except json.JSONDecodeError:
                continue
    if hits:
        print("found markers:")
        for line in hits[:40]:
            print("  " + line)
    else:
        print("no FIM / mask / fill markers found in config or tokenizer config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
