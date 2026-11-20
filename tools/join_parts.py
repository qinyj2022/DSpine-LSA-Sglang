#!/usr/bin/env python3
"""Reassemble files that publish_all.py had to split for upload.

用法：
    python join_parts.py <repo_dir>

Reads bigfiles.json in the repo root, concatenates `<path>.partNNN` in order and
writes the original file back, keeping the parts in place (they are harmless).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    """Join every recorded multi-part file."""
    root = Path(sys.argv[1])
    manifest = json.loads((root / "bigfiles.json").read_text())
    for original, parts in manifest["files"].items():
        out = bytearray()
        for part in parts:
            out += (root / part).read_bytes()
        target = root / original
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(out))
        print(f"{original}: {len(parts)} parts -> {len(out)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
