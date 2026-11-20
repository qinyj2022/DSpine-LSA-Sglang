#!/usr/bin/env python3
"""Build a launch script that overrides one LSA design knob - a cost probe.

The `lsa` block of `--json-model-override-args` is the design's flat body: the knobs of
`model.indexer` (top_k, window_size, top_p, sink_size, bucket_size, ...) and of
`model.serving` (projections, kernels, decode, prefix_reuse, ...) sit side by side.

Usage:
    python3 make_lsa_probe.py <base-script> <out-script> key=value [key=value ...]
Example:
    python3 make_lsa_probe.py graph_lsa_log.sh probe_topk_1024.sh top_k=1024
"""

from __future__ import annotations

import json
import re
import sys

OVERRIDE = re.compile(r'--json-model-override-args\s+"(.*?)"\s*\\?\n', re.S)


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    base, out = sys.argv[1], sys.argv[2]
    pairs = []
    for item in sys.argv[3:]:
        key, value = item.split("=", 1)
        pairs.append((key, value))
    text = open(base, encoding="utf-8").read()
    match = OVERRIDE.search(text)
    if match is None:
        print("no --json-model-override-args in", base)
        return 1
    block = json.loads(match.group(1).replace('\\"', '"'))
    for key, value in pairs:
        try:
            block["lsa"][key] = json.loads(value)
        except json.JSONDecodeError:
            block["lsa"][key] = value
    encoded = json.dumps(block).replace('"', '\\"')
    text = text[: match.start()] + f'--json-model-override-args "{encoded}" \\\n' + text[match.end() :]
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("written:", out)
    print("lsa block:", json.dumps(block["lsa"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
