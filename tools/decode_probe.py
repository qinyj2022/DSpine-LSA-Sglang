#!/usr/bin/env python3
"""Long-context decode rate with the prefill taken out.

Sends the same dossier prompt twice: the first with a one-token budget warms the
radix cache, so the second request's prefill is a cache hit and its wall clock is
almost all decode. That isolates the number a sparse index is supposed to move.

用法：
    python decode_probe.py 64000 400
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

from needle_test import build

BASE = "http://127.0.0.1:30000"


def ask(prompt: str, max_new: int) -> dict:
    """Send one completion request and time it.

    Args:
        prompt: The dossier and question.
        max_new: Tokens to generate at most.
    """
    body = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read())
    wall = time.time() - started
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "wall_s": round(wall, 2),
    }


LONG_QUESTION = (
    "\n\nQuestion: 逐条列出这份档案里每个 Section 关注的运营指标，并为每一项给出一个"
    "改进建议。要求输出尽量详细，至少 15 条。"
)


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 64000
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    prompt = build(target)
    if len(sys.argv) > 3 and sys.argv[3] == "long":
        # 换一个需要长输出的问题，免得模型几十个 token 就答完，样本太小
        prompt = prompt.split("\n\nQuestion:")[0] + LONG_QUESTION

    warm = ask(prompt, 1)
    print(f"warm: {json.dumps(warm)}", flush=True)
    measured = ask(prompt, max_new)
    print(f"run : {json.dumps(measured)}", flush=True)
    generated = (measured["completion_tokens"] or 0) - 1
    if generated > 0:
        print(f"decode: {generated} tokens / {measured['wall_s']}s = "
              f"{generated / measured['wall_s']:.2f} tok/s (cached prefill)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
