#!/usr/bin/env python3
"""Needle retrieval through the local tunnel - does LSA + DSpine still find a fact buried mid-context?

Builds an operations dossier of a target length, plants one sentence at the midpoint,
and asks for the codename only. Prints whether the answer holds it, plus timings.
The point of the test is the *index*: if the selection dropped the needle's cluster,
the answer is wrong no matter how fast the decode is.

用法：
    python needle_test.py 8000 32000        # 指定上下文长度
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30000"
NEEDLE = "The internal project codename is BLUEHERON-7742."
FILLER = (
    "The quarterly logistics review covers warehouse throughput, inbound freight scheduling, "
    "and inventory turnover across the regional distribution centres. Each centre reports its "
    "dock utilisation and cold-chain compliance."
)


def build(target: int) -> str:
    """Return a dossier of about `target` tokens with the needle at its midpoint.

    Args:
        target: Rough token count wanted - the filler runs about 40 tokens a section.
    """
    sections = max(4, target // 40)
    body = "\n\n".join(f"Section {index + 1}. {FILLER}" for index in range(sections))
    cut = body.rfind("\n\n", 0, len(body) // 2)
    body = body[:cut] + "\n\n" + NEEDLE + "\n\n" + body[cut:]
    return (
        "You are given an internal operations dossier.\n\n"
        + body
        + "\n\nQuestion: What is the internal project codename stated in the dossier? "
        "Answer with the codename only, no explanation."
    )


def ask(prompt: str, max_new: int = 200) -> dict:
    """Send one completion request and report what came back.

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
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    answer = (message.get("content") or "") + (message.get("reasoning_content") or "")
    completion = usage.get("completion_tokens") or 0
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "wall_s": round(wall, 2),
        "tok_per_s": round(completion / wall, 2) if wall else None,
        "found": "BLUEHERON-7742" in answer,
        "tail": answer.strip()[-60:].replace("\n", " "),
    }


def main() -> int:
    targets = [int(value) for value in sys.argv[1:]] or [8000, 32000]
    rows = {}
    for target in targets:
        try:
            row = ask(build(target))
        except Exception as failure:  # noqa: BLE001 - a test row, not a library
            row = {"error": f"{type(failure).__name__}: {failure}"}
        rows[target] = row
        print(f"ctx={target}: {json.dumps(row, ensure_ascii=False)}", flush=True)
    with open("needle.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
