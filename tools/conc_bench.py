#!/usr/bin/env python3
"""Concurrency throughput benchmark against the local tunnel.

Fires C requests at once per level, each generating up to a fixed token budget, and reports
aggregate throughput and mean per-request latency. Every request carries a unique prefix so
the radix cache cannot skip prefill and inflate the numbers.

Usage:
    python conc_bench.py 1 2 4 8 12 --tokens 256 --rounds 2
    python conc_bench.py --tokens 128 --rounds 3
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:30000"
FILLER = (
    "The quarterly logistics review covers warehouse throughput, inbound freight scheduling, "
    "and inventory turnover across the regional distribution centres. "
)
ASK = (
    "\n\nList every metric this operations review mentions, and give a concrete improvement "
    "for each one. At least 20 items, numbered, as detailed as you can."
)


def prompt_for(tag: int, round_index: int) -> str:
    """Return a request-unique prompt, so no two requests share a cached prefix.

    Args:
        tag: Request index inside the concurrency level.
        round_index: Which round this is, to keep rounds independent too.
    """
    header = f"Session {tag:03d}-{round_index:02d} archival note.\n\n"
    return header + FILLER * 6 + ASK


def generate(prompt: str, max_new: int) -> dict:
    """Send one completion request and report its timings.

    Args:
        prompt: The user message.
        max_new: Token budget.
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
    finished = time.time()
    usage = payload.get("usage") or {}
    return {
        "tokens": usage.get("completion_tokens") or 0,
        "timed": finished - started,
        "started": started,
        "finished": finished,
    }


def run_level(conc: int, max_new: int, rounds: int) -> dict:
    """Run one concurrency level and return its aggregate numbers.

    Args:
        conc: Requests in flight at once.
        max_new: Token budget each.
        rounds: Repeats, of which the median aggregate is reported.
    """
    aggregates: list[float] = []
    latencies: list[float] = []
    first_error = ""
    for round_index in range(rounds):
        results: list[dict] = []
        lock = threading.Lock()
        barrier = threading.Barrier(conc)

        def worker(tag: int) -> None:
            prompt = prompt_for(tag, round_index)
            barrier.wait()
            try:
                row = generate(prompt, max_new)
            except Exception as failure:  # noqa: BLE001 - a benchmark row
                row = {"tokens": 0, "error": f"{type(failure).__name__}: {failure}"}
            with lock:
                results.append(row)

        threads = [threading.Thread(target=worker, args=(tag,)) for tag in range(conc)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        good = [row for row in results if row.get("tokens")]
        if not good:
            first_error = results[0].get("error", "unknown") if results else "no results"
            continue
        wall = max(row["finished"] for row in good) - min(row["started"] for row in good)
        tokens = sum(row["tokens"] for row in good)
        aggregates.append(tokens / wall)
        latencies.extend(row["timed"] for row in good)
    if not aggregates:
        return {"concurrency": conc, "error": f"all rounds failed; first: {first_error}"}
    return {
        "concurrency": conc,
        "median_tok_per_s": round(statistics.median(aggregates), 1),
        "best_tok_per_s": round(max(aggregates), 1),
        "mean_latency_s": round(statistics.mean(latencies), 2),
    }


def main() -> int:
    argv = sys.argv[1:]
    max_new, rounds = 256, 2
    if "--tokens" in argv:
        max_new = int(argv[argv.index("--tokens") + 1])
    if "--rounds" in argv:
        rounds = int(argv[argv.index("--rounds") + 1])
    head = argv[: argv.index("--tokens")] if "--tokens" in argv else argv
    levels = [int(item) for item in head if item.isdigit()] or [1, 2, 4, 8, 12]

    try:
        generate(prompt_for(999, 99), 8)
        print("warmup ok", flush=True)
    except Exception as failure:  # noqa: BLE001 - a benchmark probe
        print(f"warmup failed: {failure}")
        return 1

    rows = []
    for conc in levels:
        row = run_level(conc, max_new, rounds)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    with open("conc_bench.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
