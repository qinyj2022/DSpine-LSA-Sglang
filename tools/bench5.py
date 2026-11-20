#!/usr/bin/env python3
"""五类负载的单流基准 - 经本机隧道打远端 SGLang 的 OpenAI 兼容接口。

与交接文档 §6 的基线同口径：每类负载一条请求，temperature 0，max_tokens 400，
单流串行。输出每条请求的 prompt/completion token 数、墙钟时间与 tok/s，
并写成 JSON，便于 LSA+DSpine 与 DSpine-only 两个配置逐条对比。

用法：
    python bench5.py                # 全部五类
    python bench5.py A C            # 只跑指定类别
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30000"
MAX_NEW = 400

QUOTE = (
    "The Mississippi River is the primary river of the largest drainage basin in the "
    "United States. Beginning at Lake Itasca in northern Minnesota, it flows generally "
    "south for 2,340 miles to the Gulf of Mexico, draining an area of 1,151,000 square "
    "miles. Along the way it collects the waters of the Missouri and the Ohio, and it "
    "passes the cities of Minneapolis, St. Louis, Memphis and New Orleans."
)

WORKLOADS: dict[str, str] = {
    # A 自由问答
    "A": "用中文介绍混合专家模型（MoE）的基本原理：路由、专家并行和负载均衡各自解决什么问题。约400字。",
    # B 逐字引用
    "B": "下面是一段资料，请不加改动地逐字复述它的全文，只输出原文，不要任何解释：\n\n"
    + QUOTE,
    # C 代码
    "C": "写一个带类型标注的 Python LRU 缓存实现（基于 OrderedDict 或双向链表均可），"
    "并附上覆盖淘汰、命中与容量为1的单元测试。",
    # D JSON
    "D": "生成一个 JSON 数组，包含 20 个对象，每个对象有 id(int)、name(str)、price(float)、"
    "tags(list of str) 四个字段，只输出 JSON，不要 markdown 代码块。",
    # E 推理
    "E": "一个仓库上午9点开始装货，每小时装120箱，中午12点到13点休息。下午从13点开始每小时装150箱，"
    "到16点结束。第二天每小时装140箱不间断，问第二天要几点才能装完和第一天同样多的箱数？请给出逐步推理。",
}


def generate(prompt: str) -> dict:
    """Send one completion request and time it.

    Args:
        prompt: The user message.

    Returns:
        The measured row: workload-independent timings and token counts.
    """
    body = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_NEW,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.loads(response.read())
    wall = time.time() - started
    usage = payload.get("usage") or {}
    choice = (payload.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    completion = usage.get("completion_tokens") or 0
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "wall_s": round(wall, 2),
        "tok_per_s": round(completion / wall, 2) if wall > 0 else None,
        "head": text[:80].replace("\n", " "),
        "tail": text[-60:].replace("\n", " "),
    }


def main() -> int:
    wanted = [name.upper() for name in sys.argv[1:]] or list(WORKLOADS)
    rows: dict[str, dict] = {}
    for name in wanted:
        prompt = WORKLOADS[name]
        try:
            row = generate(prompt)
        except Exception as failure:  # noqa: BLE001 - a benchmark row, not a library
            row = {"error": f"{type(failure).__name__}: {failure}"}
        rows[name] = row
        print(f"{name}: {json.dumps(row, ensure_ascii=False)}", flush=True)
    with open("bench5.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
