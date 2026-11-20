#!/usr/bin/env python3
"""Minimal Markdown -> standalone HTML converter for the report I just wrote.

Handles only the subset used in the report: ATX headings, pipe tables, fenced code,
bullet/ordered lists, horizontal rules, paragraphs, and inline `code` / **bold**.
No network, no dependencies - the output is one self-contained HTML file.
"""

from __future__ import annotations

import html
import re
import sys

INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")


def inline(text: str) -> str:
    """Escape text and apply inline code / bold."""
    out = html.escape(text)
    out = INLINE_CODE.sub(r"<code>\1</code>", out)
    out = BOLD.sub(r"<strong>\1</strong>", out)
    return out


def convert(markdown: str, title: str) -> str:
    """Convert the markdown body to an HTML fragment placed in a styled document."""
    lines = markdown.splitlines()
    body: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]

        if line.startswith("```"):
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(html.escape(lines[index]))
                index += 1
            index += 1
            body.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue

        if line.startswith("|"):
            rows: list[str] = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(lines[index])
                index += 1
            table = ["<table>"]
            for position, raw in enumerate(rows):
                cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
                    continue
                tag = "th" if position == 0 else "td"
                table.append(
                    "<tr>"
                    + "".join(f"<{tag}>{inline(cell)}</{tag}>" for cell in cells)
                    + "</tr>"
                )
            table.append("</table>")
            body.append("\n".join(table))
            continue

        match = re.match(r"^(#{1,4})\s+(.*)$", line)
        if match:
            level = len(match.group(1))
            body.append(f"<h{level}>{inline(match.group(2))}</h{level}>")
            index += 1
            continue

        if line.strip() in ("---", "***"):
            body.append("<hr>")
            index += 1
            continue

        if re.match(r"^\s*[-*]\s+", line):
            items: list[str] = []
            while index < len(lines) and re.match(r"^\s*[-*]\s+", lines[index]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[index]))
                index += 1
            body.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while index < len(lines) and re.match(r"^\s*\d+\.\s+", lines[index]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[index]))
                index += 1
            body.append("<ol>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ol>")
            continue

        if not line.strip():
            index += 1
            continue

        body.append(f"<p>{inline(line)}</p>")
        index += 1

    style = """
    :root { color-scheme: light; }
    body { margin: 0; padding: 40px 24px; background: #f6f7f9;
           font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
           color: #1c1f23; line-height: 1.65; }
    main { max-width: 1080px; margin: 0 auto; background: #fff; padding: 40px 48px;
           border-radius: 10px; box-shadow: 0 2px 14px rgba(0,0,0,.08); }
    h1 { font-size: 26px; border-bottom: 3px solid #2f6feb; padding-bottom: 12px; }
    h2 { font-size: 20px; margin-top: 34px; border-left: 4px solid #2f6feb; padding-left: 10px; }
    h3 { font-size: 16px; margin-top: 24px; color: #2b3440; }
    table { border-collapse: collapse; width: 100%; margin: 14px 0 20px; font-size: 14px; }
    th, td { border: 1px solid #d8dde3; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #eef2f7; font-weight: 600; }
    tr:nth-child(even) td { background: #fafbfc; }
    code { background: #eef2f7; padding: 1px 5px; border-radius: 4px;
           font-family: Consolas, "Courier New", monospace; font-size: 13px; }
    pre { background: #1e2430; color: #e6edf3; padding: 14px 16px; border-radius: 8px;
          overflow-x: auto; font-size: 13px; line-height: 1.5; }
    pre code { background: none; color: inherit; padding: 0; }
    hr { border: none; border-top: 1px solid #e2e6ea; margin: 28px 0; }
    ul, ol { padding-left: 24px; }
    li { margin: 4px 0; }
    """
    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{style}</style>\n</head>\n"
        f"<body>\n<main>\n{chr(10).join(body)}\n</main>\n</body>\n</html>\n"
    )


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: md2html.py <input.md> <output.html>")
        return 2
    source, target = sys.argv[1], sys.argv[2]
    with open(source, encoding="utf-8") as handle:
        markdown = handle.read()
    title = next(
        (line.lstrip("# ").strip() for line in markdown.splitlines() if line.startswith("# ")),
        "report",
    )
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(convert(markdown, title))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
