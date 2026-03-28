#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import subprocess
import tempfile
from pathlib import Path


CHROME_CANDIDATES = [
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
]


def find_chrome() -> Path:
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Chrome/Chromium not found.")


def inline_format(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def render_table(block: list[str]) -> str:
    rows = []
    for line in block:
        line = line.strip()
        if not line:
            continue
        if set(line.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
            continue
        cells = [inline_format(cell.strip()) for cell in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{cell}</th>" for cell in header)
    out.append("</tr></thead>")
    if body:
        out.append("<tbody>")
        for row in body:
            out.append("<tr>")
            out.extend(f"<td>{cell}</td>" for cell in row)
            out.append("</tr>")
        out.append("</tbody>")
    out.append("</table>")
    return "".join(out)


def markdown_to_html(md_text: str, title: str) -> str:
    lines = md_text.splitlines()
    parts: list[str] = []
    in_ul = False
    in_ol = False
    in_quote = False
    i = 0

    def close_lists() -> None:
        nonlocal in_ul, in_ol, in_quote
        if in_ul:
            parts.append("</ul>")
            in_ul = False
        if in_ol:
            parts.append("</ol>")
            in_ol = False
        if in_quote:
            parts.append("</blockquote>")
            in_quote = False

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith("|"):
            close_lists()
            table_block = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_block.append(lines[i].strip())
                i += 1
            parts.append(render_table(table_block))
            continue

        if not stripped:
            close_lists()
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            close_lists()
            level = len(heading.group(1))
            parts.append(f"<h{level}>{inline_format(heading.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("> "):
            if not in_quote:
                close_lists()
                parts.append("<blockquote>")
                in_quote = True
            parts.append(f"<p>{inline_format(stripped[2:])}</p>")
            i += 1
            continue
        elif in_quote:
            parts.append("</blockquote>")
            in_quote = False

        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet:
            if not in_ul:
                if in_ol:
                    parts.append("</ol>")
                    in_ol = False
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{inline_format(bullet.group(1))}</li>")
            i += 1
            continue

        ordered = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered:
            if not in_ol:
                if in_ul:
                    parts.append("</ul>")
                    in_ul = False
                parts.append("<ol>")
                in_ol = True
            parts.append(f"<li>{inline_format(ordered.group(1))}</li>")
            i += 1
            continue

        close_lists()
        parts.append(f"<p>{inline_format(stripped)}</p>")
        i += 1

    close_lists()

    css = """
    @page {
      size: A4;
      margin: 11mm 10mm 11mm 10mm;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body {
      font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      font-size: 10pt;
      line-height: 1.12;
      color: #111;
      column-count: 2;
      column-gap: 16pt;
      orphans: 2;
      widows: 2;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
    h1, h2, h3, h4, h5, h6 {
      margin: 0 0 5pt 0;
      line-height: 1.08;
      break-after: avoid;
      page-break-after: avoid;
    }
    h1 { font-size: 14pt; }
    h2 { font-size: 12pt; margin-top: 6pt; }
    h3 { font-size: 10.8pt; margin-top: 5pt; }
    p, ul, ol, blockquote, table {
      margin: 0 0 4pt 0;
      break-inside: avoid;
      page-break-inside: avoid;
    }
    ul, ol {
      padding-left: 13pt;
    }
    li {
      margin: 0 0 2pt 0;
    }
    blockquote {
      border-left: 2pt solid #999;
      padding-left: 6pt;
      color: #333;
    }
    code {
      font-family: "Menlo", "Monaco", monospace;
      font-size: 8.8pt;
      background: #f3f3f3;
      padding: 0 2pt;
      border-radius: 2pt;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 8.8pt;
    }
    th, td {
      border: 0.4pt solid #999;
      padding: 2pt 3pt;
      vertical-align: top;
    }
    th {
      background: #f2f2f2;
      font-weight: 600;
    }
    strong { font-weight: 700; }
    .doc-title {
      column-span: all;
      margin-bottom: 6pt;
      padding-bottom: 4pt;
      border-bottom: 1pt solid #bbb;
    }
    """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="doc-title"><h1>{html.escape(title)}</h1></div>
  {''.join(parts)}
</body>
</html>
"""


def print_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    title = markdown_path.stem
    md_text = markdown_path.read_text(encoding="utf-8")
    html_text = markdown_to_html(md_text, title)
    chrome = find_chrome()

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / f"{markdown_path.stem}.html"
        html_path.write_text(html_text, encoding="utf-8")
        cmd = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=2000",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render markdown to a compact 2-column PDF via Chrome.")
    parser.add_argument("input_markdown", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.input_markdown.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".pdf")
    print_to_pdf(input_path, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
