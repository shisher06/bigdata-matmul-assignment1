#!/usr/bin/env python3
"""Genera ``informe/Assignment01_OltraRodriguez_Report.pdf`` a partir de ``informe/informe.md``.

Markdown -> HTML (paquete ``markdown``) con estilos de impresión A4 -> PDF con Chrome o Edge
en modo headless (``--print-to-pdf``). Las figuras se enlazan con rutas relativas a las
carpetas de ``results/``, así que el PDF siempre refleja las gráficas actuales.

Uso::

    py -3.11 informe/build_pdf.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "informe.md"
HTML = HERE / "informe.html"
PDF = HERE / "Assignment01_OltraRodriguez_Report.pdf"  # nombre sugerido en el enunciado

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
:root { --ink: #1b1b1a; --ink-2: #52514e; --rule: #d9d8d3; --accent: #2a78d6; --soft: #f4f3ef; }
html { font-size: 10.2pt; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink);
       line-height: 1.45; background: #fff; margin: 0; }
h1 { font-size: 1.85rem; line-height: 1.2; margin: 0 0 .3rem; }
h2 { font-size: 1.3rem; margin: 1.5rem 0 .5rem; padding-bottom: .2rem;
     border-bottom: 1px solid var(--rule); break-after: avoid; }
h3 { font-size: 1.08rem; margin: 1.1rem 0 .4rem; break-after: avoid; }
p, li { text-align: justify; hyphens: auto; }
a { color: var(--accent); text-decoration: none; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: .88em;
       background: var(--soft); padding: 0 .2em; border-radius: 3px; }
pre { background: var(--soft); padding: .6rem .8rem; border-radius: 4px; overflow-x: auto;
      font-size: .82rem; line-height: 1.35; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: .6rem 0 .9rem; font-size: .84rem;
        font-variant-numeric: tabular-nums; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border-bottom: 1px solid var(--rule); padding: .22rem .4rem; vertical-align: top; }
th { text-align: left; color: var(--ink-2); font-weight: 600; border-bottom: 1.5px solid var(--ink-2); }
td:not(:first-child) { text-align: right; }
table.left td { text-align: left; }
img { max-width: 82%; display: block; margin: .4rem auto .2rem; }
figure { margin: .8rem 0 1rem; break-inside: avoid; }
figcaption { font-size: .85rem; color: var(--ink-2); text-align: center; }
blockquote { margin: .6rem 0; padding: .4rem .8rem; border-left: 3px solid var(--accent);
             background: var(--soft); color: var(--ink); }
.meta { color: var(--ink-2); margin-bottom: 1.2rem; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 1.2rem 0; }
.pagebreak { break-before: page; }
"""


def find_browser() -> str:
    for candidate in BROWSERS:
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return path
    sys.exit("No se encontró Chrome ni Edge para imprimir el PDF.")


def build_html() -> None:
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list",
                                               "md_in_html", "sane_lists"])
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>Assignment 1 · Matrix multiplication</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    HTML.write_text(html, encoding="utf-8")


def build_pdf() -> None:
    browser = find_browser()
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--allow-file-access-from-files", f"--print-to-pdf={PDF}", HTML.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)


def main() -> int:
    build_html()
    build_pdf()
    print(f"Generado {PDF} ({PDF.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
