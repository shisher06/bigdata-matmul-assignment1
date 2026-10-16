#!/usr/bin/env python3
"""Gráfica del barrido fino, que no cubre ``analyze.py``.

* ``results/07_extra_barrido_fino/plots/sweep_ns_per_iter.png``: coste por iteración interna
  (tiempo / n³) de C y Java en un barrido fino de n, con los múltiplos de 512 marcados.

Uso::

    py -3.11 scripts/plot_extras.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import GRID, INK, INK_2, STYLE, SURFACE, _prep_axes  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def plot_sweep(csv_path: Path, out: Path) -> Path:
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["status"] == "OK"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _prep_axes(ax, "n (matrix side, n x n)", "ns per inner iteration (time / n³)",
               "Fine sweep: cost per iteration of C (-O2) and Java (median; bars = min-max, 3 reps)")
    ns_all = sorted({int(r["n"]) for r in rows})
    for n in ns_all:
        if n % 512 == 0:
            ax.axvline(n, color=GRID, linewidth=6, zorder=0)
            ax.text(n, 0.985, f"n = {n}\n(8n B = {8 * n // 1024} KiB)", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=INK_2)
    for key in (("c", "base"), ("java", "base")):
        st = STYLE[key]
        pts = []
        for n in ns_all:
            t = [float(r["time_ns"]) / n**3 for r in rows
                 if (r["language"], r["variant"]) == key and int(r["n"]) == n]
            if t:
                pts.append((n, statistics.median(t), min(t), max(t)))
        x = [p[0] for p in pts]
        med = [p[1] for p in pts]
        ax.errorbar(x, med, yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    color=st["color"], marker=st["marker"], markersize=4.5, linewidth=1.4,
                    capsize=2, label=st["label"], zorder=3)
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, hi * 1.45)  # deja una franja libre arriba para las etiquetas de 512·k
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left",
              bbox_to_anchor=(0.0, 0.84))
    return _save(fig, out)


def main() -> int:
    sweep = ROOT / "results" / "07_extra_barrido_fino"
    written = []
    if (sweep / "raw.csv").exists():
        written.append(plot_sweep(sweep / "raw.csv", sweep / "plots" / "sweep_ns_per_iter.png"))
    print("Generado:", *written, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
