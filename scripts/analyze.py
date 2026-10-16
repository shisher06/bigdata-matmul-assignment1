#!/usr/bin/env python3
"""Genera tablas y gráficas a partir de uno o varios CSV de medidas en bruto.

Salidas (en ``--out-dir``):

* ``summary.csv``  una fila por (lenguaje, variante, n): mediana, IQR, memoria, hilos...
* ``summary.md``   tablas en Markdown: por configuración, comparación entre lenguajes,
                   memoria, límites, ajuste log-log, exponentes locales y corrección.
* ``plots/time_vs_n.png``        tiempo vs n (log-log), mediana e IQR, guías ∝ n³.
* ``plots/time_over_n3.png``     ns por iteración interna (tiempo / n³), un panel por configuración.
* ``plots/slowdown_vs_n.png``    factor de lentitud respecto a C (Python/C, Java/C).
* ``plots/memory_vs_n.png``      pico de memoria del proceso y memoria añadida vs modelos.
* ``plots/variability_vs_n.png`` IQR relativo (%) frente a n.

Estadísticos: mediana y rango intercuartílico (Q1, Q3 con interpolación lineal,
``numpy.percentile``) de las ejecuciones con ``status == OK``. Las fallidas (timeout, OOM...)
NO entran en ninguna mediana: se listan aparte como resultados.

Acepta el esquema antiguo (solo Python: ``peak_rss_bytes``...) y el actual (``variant``,
``proc_peak_wset_bytes``, ``inproc_*``...).

Uso::

    python scripts/analyze.py --csv results/04_comparativa_3_lenguajes/raw.csv \\
                              --out-dir results/04_comparativa_3_lenguajes
    python scripts/analyze.py --csv principal.csv extra.csv --out-dir results/05_...
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Paleta categórica de orden fijo (los tres primeros tonos validan todos los pares, también
# con daltonismo) y marcadores distintos para no depender solo del color.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
STYLE = {
    ("python", "base"): {"color": "#2a78d6", "marker": "o", "label": "Python"},
    ("java", "base"): {"color": "#eb6834", "marker": "s", "label": "Java"},
    ("c", "base"): {"color": "#1baf7a", "marker": "^", "label": "C (-O2)"},
    ("c", "O0"): {"color": "#eda100", "marker": "v", "label": "C (-O0)"},
    ("c", "O3native"): {"color": "#e87ba4", "marker": "D", "label": "C (-O3 -march=native)"},
    ("java", "Xint"): {"color": "#008300", "marker": "P", "label": "Java (-Xint, no JIT)"},
    ("java", "warmup5x256"): {"color": "#6250d6", "marker": "X",
                              "label": "Java (warm-up 5 x 256x256)"},
}
ORDER = list(STYLE)
FALLBACK = {"color": "#e34948", "marker": "*"}
BASE = [("python", "base"), ("java", "base"), ("c", "base")]

NUM_FIELDS = [
    "repetition", "reps_planned", "time_ns", "batch", "proc_peak_wset_bytes",
    "inproc_peak_bytes", "inproc_baseline_bytes", "array_model_bytes", "object_model_bytes",
    "max_rel_err", "max_abs_err", "os_threads_max", "jvm_threads", "gc_count_timed",
    "warmup_runs", "warmup_n", "run_order",
]
OLD_TO_NEW = {
    "peak_rss_bytes": "inproc_peak_bytes",
    "baseline_rss_bytes": "inproc_baseline_bytes",
    "memory_metric": "inproc_metric",
}


def _num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for old, new in OLD_TO_NEW.items():
                    if old in row and new not in row:
                        row[new] = row.pop(old)
                row.setdefault("variant", "base")
                row["variant"] = row["variant"] or "base"
                for key in NUM_FIELDS:
                    row[key] = _num(row.get(key))
                row["n"] = int(row["n"])
                rows.append(row)
    return rows


def key_of(row) -> tuple[str, str]:
    return row["language"], row["variant"]


def style_for(key) -> dict:
    if key in STYLE:
        return STYLE[key]
    return {**FALLBACK, "label": f"{key[0]} ({key[1]})"}


def sort_keys(keys) -> list:
    return sorted(keys, key=lambda k: (ORDER.index(k) if k in ORDER else len(ORDER), k))


def _median(values):
    values = [v for v in values if v is not None and v >= 0]
    return float(np.median(values)) if values else float("nan")


def summarise(rows: list[dict]) -> tuple[list[dict], dict]:
    """Devuelve (filas de resumen, fallos por configuración)."""
    groups: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[key_of(row)][row["n"]].append(row)

    summary: list[dict] = []
    failures: dict[tuple, list[dict]] = defaultdict(list)
    for key in sort_keys(groups):
        by_n = groups[key]
        for n in sorted(by_n):
            ok = [r for r in by_n[n] if r["status"] == "OK" and r["time_ns"] is not None]
            failures[key].extend(r for r in by_n[n] if r["status"] != "OK")
            if not ok:
                continue
            t = np.array([r["time_ns"] for r in ok]) / 1e9
            q1, med, q3 = np.percentile(t, [25, 50, 75])
            planned = max(int(r["reps_planned"] or 0) for r in ok) or len(ok)
            base = _median(r["inproc_baseline_bytes"] for r in ok)
            inproc = _median(r["inproc_peak_bytes"] for r in ok)
            verified = [r for r in ok if r.get("verified") == "yes"]
            summary.append({
                "language": key[0], "variant": key[1], "n": n,
                "reps_ok": len(ok), "reps_planned": planned,
                "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1,
                "rel_iqr_pct": 100 * (q3 - q1) / med if med > 0 else float("nan"),
                "min_s": float(t.min()), "max_s": float(t.max()),
                "ns_per_inner_iter": med * 1e9 / n**3,
                "proc_peak_mb": _median(r["proc_peak_wset_bytes"] for r in ok) / 1e6,
                "inproc_metric": ok[0].get("inproc_metric") or "",
                "inproc_peak_mb": inproc / 1e6,
                "inproc_above_baseline_mb": (inproc - base) / 1e6,
                "array_model_mb": (ok[0]["array_model_bytes"] or float("nan")) / 1e6,
                "object_model_mb": _median(r["object_model_bytes"] for r in ok) / 1e6,
                "batch": int(ok[-1]["batch"] or 1),
                "warmup": f"{int(ok[0]['warmup_runs'] or 0)}x{int(ok[0]['warmup_n'] or 0)}",
                "os_threads_max": max((r["os_threads_max"] or 0) for r in ok) or float("nan"),
                "jvm_threads": max((r["jvm_threads"] or 0) for r in ok) or float("nan"),
                "gc_in_timed": sum((r["gc_count_timed"] or 0) for r in ok),
                "max_rel_err": verified[0]["max_rel_err"] if verified else float("nan"),
                "c_sha256": next((r.get("c_sha256") for r in ok if r.get("c_sha256")), ""),
                "runtime": ok[0].get("runtime_version") or ok[0].get("python_version") or "",
                "flags": ok[0].get("runtime_flags") or "",
            })
    return summary, failures


def loglog_fit(points):
    """Pendiente de log(t) vs log(n). Devuelve (pendiente, error_estándar|None) o None."""
    if len(points) < 2:
        return None
    x = np.log([p[0] for p in points])
    y = np.log([p[1] for p in points])
    if len(points) >= 4:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        return float(coeffs[0]), float(np.sqrt(cov[0, 0]))
    return float(np.polyfit(x, y, 1)[0]), None


def fmt(x, digits=4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{digits}g}"


def fmt_s(x) -> str:
    """Tiempo legible con unidad."""
    if x is None or np.isnan(x):
        return "-"
    if x < 1e-3:
        return f"{x * 1e6:.3g} µs"
    if x < 1:
        return f"{x * 1e3:.3g} ms"
    return f"{x:.4g} s"


def series(summary, key):
    return sorted((s for s in summary if (s["language"], s["variant"]) == key),
                  key=lambda s: s["n"])


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def write_summary_csv(summary: list[dict], path: Path) -> None:
    if not summary:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def limits_lines(key, summary, failures, rows) -> list[str]:
    sel = series(summary, key)
    all_n = sorted({r["n"] for r in rows if key_of(r) == key})
    bad = failures.get(key, [])
    bad_n = sorted({int(r["n"]) for r in bad})
    ok_n = [s["n"] for s in sel if not bad_n or s["n"] < bad_n[0]]
    lines = [f"- Mayor tamaño probado dentro de los límites: **{max(ok_n) if ok_n else '-'}**",
             f"- Mayor tamaño intentado: **{max(all_n) if all_n else '-'}**"]
    for r in bad:
        lines.append(
            f"- n = {int(r['n'])}, repetición {int(r['repetition'] or 0)}: **{r['status']}** "
            f"({r.get('note') or ''}); límite de tiempo {r.get('timeout_s')} s, presupuesto de "
            f"memoria {r.get('mem_budget_mb') or r.get('mem_limit_mb') or '-'} MB. Sin tiempo asignado.")
    if not bad:
        lines.append("- Sin timeouts ni fallos de memoria.")
    return lines


def write_summary_md(summary, failures, rows, fit_min_n, path: Path) -> str:
    keys = sort_keys({(s["language"], s["variant"]) for s in summary})
    L = ["# Resumen de resultados", ""]
    L.append("Estadísticos sobre las ejecuciones con `status = OK`: mediana y rango intercuartílico "
             "(IQR = Q3 - Q1). Tiempos por multiplicación. Memoria en MB decimales (10⁶ B). "
             "Cada ejecución es un proceso nuevo.\n")

    # 1) Comparación entre lenguajes (configuraciones principales)
    base_keys = [k for k in BASE if k in keys]
    if len(base_keys) >= 2:
        by = {k: {s["n"]: s for s in series(summary, k)} for k in base_keys}
        ns = sorted(set().union(*[set(v) for v in by.values()]))
        L += ["## Comparación entre lenguajes (mediana del tiempo por multiplicación)", ""]
        head = "| n | " + " | ".join(style_for(k)["label"] for k in base_keys)
        ratios = []
        if ("c", "base") in base_keys:
            ratios = [k for k in base_keys if k != ("c", "base")]
            head += " | " + " | ".join(f"{style_for(k)['label']} / C" for k in ratios)
        L.append(head + " |")
        L.append("|---:|" + "---:|" * (len(base_keys) + len(ratios)))
        for n in ns:
            cells = [fmt_s(by[k][n]["median_s"]) if n in by[k] else "-" for k in base_keys]
            for k in ratios:
                c = by[("c", "base")].get(n)
                cells.append(f"{by[k][n]['median_s'] / c['median_s']:.3g}x"
                             if n in by[k] and c else "-")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L += ["", "«-» = sin medida válida para ese tamaño (ver *Límites*).", ""]

        L += ["## Memoria: pico del working set del proceso completo (MB)", ""]
        L.append("Medido por el runner con `GetProcessMemoryInfo` sobre el proceso hijo al terminar "
                 "(misma herramienta y definición en los tres lenguajes; incluye intérprete/JVM).\n")
        L.append("| n | " + " | ".join(style_for(k)["label"] for k in base_keys) +
                 " | modelo 3·8n² |")
        L.append("|---:|" + "---:|" * (len(base_keys) + 1))
        for n in ns:
            cells = [fmt(by[k][n]["proc_peak_mb"], 4) if n in by[k] else "-" for k in base_keys]
            model = next((by[k][n]["array_model_mb"] for k in base_keys if n in by[k]), float("nan"))
            L.append(f"| {n} | " + " | ".join(cells) + f" | {fmt(model, 4)} |")
        L.append("")

        # identidad bit a bit
        L += ["## Corrección", ""]
        same, diff = [], []
        for n in ns:
            shas = {k: by[k][n]["c_sha256"] for k in base_keys if n in by[k] and by[k][n]["c_sha256"]}
            if len(shas) >= 2:
                (same if len(set(shas.values())) == 1 else diff).append(n)
        L.append(f"- Matriz C idéntica bit a bit (mismo SHA-256 del volcado) en todos los lenguajes "
                 f"medidos: n = {', '.join(map(str, same)) or '-'}.")
        if diff:
            L.append(f"- C distinta entre lenguajes en n = {', '.join(map(str, diff))}.")
        errs = [s["max_rel_err"] for s in summary if not np.isnan(s["max_rel_err"])]
        if errs:
            L.append(f"- Error relativo máximo frente a NumPy (repetición 1, todas las configuraciones): "
                     f"{max(errs):.3g} (tolerancia rtol = 1e-9).")
        L.append("- Repeticiones 2..k: checksum igual al de la repetición 1 en todas las filas `OK`.")
        L.append("")

    # 2) Tablas por configuración
    for key in keys:
        sel = series(summary, key)
        st = style_for(key)
        L += [f"## {st['label']} - `{key[0]}/{key[1]}`", ""]
        if sel:
            L.append(f"Runtime: {sel[0]['runtime']}; flags: `{sel[0]['flags'] or '-'}`; "
                     f"métrica interna de memoria: `{sel[0]['inproc_metric']}`. «calent.» = "
                     f"ejecuciones x tamaño del calentamiento (incluye la calibración del lote).\n")
        L.append("| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | "
                 "pico proceso (MB) | interno (MB) | interno - base (MB) | modelo almac. (MB) | "
                 "hilos SO |")
        L.append("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in sel:
            reps = f"{s['reps_ok']}" + (f"/{s['reps_planned']}" if s["reps_ok"] != s["reps_planned"] else "")
            if s["reps_ok"] < 5:
                reps += " (!)"
            L.append(
                f"| {s['n']} | {reps} | {fmt_s(s['median_s'])} | {fmt_s(s['q1_s'])} | "
                f"{fmt_s(s['q3_s'])} | {fmt(s['rel_iqr_pct'], 3)} | {fmt(s['ns_per_inner_iter'], 4)} | "
                f"{s['batch']} | {s['warmup']} | {fmt(s['proc_peak_mb'], 4)} | {fmt(s['inproc_peak_mb'], 4)} | "
                f"{fmt(s['inproc_above_baseline_mb'], 4)} | {fmt(s['object_model_mb'], 4)} | "
                f"{fmt(s['os_threads_max'], 3)} |")
        L += ["", "(!) = menos de 5 repeticiones (justificado por coste; ver el informe).", ""]
        L += ["### Límites", "", *limits_lines(key, summary, failures, rows), ""]

        L += ["### Crecimiento frente al modelo cúbico", ""]
        pts = [(s["n"], s["median_s"]) for s in sel if s["n"] >= fit_min_n]
        fit = loglog_fit(pts)
        if fit is None:
            L.append(f"- Menos de 2 tamaños con n >= {fit_min_n}: no se puede ajustar la pendiente.")
        else:
            slope, se = fit
            se_txt = f" +- {se:.3f} (error estándar)" if se is not None else ""
            L.append(f"- Pendiente log-log de la mediana (n >= {fit_min_n}, {len(pts)} tamaños): "
                     f"**{slope:.3f}**{se_txt}; el modelo algorítmico predice 3.")
        if len(sel) >= 2:
            local = []
            for a, b in zip(sel, sel[1:]):
                e = np.log(b["median_s"] / a["median_s"]) / np.log(b["n"] / a["n"])
                local.append(f"{a['n']} a {b['n']}: {e:.2f}")
            L.append("- Exponente local entre tamaños consecutivos: " + "; ".join(local) + ".")
        L.append("")
    text = "\n".join(L)
    path.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
# Gráficas
# --------------------------------------------------------------------------- #
def _prep_axes(ax, xlabel: str, ylabel: str, title: str | None = None):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _legend(ax, fontsize=8.5, **kw):
    return ax.legend(frameon=False, fontsize=fontsize, labelcolor=INK, **kw)


def _auto_legend(ax, nseries: int, loc: str):
    """Con más de 3 series la leyenda va fuera, a la derecha, para no tapar datos."""
    if nseries > 3:
        return _legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    return _legend(ax, loc=loc)


def _end_labels(ax, items, min_gap_decades=0.085):
    """Etiquetas al final de cada serie (escala log en y) separadas para que no se solapen."""
    items = sorted(items, key=lambda it: it[1])
    placed = []
    for x, y, text in items:
        ly = np.log10(y)
        if placed and ly - placed[-1][1] < min_gap_decades:
            ly = placed[-1][1] + min_gap_decades
        placed.append((x, ly, text, y))
    for x, ly, text, _ in placed:
        ax.text(x * 1.06, 10 ** ly, text, va="center", fontsize=8.5, color=INK, clip_on=False)


def _new_fig(w=7.2, h=4.6, ncols=1):
    fig, ax = plt.subplots(1, ncols, figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def _xticks_n(ax, ns, sparse=False):
    candidates = (10, 100, 1000) if sparse else (10, 25, 50, 100, 200, 500, 1000, 2000)
    ticks = [n for n in candidates if min(ns) <= n <= max(ns)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def plot_all(summary: list[dict], out_dir: Path) -> list[Path]:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    keys = sort_keys({(s["language"], s["variant"]) for s in summary})
    all_n = sorted({s["n"] for s in summary})
    written = []

    # 1) tiempo vs n (log-log) con guías ∝ n³
    fig, ax = _new_fig()
    _prep_axes(ax, "n (matrix side, n x n)", "time per multiplication (s, log scale)",
               "Runtime vs. matrix size (median; bars = IQR)")
    guide_done = False
    for key in keys:
        st, s = style_for(key), series(summary, key)
        n = np.array([r["n"] for r in s], dtype=float)
        med = np.array([r["median_s"] for r in s])
        err = [med - np.array([r["q1_s"] for r in s]), np.array([r["q3_s"] for r in s]) - med]
        ax.errorbar(n, med, yerr=err, color=st["color"], marker=st["marker"], markersize=4.5,
                    linewidth=1.6, capsize=2, label=st["label"], zorder=3)
        anchor = next((r for r in s if r["n"] >= 100), None)
        if anchor is not None:
            xs = np.array([max(min(n), 50), max(n)], dtype=float)
            ax.plot(xs, anchor["median_s"] * (xs / anchor["n"]) ** 3, linestyle=(0, (4, 3)),
                    color=st["color"], alpha=0.45, linewidth=1.0, zorder=2,
                    label=None if guide_done else "n³ trend anchored at n = 100 (same colour)")
            guide_done = True
    ax.set_xscale("log")
    ax.set_yscale("log")
    _xticks_n(ax, all_n)
    _auto_legend(ax, len(keys), "upper left")
    written.append(_save(fig, plots / "time_vs_n.png"))

    # 2) ns por iteración interna: un panel por configuración (escala propia, desde 0)
    fig, axes = _new_fig(w=3.0 * len(keys) + 0.6, h=3.6, ncols=len(keys))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        st, s = style_for(key), series(summary, key)
        _prep_axes(ax, "n", "ns per inner iteration (time / n³)" if ax is axes[0] else "",
                   st["label"])
        ax.plot([r["n"] for r in s], [r["ns_per_inner_iter"] for r in s], color=st["color"],
                marker=st["marker"], markersize=4.5, linewidth=1.6)
        ax.set_xscale("log")
        _xticks_n(ax, all_n, sparse=True)
        ax.set_ylim(bottom=0, top=max(r["ns_per_inner_iter"] for r in s) * 1.15)
    fig.suptitle("Cost per inner iteration: a flat line would mean exact n³ scaling",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    written.append(_save(fig, plots / "time_over_n3.png"))

    # 3) factor de lentitud respecto a C (-O2)
    c = {s["n"]: s for s in series(summary, ("c", "base"))}
    others = [k for k in keys if k != ("c", "base")]
    if c and others:
        fig, ax = _new_fig()
        _prep_axes(ax, "n (matrix side, n x n)", "time / time of C -O2 (log scale)",
                   "Slowdown relative to C (-O2): same algorithm, same inputs")
        ends = []
        for key in others:
            st = style_for(key)
            pts = [(r["n"], r["median_s"] / c[r["n"]]["median_s"]) for r in series(summary, key)
                   if r["n"] in c]
            if pts:
                ax.plot(*zip(*pts), color=st["color"], marker=st["marker"], markersize=4.5,
                        linewidth=1.6, label=f"{st['label']} / C")
                ends.append((pts[-1][0], pts[-1][1], f"{pts[-1][1]:.3g}x"))
        ax.axhline(1.0, color=INK_2, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _xticks_n(ax, all_n)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0.8), hi)
        _end_labels(ax, ends)
        _auto_legend(ax, len(others), "best")
        written.append(_save(fig, plots / "slowdown_vs_n.png"))

    # 4) memoria: (a) pico del proceso completo, (b) memoria añadida vs modelos de almacenamiento
    fig, (ax1, ax2) = _new_fig(w=11.0, h=4.4, ncols=2)
    _prep_axes(ax1, "n (matrix side, n x n)", "MB (10^6 bytes, log scale)",
               "(a) Peak working set of the whole process")
    _prep_axes(ax2, "n (matrix side, n x n)", "MB (10^6 bytes, log scale)",
               "(b) Memory added inside the process (peak - start) and models")
    model_done = False
    for key in keys:
        st = style_for(key)
        s = [r for r in series(summary, key) if not np.isnan(r["proc_peak_mb"])]
        if not s:
            continue
        n = [r["n"] for r in s]
        ax1.plot(n, [r["proc_peak_mb"] for r in s], color=st["color"], marker=st["marker"],
                 markersize=4.5, linewidth=1.6, label=st["label"])
        # métrica interna: working set (Python, C) o heap usado (Java), pico - antes de cargar datos
        what = "heap" if s[0]["inproc_metric"].startswith("heap") else "working set"
        added = [(r["n"], r["inproc_above_baseline_mb"]) for r in s
                 if r["inproc_above_baseline_mb"] > 0.005]
        if added:
            ax2.plot(*zip(*added), color=st["color"], marker=st["marker"], markersize=4.5,
                     linewidth=1.6, label=f"{st['label']}: {what}, peak - start")
        model = [(r["n"], r["object_model_mb"]) for r in s if r["object_model_mb"] > 0]
        if model and key[0] != "c":
            ax2.plot(*zip(*model), color=st["color"], linestyle=(0, (1, 2)), linewidth=1.4,
                     label=f"{st['label']}: model of its data structure (A, B, C)")
        if not model_done:
            ax2.plot(n, [r["array_model_mb"] for r in s], color=INK_2, linestyle=(0, (4, 3)),
                     linewidth=1.0, label="3 x 8 x n² bytes (= C data structure)")
            model_done = True
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        _xticks_n(ax, all_n)
    _legend(ax1, loc="best")
    _legend(ax2, loc="lower right", fontsize=7.5)
    written.append(_save(fig, plots / "memory_vs_n.png"))

    # 5) variabilidad
    fig, ax = _new_fig(h=3.8)
    _prep_axes(ax, "n (matrix side, n x n)", "IQR / median (%)",
               "Variability across repetitions (each repetition is a fresh process)")
    for key in keys:
        st, s = style_for(key), series(summary, key)
        ax.plot([r["n"] for r in s], [r["rel_iqr_pct"] for r in s], color=st["color"],
                marker=st["marker"], markersize=4.5, linewidth=1.6, label=st["label"])
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    _xticks_n(ax, all_n)
    _auto_legend(ax, len(keys), "upper right")
    written.append(_save(fig, plots / "variability_vs_n.png"))
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, nargs="+", default=[ROOT / "results" / "raw.csv"])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--fit-min-n", type=int, default=100,
                        help="menor n del ajuste log-log (evita el régimen dominado por costes fijos)")
    args = parser.parse_args(argv)

    missing = [p for p in args.csv if not p.exists()]
    if missing:
        sys.exit(f"No existe {missing[0]}. Ejecuta primero scripts/run_benchmarks.py")
    rows = load_rows(args.csv)
    if not rows:
        sys.exit("Los CSV no contienen medidas")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, failures = summarise(rows)
    if not summary:
        sys.exit("No hay ninguna medida con status OK que resumir.")
    write_summary_csv(summary, args.out_dir / "summary.csv")
    text = write_summary_md(summary, failures, rows, args.fit_min_n, args.out_dir / "summary.md")
    plots = plot_all(summary, args.out_dir)
    if hasattr(sys.stdout, "reconfigure"):  # consola de Windows redirigida (cp1252)
        sys.stdout.reconfigure(errors="replace")
    print(text)
    print("Generado:", args.out_dir / "summary.csv", args.out_dir / "summary.md",
          *[str(p) for p in plots], sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
