#!/usr/bin/env python3
"""Experimento aparte: ¿por qué Python 3.11 se dispara en n = 1000?

En la campaña principal el coste por iteración interna de Python pasa de ~95 ns (n = 800) a
~180 ns (n = 1000). Hipótesis: es un efecto de DISPOSICIÓN EN MEMORIA. El orden i-j-k recorre
B por columnas (B[0][j], B[1][j], ...) y en CPython cada elemento es un objeto ``float``
independiente; si la distancia entre las direcciones de B[k][j] y B[k+1][j] es (casi siempre)
múltiplo de 4096 B, todos esos objetos caen en los mismos conjuntos de las cachés L1/L2 y se
expulsan entre sí (fallos por conflicto), además de usar una página distinta cada vez.

Prueba: el MISMO kernel i-j-k, ejecutado solo sobre las primeras filas de C (las filas son
independientes, así que el coste por iteración es representativo), con dos disposiciones de B
que contienen exactamente los mismos valores:

* ``filas``    los objetos float de B se crean fila a fila (como al leer los ficheros en la campaña);
* ``columnas`` los objetos float de B se crean columna a columna, de modo que B[k][j] y
               B[k+1][j] quedan contiguos en memoria (el recorrido por columnas pasa a ser secuencial).

* ``traspuesta`` (DIAGNÓSTICO, opcional con ``--layouts``): B se guarda traspuesta y el kernel lee
               ``Bt[j][k]`` en lugar de ``B[k][j]``. Mismas operaciones y mismo orden de sumas
               (se comprueba que C sale idéntica bit a bit), pero B se recorre por filas. No forma
               parte de la comparación básica: solo sirve para aislar el efecto del recorrido por
               columnas.

Si la hipótesis es cierta, ``columnas`` elimina el salto de n = 1000 y ``filas`` lo reproduce.
Además se registra, con ``id()`` (la dirección del objeto en CPython), qué fracción de los saltos
B[k][j] -> B[k+1][j] es múltiplo exacto de 4096 B.

Salida: CSV (una fila por medida) y un resumen por pantalla. Uso::

    py -3.11 scripts/probe_python_layout.py --out results/06_extra_python_layout/raw.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import random
import statistics
import sys
from array import array
from collections import Counter
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIZES = [600, 800, 900, 960, 1000, 1024, 1040, 1100, 1200]


def partial_ijk(A, B, n, rows):
    """El kernel de la campaña (src/python/matmul.py) limitado a las primeras ``rows`` filas."""
    C = [[0.0] * n for _ in range(rows)]
    for i in range(rows):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def partial_ijk_bt(A, Bt, n, rows):
    """Diagnóstico: mismo cálculo leyendo B traspuesta por filas (Bt[j][k] == B[k][j])."""
    C = [[0.0] * n for _ in range(rows)]
    for i in range(rows):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * Bt[j][k]
    return C


def flat_values(n, seed):
    rnd = random.Random(seed)
    return array("d", [rnd.random() for _ in range(n * n)])


def by_rows(flat, n):
    """Objetos float creados fila a fila (como matrix_io.read_matrix)."""
    return [flat[i * n:(i + 1) * n].tolist() for i in range(n)]


def by_columns(flat, n):
    """Mismos valores; los objetos float se crean columna a columna (indexar un array('d')
    crea un objeto float nuevo cada vez)."""
    M = [[None] * n for _ in range(n)]
    for j in range(n):
        for k in range(n):
            M[k][j] = flat[k * n + j]
    return M


def stride_stats(B, n, cols=8, transposed=False):
    """Saltos de dirección entre los elementos consecutivos del recorrido interno (k -> k+1)."""
    strides = []
    for j in range(0, n, max(1, n // cols)):
        for k in range(n - 1):
            if transposed:
                strides.append(id(B[j][k + 1]) - id(B[j][k]))
            else:
                strides.append(id(B[k + 1][j]) - id(B[k][j]))
    mult4k = sum(1 for s in strides if s % 4096 == 0) / len(strides)
    mode, count = Counter(strides).most_common(1)[0]
    return mult4k, mode, count / len(strides)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--target-iters", type=float, default=2e7,
                   help="iteraciones internas por medida (determina cuántas filas se calculan)")
    p.add_argument("--layouts", nargs="+", default=["filas", "columnas"],
                   choices=["filas", "columnas", "traspuesta"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    fh = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["python_version", "n", "layout", "repetition", "rows", "time_ns",
                    "ns_per_inner_iter", "frac_strides_mult_4096", "stride_mode_bytes",
                    "stride_mode_frac"])
    pyver = platform.python_version()
    print(f"CPython {pyver}")
    # calentamiento del intérprete adaptativo (como en el worker)
    w_flat = flat_values(64, 1)
    partial_ijk(by_rows(w_flat, 64), by_rows(w_flat, 64), 64, 64)
    for n in args.sizes:
        rows = max(2, round(args.target_iters / (n * n)))
        flat_a, flat_b = flat_values(n, 2 * n), flat_values(n, 2 * n + 1)
        A = by_rows(flat_a, n)
        builders = {
            "filas": lambda: by_rows(flat_b, n),
            "columnas": lambda: by_columns(flat_b, n),
            "traspuesta": lambda: by_rows(array("d", (flat_b[k * n + j] for j in range(n)
                                                       for k in range(n))), n),
        }
        layouts = {name: builders[name]() for name in args.layouts}
        stats = {name: stride_stats(B, n, transposed=(name == "traspuesta"))
                 for name, B in layouts.items()}
        reference = None
        res = {name: [] for name in layouts}
        for rep in range(1, args.reps + 1):
            for name, B in layouts.items():  # alternadas
                gc.disable()
                t0 = perf_counter_ns()
                kernel = partial_ijk_bt if name == "traspuesta" else partial_ijk
                C = kernel(A, B, n, rows)
                t = perf_counter_ns() - t0
                gc.enable()
                if reference is None:
                    reference = C
                assert C == reference, "las disposiciones deben dar la misma C bit a bit"
                ns_iter = t / (rows * n * n)
                res[name].append(ns_iter)
                m4k, mode, mode_frac = stats[name]
                w.writerow([pyver, n, name, rep, rows, t, f"{ns_iter:.3f}", f"{m4k:.4f}", mode,
                            f"{mode_frac:.4f}"])
                fh.flush()
        line = "  ".join(
            f"{name}: {statistics.median(v):6.1f} ns/iter (saltos múltiplo de 4 KiB: "
            f"{stats[name][0] * 100:5.1f} %, salto más común {stats[name][1]} B)"
            for name, v in res.items())
        print(f"n={n:5d} filas_calculadas={rows:3d}  {line}", flush=True)
        del layouts, A
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
