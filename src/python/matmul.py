#!/usr/bin/env python3
"""Kernel de la Práctica 1 (Python) y *worker* que hace UNA medición.

Algoritmo: multiplicación densa ingenua, triple bucle en orden i-j-k::

    for i: for j: for k: C[i][j] += A[i][k] * B[k][j]

Python puro: sin NumPy, BLAS ni ninguna multiplicación de librería (trabajo O(n³)).
Estructura de datos: lista de listas de ``float`` de Python.

Qué cronometra el worker (medición del *kernel*):

* DENTRO: la llamada a ``matmul_ijk`` (reserva e inicializa ``C`` y ejecuta los bucles).
* FUERA: lectura de ficheros, calentamiento, calibración, checksum, volcado de ``C``.
* Temporizador: ``time.perf_counter_ns()`` (monótono).
* El recolector de basura se desactiva durante la región cronometrada (como ``timeit``).

Calentamiento: ``--warmup-runs`` ejecuciones del kernel sobre una submatriz de
``--warmup-n`` x ``--warmup-n`` (o de ``n`` si es menor). Nunca forman parte de la muestra.

Tareas muy cortas: si el coste estimado de una multiplicación es menor que
``--batch-target-ms`` el worker hace una ejecución de calibración (que se descarta y
cuenta como un calentamiento más) y mide después un LOTE de multiplicaciones dentro de
una sola región cronometrada. ``time_ns`` es siempre el tiempo POR multiplicación
(``time_total_ns / batch``). Con ``--batch-target-ms 0`` nunca se usa lote.

Salida: la última línea de stdout es un objeto JSON. Códigos de salida:
0 = OK, 2 = error de entrada/salida o de argumentos, 3 = memoria insuficiente.

Uso::

    python src/python/matmul.py --n 200 --a data/A_200.bin --b data/B_200.bin
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matrix_io  # noqa: E402
import memtools  # noqa: E402

LANGUAGE = "python"
METHOD = "triple_loop_ijk"
DTYPE = "float64"
FLOAT_OBJECT_BYTES = sys.getsizeof(0.0)  # 24 B en CPython de 64 bits


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
def matmul_ijk(A: list[list[float]], B: list[list[float]], n: int) -> list[list[float]]:
    """C = A x B (matrices n x n), triple bucle i-j-k, ``C`` inicializada con 0.0."""
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def checksum_fsum(C: list[list[float]]) -> float:
    """Suma exactamente redondeada de todos los elementos (consume el resultado)."""
    return math.fsum(x for row in C for x in row)


# --------------------------------------------------------------------------- #
# Modelos de memoria (estimaciones, no mediciones)
# --------------------------------------------------------------------------- #
def array_model_bytes(n: int) -> int:
    """Mínimo teórico: tres matrices float64 densas, 3 * 8 * n^2 bytes."""
    return 3 * 8 * n * n


def _matrix_object_bytes(M: list[list[float]]) -> int:
    outer = sys.getsizeof(M)
    rows = sum(sys.getsizeof(row) for row in M)
    elements = sum(len(row) for row in M) * FLOAT_OBJECT_BYTES  # un objeto float por elemento
    return outer + rows + elements


def python_object_model_bytes(*matrices: list[list[float]]) -> int:
    """Almacenamiento de listas de listas con un objeto ``float`` por elemento (sys.getsizeof)."""
    return sum(_matrix_object_bytes(M) for M in matrices)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def warm_up(A, B, n: int, runs: int, warmup_n: int) -> tuple[int, int]:
    """Ejecuta el kernel sobre una esquina de las entradas. Devuelve (ejecuciones, tamaño)."""
    m = max(1, min(n, warmup_n))
    a = [row[:m] for row in A[:m]]
    b = [row[:m] for row in B[:m]]
    for _ in range(runs):
        matmul_ijk(a, b, m)
    return runs, m


def timed(A, B, n: int, batch: int):
    """Región cronometrada: ``batch`` multiplicaciones. Devuelve (C, tiempo_total_ns)."""
    C = None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = perf_counter_ns()
        for _ in range(batch):
            C = matmul_ijk(A, B, n)
        t1 = perf_counter_ns()
    finally:
        if gc_was_enabled:
            gc.enable()
    return C, t1 - t0


def _emit(payload: dict) -> None:
    print(json.dumps(payload))
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, required=True, help="lado de la matriz")
    p.add_argument("--a", type=Path, required=True, help="fichero binario de A")
    p.add_argument("--b", type=Path, required=True, help="fichero binario de B")
    p.add_argument("--dump-c", type=Path, default=None,
                   help="escribe C (binario compartido) DESPUÉS de medir tiempo y memoria")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--warmup-n", type=int, default=64)
    p.add_argument("--batch-target-ms", type=float, default=100.0,
                   help="duración mínima de la región cronometrada; 0 = sin lote")
    args = p.parse_args(argv)
    n = args.n
    if n < 1:
        p.error("--n debe ser >= 1")

    baseline, metric = memtools.peak_rss_bytes()  # intérprete antes de cargar datos
    try:
        A = matrix_io.read_matrix(args.a, n)
        B = matrix_io.read_matrix(args.b, n)

        warmup_runs, warmup_size = 0, 0
        batch = 1
        if args.warmup_runs > 0:
            t0 = perf_counter_ns()
            warmup_runs, warmup_size = warm_up(A, B, n, args.warmup_runs, args.warmup_n)
            warm_ns = perf_counter_ns() - t0
            if args.batch_target_ms > 0:
                # coste estimado de una multiplicación completa a partir del calentamiento
                est_ns = warm_ns / args.warmup_runs * (n / warmup_size) ** 3
                if est_ns < args.batch_target_ms * 1e6:
                    _, cal_ns = timed(A, B, n, 1)      # calibración: se descarta
                    warmup_runs += 1
                    batch = max(1, math.ceil(args.batch_target_ms * 1e6 / max(cal_ns, 1)))
        elif args.batch_target_ms > 0:
            _, cal_ns = timed(A, B, n, 1)
            warmup_runs += 1
            if cal_ns < args.batch_target_ms * 1e6:
                batch = max(1, math.ceil(args.batch_target_ms * 1e6 / max(cal_ns, 1)))

        C, total_ns = timed(A, B, n, batch)
        peak, metric = memtools.peak_rss_bytes()      # justo después de multiplicar
        checksum = checksum_fsum(C)
        object_model = python_object_model_bytes(A, B, C)
        if args.dump_c is not None:
            matrix_io.write_matrix(args.dump_c, C)
    except MemoryError:
        print("MemoryError", file=sys.stderr)
        _emit({"status": "OOM", "n": n, "note": "MemoryError"})
        return 3
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        _emit({"status": "error", "n": n, "note": str(exc)})
        return 2

    _emit({
        "status": "OK",
        "language": LANGUAGE,
        "method": METHOD,
        "n": n,
        "dtype": DTYPE,
        "threads": 1,
        "batch": batch,
        "time_total_ns": total_ns,
        "time_ns": total_ns / batch,
        "warmup_runs": warmup_runs,
        "warmup_n": warmup_size,
        "peak_rss_bytes": peak,
        "baseline_rss_bytes": baseline,
        "memory_metric": metric,
        "array_model_bytes": array_model_bytes(n),
        "object_model_bytes": object_model,
        "checksum": repr(checksum),
        "python_version": platform.python_version(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
