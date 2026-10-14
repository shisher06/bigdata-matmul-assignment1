#!/usr/bin/env python3
"""Verificación de corrección frente a una referencia fiable (NumPy / BLAS).

NumPy se usa SOLO aquí, como referencia para comprobar resultados; nunca en las
implementaciones medidas.

Tolerancias (float64):

* ``rtol = 1e-9`` y ``atol = 1e-12`` (criterio de ``numpy.allclose``).
* Justificación: la suma ingenua de n productos acumula un error relativo de
  como mucho ~ n * u con u = 2^-53 ~ 1.1e-16 (cota pesimista; en la práctica
  crece como ~ sqrt(n) * u). Para n = 2000 la cota es ~ 2e-13. BLAS suma en otro
  orden, así que se esperan diferencias de ese orden. ``rtol = 1e-9`` deja
  cuatro órdenes de margen sobre esa cota y aun así detecta cualquier error
  real de lógica (un índice mal puesto produce errores relativos de orden 1).
  ``atol = 1e-12`` solo importa cerca de cero, donde los valores relativos no
  son informativos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("verify.py necesita NumPy: pip install numpy")

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


def compare(C, R, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> dict:
    """Compara C con la referencia R. Devuelve errores máximos y veredicto."""
    C = np.asarray(C, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if C.shape != R.shape:
        return {"ok": False, "reason": f"formas distintas {C.shape} vs {R.shape}"}
    if not np.all(np.isfinite(C)):
        return {"ok": False, "reason": "C contiene valores no finitos"}
    abs_err = np.abs(C - R)
    denom = np.maximum(np.abs(R), np.finfo(np.float64).tiny)
    return {
        "ok": bool(np.allclose(C, R, rtol=rtol, atol=atol)),
        "max_abs_err": float(abs_err.max()),
        "max_rel_err": float((abs_err / denom).max()),
        "rtol": rtol,
        "atol": atol,
    }


def load_matrix(path, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def verify_files(n: int, a_path, b_path, c_path, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL) -> dict:
    A = load_matrix(a_path, n)
    B = load_matrix(b_path, n)
    C = load_matrix(c_path, n)
    return compare(C, A @ B, rtol, atol)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compara C (binario) con A @ B de NumPy.")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--c", type=Path, required=True)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = p.parse_args(argv)
    result = verify_files(args.n, args.a, args.b, args.c, args.rtol, args.atol)
    print(json.dumps(result))
    return 0 if result["ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
