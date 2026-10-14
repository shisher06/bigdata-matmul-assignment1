"""Lectura y escritura de matrices cuadradas en el formato binario compartido.

Formato (idéntico para Python, Java y C):

* ``float64`` IEEE-754, *little-endian*.
* Orden por filas (*row-major*): primero toda la fila 0, luego la fila 1, etc.
* Sin cabecera: el fichero tiene exactamente ``8 * n * n`` bytes.
"""

from __future__ import annotations

import os
import sys
from array import array
from pathlib import Path

ITEMSIZE = 8  # bytes por float64


def expected_size(n: int) -> int:
    return ITEMSIZE * n * n


def read_matrix(path: str | os.PathLike, n: int) -> list[list[float]]:
    """Lee una matriz n x n como lista de listas de ``float`` de Python."""
    path = Path(path)
    size = path.stat().st_size
    if size != expected_size(n):
        raise ValueError(
            f"{path} tiene {size} bytes; para n={n} se esperaban {expected_size(n)}"
        )
    flat = array("d")
    with open(path, "rb") as fh:
        flat.fromfile(fh, n * n)
    if sys.byteorder == "big":  # los ficheros son little-endian
        flat.byteswap()
    return [flat[i * n : (i + 1) * n].tolist() for i in range(n)]


def write_matrix(path: str | os.PathLike, rows: list[list[float]]) -> None:
    """Escribe una matriz cuadrada (lista de listas) en el formato compartido.

    Fila a fila, para no crear una copia completa de la matriz (8·n² bytes): así volcar C
    en la repetición verificada no aumenta el pico de memoria del proceso.
    """
    with open(path, "wb") as fh:
        for row in rows:
            buf = array("d", row)
            if sys.byteorder == "big":
                buf.byteswap()
            buf.tofile(fh)
