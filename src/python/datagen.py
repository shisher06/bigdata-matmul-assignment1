#!/usr/bin/env python3
"""Generador determinista de las matrices de entrada compartidas.

Genera, para cada tamaño ``n``, dos ficheros binarios ``A_<n>.bin`` y
``B_<n>.bin`` (float64 little-endian, row-major, valores uniformes en [0, 1)).
Los programas de Python, Java y C leen estos MISMOS ficheros, de modo que la
comparación usa las mismas matrices lógicas aunque los generadores aleatorios
de cada lenguaje sean distintos.

Reproducibilidad:

* Semilla de A_n = base_seed * 1_000_003 + 2n; la de B_n es esa + 1.
* ``random.Random(seed).random()`` produce la misma secuencia en cualquier
  versión de CPython 3 y en cualquier sistema operativo.
* ``data/manifest.json`` guarda semilla, formato y SHA-256 de cada fichero.
  Si regeneras y el SHA-256 no coincide con el del manifest, el script avisa.

Uso::

    python src/python/datagen.py                  # tamaños por defecto
    python src/python/datagen.py --sizes 10 100   # tamaños concretos
    python src/python/datagen.py --verify         # comprueba integridad
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_SEED = 2026
DEFAULT_SIZES = [10, 25, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]
MANIFEST_NAME = "manifest.json"

GENERATOR_DESC = (
    "CPython random.Random(seed).random() (Mersenne Twister), uniforme en [0, 1)"
)


def seeds_for(base_seed: int, n: int) -> tuple[int, int]:
    seed_a = base_seed * 1_000_003 + 2 * n
    return seed_a, seed_a + 1


def generate_values(seed: int, n: int) -> array:
    rnd = random.Random(seed)
    rand = rnd.random
    return array("d", [rand() for _ in range(n * n)])


def to_le_bytes(values: array) -> bytes:
    if sys.byteorder == "big":
        values = array("d", values)
        values.byteswap()
    return values.tobytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_path(data_dir: Path, name: str, n: int) -> Path:
    return data_dir / f"{name}_{n}.bin"


def _new_manifest(seed: int) -> dict:
    return {
        "generator": GENERATOR_DESC,
        "dtype": "float64 little-endian",
        "layout": "row-major, sin cabecera (8*n*n bytes)",
        "value_range": "[0, 1)",
        "base_seed": seed,
        "seed_rule": "A_n: base_seed*1000003 + 2n ; B_n: A_n + 1",
        "files": {},
    }


def load_manifest(data_dir: Path) -> dict | None:
    path = data_dir / MANIFEST_NAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(data_dir: Path, manifest: dict) -> None:
    manifest["files"] = dict(sorted(manifest["files"].items(), key=lambda kv: int(kv[0])))
    with open(data_dir / MANIFEST_NAME, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


def generate(
    sizes: list[int],
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    log=print,
) -> dict:
    """Genera (si faltan o están corruptos) los ficheros de ``sizes``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    if manifest is None or force and manifest.get("base_seed") != seed:
        manifest = _new_manifest(seed)
    elif manifest["base_seed"] != seed:
        raise ValueError(
            f"{MANIFEST_NAME} se creó con la semilla {manifest['base_seed']}, no {seed}. "
            "Usa --force para regenerar todo con la nueva semilla."
        )

    for n in sizes:
        if n < 1:
            raise ValueError(f"Tamaño inválido: {n}")
        seed_a, seed_b = seeds_for(seed, n)
        entry = manifest["files"].setdefault(str(n), {})
        for name, s in (("A", seed_a), ("B", seed_b)):
            path = matrix_path(data_dir, name, n)
            expected = entry.get(name, {}).get("sha256")
            if path.exists() and not force and expected and sha256_file(path) == expected:
                log(f"  {path.name}: ya existe y coincide con el manifest")
                continue
            payload = to_le_bytes(generate_values(s, n))
            digest = sha256_bytes(payload)
            if expected and not force and digest != expected:
                raise RuntimeError(
                    f"La regeneración de {path.name} con la semilla {s} NO reproduce el "
                    f"SHA-256 del manifest ({digest[:12]}… != {expected[:12]}…)."
                )
            path.write_bytes(payload)
            entry[name] = {
                "file": path.name,
                "seed": s,
                "bytes": len(payload),
                "sha256": digest,
            }
            log(f"  {path.name}: generado ({len(payload) / 1e6:.2f} MB)")

    save_manifest(data_dir, manifest)
    return manifest


def verify(data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    """Comprueba que los ficheros existen y coinciden con el manifest."""
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    if manifest is None:
        return [f"No existe {data_dir / MANIFEST_NAME}"]
    problems: list[str] = []
    for n, entry in manifest["files"].items():
        for name, info in entry.items():
            path = data_dir / info["file"]
            if not path.exists():
                problems.append(f"{path.name}: falta")
            elif path.stat().st_size != info["bytes"]:
                problems.append(f"{path.name}: tamaño distinto del manifest")
            elif sha256_file(path) != info["sha256"]:
                problems.append(f"{path.name}: SHA-256 distinto del manifest")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="regenera aunque existan")
    parser.add_argument("--verify", action="store_true", help="solo comprueba integridad")
    args = parser.parse_args(argv)

    if args.verify:
        problems = verify(args.data_dir)
        if problems:
            print("PROBLEMAS:\n  " + "\n  ".join(problems))
            return 1
        print("Todos los ficheros coinciden con el manifest.")
        return 0

    print(f"Generando datos en {args.data_dir} (semilla base {args.seed})")
    generate(sorted(set(args.sizes)), args.data_dir, args.seed, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
