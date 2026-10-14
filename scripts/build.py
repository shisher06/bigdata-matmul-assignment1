#!/usr/bin/env python3
"""Compila los workers de C y Java y documenta compiladores y flags.

Salidas::

    build/c/matmul.exe            C, configuración principal (-O2), la que se compara
    build/c/matmul_O0.exe         C sin optimizar        (experimento aparte, no se mezcla)
    build/c/matmul_O3native.exe   C -O3 -march=native     (experimento aparte, no se mezcla)
    build/java/MatMul.class       Java
    build/build_info.json         versiones exactas y flags usados

``-ffp-contract=off`` impide que GCC fusione ``C[i][j] += A[i][k] * B[k][j]`` en una sola
instrucción FMA: con FMA cambia el redondeo y C dejaría de coincidir bit a bit con Python y
Java. Ninguna variante usa ``-ffast-math``, así que el orden de las sumas no cambia.

Uso::

    python scripts/build.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
EXE = ".exe" if sys.platform == "win32" else ""

C_SOURCE = ROOT / "src" / "c" / "matmul.c"
JAVA_SOURCE = ROOT / "src" / "java" / "MatMul.java"
C_COMMON = ["-std=c11", "-Wall", "-Wextra", "-ffp-contract=off"]
# variante -> flags de optimización. "O2" es la configuración de la comparación principal.
C_VARIANTS = {
    "O2": ["-O2"],
    "O0": ["-O0"],
    "O3native": ["-O3", "-march=native"],
}
JAVA_RELEASE = "17"
# Flags de la JVM para la comparación principal (los usa run_benchmarks.py).
JVM_FLAGS = ["-XX:+UseSerialGC", "-Xms2g", "-Xmx2g"]


def c_binary(variant: str = "O2") -> Path:
    name = "matmul" if variant == "O2" else f"matmul_{variant}"
    return BUILD / "c" / f"{name}{EXE}"


def java_classpath() -> Path:
    return BUILD / "java"


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _first_line(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or proc.stderr).strip().splitlines()
    return text[0] if text else "desconocido"


def _all_lines(cmd: list[str]) -> list[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.stdout + proc.stderr).strip().splitlines()


def c_flags(variant: str) -> list[str]:
    return [*C_COMMON, *C_VARIANTS[variant]]


def build_c(gcc: str = "gcc") -> dict:
    (BUILD / "c").mkdir(parents=True, exist_ok=True)
    built = {}
    for variant in C_VARIANTS:
        flags = c_flags(variant)
        out = c_binary(variant)
        cmd = ["gcc", *flags, f'-DBUILD_FLAGS="{" ".join(flags)}"', "-o", _rel(out), _rel(C_SOURCE)]
        if sys.platform == "win32":
            cmd.append("-lpsapi")
        else:
            cmd.append("-lm")
        print("  $", " ".join(cmd), flush=True)
        subprocess.run([gcc, *cmd[1:]], check=True, cwd=ROOT)
        built[variant] = {"binary": _rel(out), "flags": " ".join(flags),
                          "command": " ".join(cmd)}
    return built


def build_java(javac: str = "javac") -> dict:
    out = java_classpath()
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["javac", "--release", JAVA_RELEASE, "-encoding", "UTF-8", "-Xlint:all", "-d", _rel(out),
           _rel(JAVA_SOURCE)]
    print("  $", " ".join(cmd), flush=True)
    subprocess.run([javac, *cmd[1:]], check=True, cwd=ROOT)
    return {"classpath": _rel(out), "command": " ".join(cmd),
            "jvm_flags": " ".join(JVM_FLAGS)}


def build() -> dict:
    gcc = shutil.which("gcc")
    javac = shutil.which("javac")
    java = shutil.which("java")
    if not gcc or not javac or not java:
        raise RuntimeError(f"Faltan herramientas: gcc={gcc} javac={javac} java={java}")
    print("== Compilación ==", flush=True)
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gcc": {"path": gcc, "version": _first_line([gcc, "--version"]),
                "target": _first_line([gcc, "-dumpmachine"])},
        "javac": {"path": javac, "version": _first_line([javac, "-version"])},
        "java": {"path": java, "version": _all_lines([java, "-version"])},
        "c": build_c(gcc),
        "java_build": build_java(javac),
    }
    (BUILD / "build_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return info


def load_info() -> dict | None:
    path = BUILD / "build_info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    info = build()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
