#!/usr/bin/env python3
"""Documenta el entorno de ejecución (CPU, núcleos, RAM, SO, versiones, compiladores, hilos).

Genera ``results/environment.json`` y ``results/environment.md`` (este último
listo para pegar en el informe).  Uso::

    python scripts/env_info.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or "desconocido"


def _os_pretty() -> str:
    """Nombre legible del SO. En Windows, platform y el registro dicen "Windows 10" también en
    Windows 11 (conservan ese nombre por compatibilidad): se distingue por el número de build."""
    if sys.platform != "win32":
        return platform.platform()
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")

        def q(name):
            try:
                return winreg.QueryValueEx(key, name)[0]
            except OSError:
                return None

        build = int(q("CurrentBuild") or 0)
        family = "Windows 11" if build >= 22000 else "Windows 10"
        edition = q("EditionID") or ""
        edition = {"Core": "Home", "Professional": "Pro"}.get(edition, edition)
        return f"{family} {edition} {q('DisplayVersion') or ''} (build {build}.{q('UBR') or 0})".strip()
    except Exception:
        return platform.platform()


def _ram_total_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _physical_cores() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="ascii") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _power_state() -> dict:
    info: dict = {}
    try:
        import psutil

        battery = psutil.sensors_battery()
        if battery is not None:
            info["battery_percent"] = round(battery.percent, 1)
            info["power_plugged"] = bool(battery.power_plugged)
        freq = psutil.cpu_freq()
        if freq is not None:
            info["cpu_freq_mhz_current"] = round(freq.current, 1)
            info["cpu_freq_mhz_max"] = round(freq.max, 1) if freq.max else None
    except Exception:
        pass
    governor = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor:
        info["cpu_governor"] = governor
    return info


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "instalado")
    except Exception:
        return None


def _toolchain() -> dict:
    """Versiones de compiladores/JVM y flags, desde build/build_info.json (scripts/build.py)."""
    path = ROOT / "build" / "build_info.json"
    if not path.exists():
        return {}
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        "gcc": f"{info['gcc']['version']} ({info['gcc']['target']})",
        "c_flags": {v: d["flags"] for v, d in info["c"].items()},
        "javac": info["javac"]["version"],
        "java": " | ".join(info["java"]["version"]),
        "jvm_flags": info["java_build"]["jvm_flags"],
    }


def collect() -> dict:
    clock = time.get_clock_info("perf_counter")
    gil = None
    if hasattr(sys, "_is_gil_enabled"):
        gil = bool(sys._is_gil_enabled())
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": {
            "pretty": _os_pretty(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": _cpu_model(),
            "cores_physical": _physical_cores(),
            "cores_logical": os.cpu_count(),
        },
        "ram_total_bytes": _ram_total_bytes(),
        "toolchain": _toolchain(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
            "executable": sys.executable,
            "gil_enabled": gil,
        },
        "libraries": {
            "numpy (solo verificación y análisis)": _module_version("numpy"),
            "matplotlib (solo gráficas)": _module_version("matplotlib"),
            "psutil (opcional)": _module_version("psutil"),
        },
        "timer": {
            "function": "time.perf_counter_ns",
            "monotonic": clock.monotonic,
            "resolution_s": clock.resolution,
            "implementation": clock.implementation,
        },
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "power": _power_state(),
        "notes": [
            "Los tres kernels (Python, Java, C) se ejecutan en un solo hilo (threads = 1). "
            "La JVM crea además hilos propios (JIT, GC, señales...): su número queda en "
            "os_threads_max / jvm_threads del CSV.",
            "El runner fija OMP/OPENBLAS/MKL_NUM_THREADS=1 en los procesos hijo; "
            "NumPy solo se usa para verificar, fuera del cronómetro.",
        ],
    }


def to_markdown(env: dict) -> str:
    ram = env["ram_total_bytes"]
    lines = [
        "# Entorno de ejecución",
        "",
        f"- Fecha (UTC): {env['timestamp_utc']}",
        f"- Sistema operativo: {env['os'].get('pretty') or env['os']['platform']} "
        f"(platform: {env['os']['platform']})",
        f"- CPU: {env['cpu']['model']}",
        f"- Núcleos físicos / lógicos: {env['cpu']['cores_physical']} / {env['cpu']['cores_logical']}",
        f"- RAM total: {ram / 2**30:.1f} GiB" if ram else "- RAM total: desconocida",
        f"- Python: {env['python']['implementation']} {env['python']['version']} "
        f"({env['python']['compiler']})",
        f"- GIL activado: {env['python']['gil_enabled']}",
        f"- Temporizador: {env['timer']['function']} "
        f"(monótono={env['timer']['monotonic']}, resolución={env['timer']['resolution_s']} s)",
    ]
    tc = env.get("toolchain") or {}
    if tc:
        lines.append(f"- C: {tc['gcc']}; flags: " +
                     "; ".join(f"{v}: `{f}`" for v, f in tc["c_flags"].items()) +
                     " (O2 = comparación principal)")
        lines.append(f"- Java: {tc['java']}; {tc['javac']}; flags JVM: `{tc['jvm_flags']}`")
    for name, version in env["libraries"].items():
        lines.append(f"- {name}: {version}")
    threads = ", ".join(f"{k}={v}" for k, v in env["thread_env"].items() if v)
    lines.append(f"- Variables de hilos en el shell: {threads or 'ninguna definida'}")
    for key, value in env["power"].items():
        lines.append(f"- {key}: {value}")
    extra = env.get("campaign")
    if extra:
        lines += ["", "## Campaña", ""]
        lines.append(f"- Tamaños: {extra['sizes']}; repeticiones: {extra['reps']}; "
                     f"límite de tiempo: {extra['timeout_s']:g} s por proceso; "
                     f"presupuesto de memoria: {extra['mem_budget_mb']} MB por proceso")
        for name, cfg in extra["configs"].items():
            lines.append(f"- `{name}`: `{' '.join(_short(a) for a in cfg['cmd'])}`")
    lines += ["", *[f"> {note}" for note in env["notes"]], ""]
    return "\n".join(lines)


def _short(arg: str) -> str:
    """Rutas relativas a la raíz del proyecto, para que el registro sea legible."""
    try:
        return Path(arg).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return Path(arg).name if ("\\" in arg or "/" in arg) else arg


def write(out_dir: Path, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect()
    if extra:
        env["campaign"] = extra
    with open(out_dir / "environment.json", "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    (out_dir / "environment.md").write_text(to_markdown(env), encoding="utf-8")
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    env = write(args.out_dir)
    print(to_markdown(env))
    print(f"Guardado en {args.out_dir}/environment.json y environment.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
