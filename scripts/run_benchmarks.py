#!/usr/bin/env python3
"""Campaña de la Práctica 1: compilación -> tests -> datos -> mediciones -> CSV.

Compara el MISMO algoritmo (triple bucle i-j-k, float64) en Python, Java y C::

    python scripts/run_benchmarks.py --out results/04_comparativa_3_lenguajes/raw.csv

1. Documenta el entorno (``environment.json`` y ``.md`` junto al CSV).
2. Compila C y Java (``scripts/build.py``) y registra compiladores y flags.
3. Ejecuta TODOS los tests de corrección antes de medir (``--skip-tests`` para omitirlos).
4. Genera/verifica las matrices de entrada compartidas (``src/python/datagen.py``).
5. Para cada tamaño y repetición lanza un PROCESO NUEVO de cada configuración. El orden de
   las configuraciones rota en cada repetición (intercalado), de modo que una deriva lenta de
   la máquina (temperatura, tareas de fondo) no favorezca sistemáticamente a ningún lenguaje.
6. Escribe una fila por ejecución en el CSV en cuanto termina (``fsync``).

Configuraciones (``--configs``)
    Principales (la comparación controlada): ``python``, ``java``, ``c``.
    Experimentos aparte (no se mezclan con la comparación principal): ``c-O0``,
    ``c-O3native``, ``java-Xint``, ``java-warmup5x256``.

Memoria
    * ``proc_peak_wset_bytes``: pico del working set del proceso completo durante toda su vida,
      leído por ESTE runner con ``GetProcessMemoryInfo`` sobre el handle del hijo cuando termina
      (Windows). Misma herramienta y misma definición para los tres lenguajes: es la métrica
      que se compara entre lenguajes.
    * ``inproc_*``: lo que mide el propio worker justo después de multiplicar. Python y C: pico
      de working set (el mismo contador del SO). Java: pico de heap usado (MXBeans), que es
      otra magnitud (no incluye la JVM).
    * ``os_threads_max``: máximo de hilos del SO del proceso, muestreado cada 50 ms durante los
      2 primeros segundos y cada segundo después.

Corrección durante la campaña
    En la repetición 1 de cada (configuración, n) el worker vuelca ``C`` (después de medir) y un
    proceso aparte lo compara con ``A @ B`` de NumPy (``verify.py``); se guarda además el SHA-256
    del volcado para comprobar que los lenguajes producen exactamente los mismos bits. En las
    demás repeticiones el ``checksum`` debe coincidir con el de la repetición 1.

Presupuestos y política de parada (declarados ANTES de medir)
    * ``--timeout`` segundos de reloj por proceso (incluye arranque y carga de datos).
    * ``--mem-budget-mb``: en Linux se aplica con ``RLIMIT_AS`` (Python y C); en Windows se
      comprueba con el pico medido por el runner y, si se supera, se registra como fallo.
    * Un timeout, OOM, error o resultado incorrecto se registra como resultado (sin tiempo
      inventado) y esa configuración NO intenta tamaños mayores; las demás siguen.
    * Si la 1.ª repetición de un tamaño tarda más de ``--slow-threshold-s`` s, esa
      configuración mide ese tamaño solo ``--reps-slow`` veces (queda en ``reps_planned``).

Seguridad de los datos
    Nunca sobrescribe un CSV existente: usa ``--resume`` para continuar, ``--out`` para otro
    fichero, o bórralo tú.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "python"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import build  # noqa: E402
import datagen  # noqa: E402
import env_info  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

METHOD = "triple_loop_ijk"
CSV_COLUMNS = [
    "timestamp_utc", "language", "variant", "method", "n", "dtype", "threads",
    "repetition", "reps_planned", "run_order", "status",
    "time_ns", "time_total_ns", "batch", "warmup_runs", "warmup_n",
    "proc_peak_wset_bytes", "inproc_peak_bytes", "inproc_baseline_bytes", "inproc_metric",
    "array_model_bytes", "object_model_bytes",
    "checksum", "verified", "max_abs_err", "max_rel_err", "c_sha256",
    "os_threads_max", "jvm_threads", "gc_count_timed", "gc_ms_timed",
    "timeout_s", "mem_budget_mb", "wall_s", "returncode",
    "runtime_version", "runtime_flags", "note",
]
MAIN_CONFIGS = ["python", "java", "c"]


# --------------------------------------------------------------------------- #
# Configuraciones: cómo se lanza cada worker
# --------------------------------------------------------------------------- #
def config_table() -> dict[str, dict]:
    java = shutil.which("java") or "java"
    jvm = build.JVM_FLAGS
    cp = ["-cp", str(build.java_classpath()), "MatMul"]
    return {
        "python": {"language": "python", "variant": "base",
                   "cmd": [sys.executable, str(SRC / "matmul.py")]},
        "java": {"language": "java", "variant": "base", "cmd": [java, *jvm, *cp]},
        "c": {"language": "c", "variant": "base", "cmd": [str(build.c_binary("O2"))]},
        # --- experimentos aparte ---
        "c-O0": {"language": "c", "variant": "O0", "cmd": [str(build.c_binary("O0"))]},
        "c-O3native": {"language": "c", "variant": "O3native",
                       "cmd": [str(build.c_binary("O3native"))]},
        "java-Xint": {"language": "java", "variant": "Xint",  # sin JIT: calentamiento como C/Python
                      "cmd": [java, *jvm, "-Xint", *cp, "--warmup-runs", "1", "--warmup-n", "64"]},
        "java-warmup5x256": {"language": "java", "variant": "warmup5x256",
                             "cmd": [java, *jvm, *cp, "--warmup-runs", "5", "--warmup-n", "256"]},
    }


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def child_env() -> dict:
    """Entorno de los procesos hijo: una sola hebra incluso en librerías numéricas."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    return env


def make_preexec(mem_budget_mb: int, language: str):
    """Límite de espacio de direcciones (solo Linux, Python y C). Devuelve función o None."""
    if not mem_budget_mb or not sys.platform.startswith("linux") or language == "java":
        return None  # la JVM reserva mucho espacio virtual: en Java el límite es -Xmx

    def apply_limit():
        import resource

        limit = mem_budget_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply_limit


if sys.platform == "win32":
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _K32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    _K32.K32GetProcessMemoryInfo.restype = wintypes.BOOL


def child_peak_wset(proc: subprocess.Popen) -> int | None:
    """Pico de working set de un hijo YA terminado (Windows). El handle de Popen sigue abierto,
    así que el SO conserva sus contadores. En otros sistemas devuelve None."""
    if sys.platform != "win32":
        return None
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    ok = _K32.K32GetProcessMemoryInfo(int(proc._handle), ctypes.byref(pmc), pmc.cb)
    return int(pmc.PeakWorkingSetSize) if ok else None


def parse_worker_json(stdout: str) -> dict | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def run_worker(cmd: list[str], n: int, a: Path, b: Path, dump_c: Path | None,
               timeout_s: float, preexec) -> dict:
    """Lanza un worker y espera a que termine muestreando sus hilos del SO.

    Devuelve un dict con status, data (JSON del worker o None), returncode, wall, note,
    proc_peak (bytes o None) y threads_max."""
    cmd = [*cmd, "--n", str(n), "--a", str(a), "--b", str(b)]
    if dump_c is not None:
        cmd += ["--dump-c", str(dump_c)]
    kwargs = {"preexec_fn": preexec} if preexec else {}
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=child_env(), cwd=ROOT, **kwargs)
    threads_max = 0
    ps = None
    if psutil is not None:
        try:
            ps = psutil.Process(proc.pid)
        except psutil.Error:
            ps = None
    timed_out = False
    while True:
        if ps is not None:
            try:
                threads_max = max(threads_max, ps.num_threads())
            except psutil.Error:
                pass
        remaining = timeout_s - (time.perf_counter() - t0)
        if remaining <= 0:
            proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True
            break
        # muestreo denso al principio (la JVM crea sus hilos al arrancar) y espaciado después
        interval = 0.05 if time.perf_counter() - t0 < 2.0 else 1.0
        try:
            stdout, stderr = proc.communicate(timeout=min(interval, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    wall = time.perf_counter() - t0
    peak = child_peak_wset(proc)
    result = {"data": None, "returncode": proc.returncode, "wall": wall, "proc_peak": peak,
              "threads_max": threads_max or None, "note": ""}
    if timed_out:
        result.update(status="timeout", returncode=None, note=f"cancelado a los {timeout_s:g} s")
        return result
    data = parse_worker_json(stdout)
    rc = proc.returncode
    if rc == 0 and data and data.get("status") == "OK":
        result.update(status="OK", data=data)
    elif rc == 3 or "MemoryError" in stderr or "OutOfMemoryError" in stderr:
        result.update(status="OOM", note="memoria insuficiente: " + (stderr.strip()[:200]))
    elif rc is not None and rc < 0:
        result.update(status="killed", note=f"proceso terminado por la señal {-rc}")
    else:
        result.update(status="error", note=(stderr.strip() or stdout.strip())[:300])
    return result


def run_verify(n: int, a: Path, b: Path, c: Path) -> dict:
    cmd = [sys.executable, str(SRC / "verify.py"), "--n", str(n),
           "--a", str(a), "--b", str(b), "--c", str(c)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=child_env())
    result = parse_worker_json(proc.stdout) or {"ok": False, "reason": proc.stderr.strip()[:200]}
    result["ok"] = bool(result.get("ok")) and proc.returncode == 0
    return result


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_tests() -> bool:
    print("== Tests de corrección (antes de medir) ==", flush=True)
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stderr.strip().splitlines() or [""])[-3:]
    print("   " + "\n   ".join(tail), flush=True)
    if proc.returncode != 0:
        print(proc.stderr)
    return proc.returncode == 0


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def runtime_label(cfg: dict, data: dict) -> tuple[str, str]:
    if cfg["language"] == "python":
        return f"{platform.python_implementation()} {data.get('python_version', '')}", ""
    return data.get("runtime_version", ""), data.get("runtime_flags", "")


# --------------------------------------------------------------------------- #
# Campaña
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=MAIN_CONFIGS,
                   help=f"configuraciones a medir (por defecto {' '.join(MAIN_CONFIGS)})")
    p.add_argument("--sizes", type=int, nargs="+", default=datagen.DEFAULT_SIZES)
    p.add_argument("--reps", type=int, default=5, help="repeticiones medidas por tamaño")
    p.add_argument("--reps-slow", type=int, default=3,
                   help="repeticiones totales si la 1.ª supera --slow-threshold-s")
    p.add_argument("--slow-threshold-s", type=float, default=120.0)
    p.add_argument("--timeout", type=float, default=300.0, help="segundos de reloj por proceso")
    p.add_argument("--mem-budget-mb", type=int, default=4096,
                   help="presupuesto de memoria por proceso en MB; 0 = sin presupuesto")
    p.add_argument("--seed", type=int, default=datagen.DEFAULT_SEED)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "raw.csv")
    p.add_argument("--resume", action="store_true",
                   help="continúa una campaña interrumpida sin duplicar filas")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args(argv)

    table = config_table()
    unknown = [c for c in args.configs if c not in table]
    if unknown:
        p.error(f"configuraciones desconocidas: {unknown}; opciones: {sorted(table)}")
    configs = list(dict.fromkeys(args.configs))
    sizes = sorted(set(args.sizes))
    if args.reps < 1 or args.reps_slow < 1:
        p.error("--reps y --reps-slow deben ser >= 1")
    if args.out.exists() and not args.resume:
        print(f"{args.out} ya existe. Usa --resume para continuar, --out para otro fichero, "
              "o bórralo si quieres empezar de cero.", file=sys.stderr)
        return 2

    if not args.skip_build:
        build.build()
    results_dir = args.out.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    env_info.write(results_dir, extra={"configs": {c: table[c] for c in configs},
                                       "sizes": sizes, "reps": args.reps,
                                       "timeout_s": args.timeout,
                                       "mem_budget_mb": args.mem_budget_mb})
    print(f"Entorno documentado en {results_dir}/environment.json y environment.md")

    if not args.skip_tests and not run_tests():
        print("Los tests fallan: no se mide nada hasta que pasen.", file=sys.stderr)
        return 1

    print("== Datos de entrada ==", flush=True)
    datagen.generate(sizes, args.data_dir, args.seed)
    problems = datagen.verify(args.data_dir)
    if problems:
        print("Datos de entrada inválidos:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    # Estado por configuración (se reconstruye desde el CSV con --resume)
    existing = load_existing(args.out) if args.resume else []
    state = {}
    for name in configs:
        cfg = table[name]
        rows = [r for r in existing
                if (r["language"], r["variant"]) == (cfg["language"], cfg["variant"])]
        state[name] = {
            "done": {(int(r["n"]), int(r["repetition"])): r for r in rows if r["status"] == "OK"},
            "failed_n": min((int(r["n"]) for r in rows if r["status"] != "OK"), default=None),
            "planned": {int(r["n"]): int(float(r["reps_planned"])) for r in rows
                        if r.get("reps_planned")},
            "ref_checksum": {int(r["n"]): r["checksum"] for r in rows
                             if r["status"] == "OK" and r["repetition"] == "1"},
        }
    sha_by_n: dict[int, dict[str, str]] = {}
    for r in existing:
        if r.get("c_sha256"):
            sha_by_n.setdefault(int(r["n"]), {})[f"{r['language']}/{r['variant']}"] = r["c_sha256"]
    run_order = len(existing)

    new_file = not args.out.exists()
    out = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS)
    if new_file:
        writer.writeheader()
        out.flush()

    def emit(row: dict) -> None:
        writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})
        out.flush()
        os.fsync(out.fileno())

    print("== Campaña ==", flush=True)
    print(f"configs={configs} tamaños={sizes} reps={args.reps} timeout={args.timeout:g}s "
          f"presupuesto memoria={args.mem_budget_mb or 'sin límite'}MB", flush=True)
    wrong_result = False
    tmp = tempfile.TemporaryDirectory(prefix="matmul_c_")
    try:
        for n in sizes:
            a_path, b_path = args.data_dir / f"A_{n}.bin", args.data_dir / f"B_{n}.bin"
            active = [c for c in configs
                      if state[c]["failed_n"] is None or n < state[c]["failed_n"]]
            for c in configs:
                if c not in active and state[c].get("skip_logged") != n:
                    print(f"{c} n={n}: se omite (falló antes en n={state[c]['failed_n']})")
                    state[c]["skip_logged"] = n
            if not active:
                break
            for c in active:
                state[c]["planned"].setdefault(n, args.reps)
            rep = 1
            while any(rep <= state[c]["planned"][n] for c in active):
                k = (rep - 1) % len(active)
                for name in active[k:] + active[:k]:  # orden rotado en cada repetición
                    st, cfg = state[name], table[name]
                    if st["failed_n"] is not None and n >= st["failed_n"]:
                        continue
                    planned = st["planned"][n]
                    if rep > planned:
                        continue
                    if (n, rep) in st["done"]:
                        continue
                    ref = st["ref_checksum"].get(n)
                    need_verify = ref is None
                    dump = Path(tmp.name) / f"C_{name}_{n}.bin" if need_verify else None
                    res = run_worker(cfg["cmd"], n, a_path, b_path, dump, args.timeout,
                                     make_preexec(args.mem_budget_mb, cfg["language"]))
                    run_order += 1
                    row = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "language": cfg["language"], "variant": cfg["variant"], "method": METHOD,
                        "n": n, "dtype": "float64", "threads": 1, "repetition": rep,
                        "reps_planned": planned, "run_order": run_order, "status": res["status"],
                        "proc_peak_wset_bytes": res["proc_peak"] if res["proc_peak"] else "",
                        "os_threads_max": res["threads_max"] or "",
                        "timeout_s": f"{args.timeout:g}", "mem_budget_mb": args.mem_budget_mb or "",
                        "wall_s": f"{res['wall']:.3f}",
                        "returncode": "" if res["returncode"] is None else res["returncode"],
                        "note": res["note"],
                    }
                    data = res["data"]
                    if res["status"] == "OK":
                        t_s = data["time_ns"] / 1e9
                        if rep == 1 and t_s > args.slow_threshold_s and planned > args.reps_slow:
                            planned = st["planned"][n] = args.reps_slow
                            row["reps_planned"] = planned
                            row["note"] = (f"1.ª repetición > {args.slow_threshold_s:g} s: "
                                           f"{args.reps_slow} repeticiones en total")
                        for key in ("time_ns", "time_total_ns", "batch", "warmup_runs", "warmup_n",
                                    "array_model_bytes", "object_model_bytes", "checksum",
                                    "jvm_threads", "gc_count_timed", "gc_ms_timed"):
                            row[key] = data.get(key, "")
                        row["inproc_peak_bytes"] = data["peak_rss_bytes"]
                        row["inproc_baseline_bytes"] = data["baseline_rss_bytes"]
                        row["inproc_metric"] = data["memory_metric"]
                        row["runtime_version"], row["runtime_flags"] = runtime_label(cfg, data)
                        if need_verify:
                            v = run_verify(n, a_path, b_path, dump)
                            row["max_abs_err"] = v.get("max_abs_err", "")
                            row["max_rel_err"] = v.get("max_rel_err", "")
                            row["verified"] = "yes" if v["ok"] else "no"
                            if dump.exists():
                                digest = sha256_file(dump)
                                row["c_sha256"] = digest
                                others = sha_by_n.setdefault(n, {})
                                diff = [k2 for k2, d2 in others.items() if d2 != digest]
                                if diff:
                                    row["note"] = (row["note"] + " " if row["note"] else "") + \
                                        f"C no es idéntica bit a bit a la de {', '.join(diff)}"
                                others[f"{cfg['language']}/{cfg['variant']}"] = digest
                                dump.unlink()
                            if v["ok"]:
                                st["ref_checksum"][n] = data["checksum"]
                            else:
                                row["status"] = "WRONG_RESULT"
                                row["note"] = f"verify.py rechaza C: {v}"
                        else:
                            same = float(data["checksum"]) == float(ref)
                            row["verified"] = "checksum" if same else "no"
                            if not same:
                                row["status"] = "WRONG_RESULT"
                                row["note"] = "checksum distinto del de la repetición 1"
                        peak = res["proc_peak"]
                        if (row["status"] == "OK" and args.mem_budget_mb and peak
                                and peak > args.mem_budget_mb * 1e6):
                            row["status"] = "mem_budget_exceeded"
                            row["note"] = f"pico {peak / 1e6:.0f} MB > {args.mem_budget_mb} MB"
                    emit(row)
                    label = f"{name:<16} n={n:<5} rep={rep}/{planned}"
                    if row["status"] == "OK":
                        st["done"][(n, rep)] = row
                        print(f"{label}: OK  {t_s:10.4f} s  pico proceso "
                              f"{(res['proc_peak'] or 0) / 1e6:7.1f} MB  hilos SO "
                              f"{res['threads_max']}  (batch={data['batch']}, "
                              f"verificado={row['verified']})", flush=True)
                    else:
                        print(f"{label}: {row['status']}  {row['note']}", flush=True)
                        st["failed_n"] = n
                        print(f"  -> {name}: no se intentan tamaños >= {n}")
                        if row["status"] == "WRONG_RESULT":
                            wrong_result = True
                rep += 1
            if wrong_result:
                print("Resultado incorrecto: campaña abortada.")
                break
    except KeyboardInterrupt:
        print("\nInterrumpido. Lo medido está guardado; continúa con --resume.")
        return 130
    finally:
        out.close()
        tmp.cleanup()
    print(f"Datos en bruto: {args.out}")
    print(f"Siguiente paso: python scripts/analyze.py --csv {args.out}")
    # timeout / OOM son resultados válidos (código 0); un resultado incorrecto es un fallo
    return 1 if wrong_result else 0


if __name__ == "__main__":
    raise SystemExit(main())
