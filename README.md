# Assignment 1 - Basic matrix multiplication in Python, Java and C

Big Data, Grado en Ciencia e Ingeniería de Datos (ULPGC), 2026-2027 - Guillermo Oltra Rodriguez.

Empirical study of the **triple-loop** (`i-j-k`) multiplication of dense `n x n` `float64` matrices,
implemented in **Python, Java and C** with the same loop order, the same numeric type and **the same
input files**. It measures **time** and **memory** over an increasing sequence of sizes, validates
correctness against NumPy and produces the raw data (CSV), tables, plots and the report.

**Research question:** how do runtime and memory grow with the size `n` for the triple-loop
multiplication implemented in Python, Java and C, on this machine and with these settings, and how
does that growth compare with the cubic model `O(n³)`?

- **Report (English):** [`informe/informe.md`](informe/informe.md) and PDF
  `informe/Assignment01_OltraRodriguez_Report.pdf`.
- Source-code comments, the per-campaign summaries (`summary.md`) and `results/LEEME.md` are in Spanish.

---

## 1. Requirements

| Tool | Version used | Purpose |
|---|---|---|
| CPython | 3.11.9 | Python kernel, runner, tests and analysis |
| GCC (MinGW-w64) | 11.2.0, `x86_64-w64-mingw32` | build the C kernel |
| JDK (Eclipse Temurin) | 17.0.20.1 | build and run the Java kernel |
| numpy, matplotlib, psutil | 2.4.6, 3.11.2, 7.2.2 | verification, plots, memory/thread sampling |
| markdown + Chrome/Edge | 3.x | only to build the PDF report |

```bash
py -3.11 -m pip install -r requirements.txt      # Linux/macOS: python3 -m pip ...
```

`gcc`, `javac` and `java` must be on the `PATH`. On Windows use `py -3.11` (or `python` if it points to
3.11); on Linux/macOS, `python3`.

## 2. Reproducing the whole study

From the project root:

```bash
# 1) Main campaign: builds, runs the tests, generates the inputs and measures Python, Java and C
py -3.11 scripts/run_benchmarks.py --out results/04_comparativa_3_lenguajes/raw.csv

# 2) Separate experiment: C and JVM flags, interleaved with the baseline configurations
py -3.11 scripts/run_benchmarks.py --configs c c-O0 c-O3native java java-warmup5x256 java-Xint \
    --sizes 10 50 100 200 400 600 800 1000 --out results/05_extra_variantes/raw.csv

# 3) Separate experiment: fine sweep of n in C and Java (own inputs in data_barrido/, ~0.5 GB, not versioned)
py -3.11 scripts/run_benchmarks.py --configs c java --reps 3 --data-dir data_barrido \
    --sizes 400 450 500 512 550 600 650 700 750 800 850 900 950 1000 1024 1050 1100 1150 \
            1200 1250 1300 1350 1400 1450 1500 1536 1550 1600 \
    --out results/07_extra_barrido_fino/raw.csv

# 4) Tables and plots
py -3.11 scripts/analyze.py --csv results/04_comparativa_3_lenguajes/raw.csv \
    --out-dir results/04_comparativa_3_lenguajes
py -3.11 scripts/analyze.py --csv results/05_extra_variantes/raw.csv \
    --out-dir results/05_extra_variantes
py -3.11 scripts/analyze.py --csv results/07_extra_barrido_fino/raw.csv \
    --out-dir results/07_extra_barrido_fino
py -3.11 scripts/plot_extras.py                    # figure of the fine sweep (07)

# 5) PDF report (optional)
py -3.11 informe/build_pdf.py
```

`run_benchmarks.py` does everything in order: records the environment, **builds** C and Java
(`scripts/build.py`), **runs all tests before measuring**, generates/verifies the input matrices and
launches the repetitions. Each measurement is appended to the CSV as soon as it finishes; an
interrupted campaign resumes without duplicating rows:

```bash
py -3.11 scripts/run_benchmarks.py --out results/04_comparativa_3_lenguajes/raw.csv --resume
```

It never overwrites an existing CSV (use `--resume`, another `--out`, or delete it). Individual steps:

```bash
py -3.11 scripts/build.py                          # builds C (3 variants) and Java
py -3.11 -m unittest discover -s tests -v          # 36 tests (Python, C, Java and runner)
py -3.11 src/python/datagen.py                     # generates data/A_n.bin and data/B_n.bin
py -3.11 src/python/datagen.py --verify            # checks the SHA-256 in the manifest
py -3.11 scripts/env_info.py --out-dir results/x   # records the environment
```

Running a worker by hand (all share the same interface and print one JSON line):

```bash
py -3.11 src/python/matmul.py --n 200 --a data/A_200.bin --b data/B_200.bin
build/c/matmul.exe            --n 200 --a data/A_200.bin --b data/B_200.bin
java -XX:+UseSerialGC -Xms2g -Xmx2g -cp build/java MatMul --n 200 --a data/A_200.bin --b data/B_200.bin
```

## 3. Layout

```
practica1/
├── README.md, requirements.txt
├── data/
│   ├── manifest.json            seed, format and SHA-256 of every matrix (versioned)
│   └── A_<n>.bin, B_<n>.bin     shared inputs (regenerated; not versioned)
├── src/
│   ├── python/matmul.py         Python kernel + worker performing ONE measurement
│   ├── python/datagen.py        deterministic generator of the shared inputs
│   ├── python/matrix_io.py      I/O of the shared binary format
│   ├── python/memtools.py       peak resident memory (Python)
│   ├── python/verify.py         comparison with NumPy (verification only)
│   ├── c/matmul.c               C kernel + worker (same protocol)
│   └── java/MatMul.java         Java kernel + worker (same protocol)
├── build/build_info.json        exact compilers, versions and flags (binaries are rebuilt)
├── scripts/
│   ├── build.py                 builds C (-O2, -O0, -O3 -march=native) and Java
│   ├── run_benchmarks.py        campaign: build -> tests -> data -> measurements -> CSV
│   ├── env_info.py              CPU, cores, RAM, OS, versions, flags, threads, power
│   ├── analyze.py               summary.csv/.md and plots/*.png
│   └── plot_extras.py           figure of the fine sweep (07)
├── tests/
│   ├── test_matmul.py           24 tests: Python kernel, I/O, data, verifier
│   └── test_workers.py          12 tests: C/Java workers (6 configurations) and the runner
├── results/                     one folder per campaign (see results/LEEME.md)
└── informe/                     report (informe.md), build_pdf.py and the PDF
```

## 4. Measurement decisions

| Aspect | Decision (same in the three languages unless stated) |
|---|---|
| Algorithm | Triple loop `i-j-k`: `C[i][j] += A[i][k] * B[k][j]`. No multiplication library. `n³` multiply-adds. |
| Numeric type | `float64` / `double`. `C` starts at `0.0`. |
| Inputs | **The same binary files** (float64 little-endian, row-major, no header) for the three languages. Uniform in [0, 1). Base seed 2026; `A_n`: `2026 * 1000003 + 2n`, `B_n`: +1. Generator: CPython `random.Random` (Mersenne Twister). |
| Timer | Monotonic: Python `time.perf_counter_ns()`, Java `System.nanoTime()`, C `QueryPerformanceCounter` (Windows) / `clock_gettime(CLOCK_MONOTONIC)`. |
| Timer boundary | **Kernel only**: the function that allocates `C`, initialises it to 0.0 and runs the loops. File reading, runtime start-up, warm-up, checksum and dump are **outside**. |
| Process | A **fresh process per repetition**, for each language. |
| Order | Interleaved: for each (n, repetition) the three languages run, rotating the order each repetition. |
| Repetitions | 5 per size. If the first takes > 120 s, 3 in total (recorded in `reps_planned`). |
| Very short tasks | If one multiplication takes < 100 ms a **batch** is timed and the time **per multiplication** is recorded (`time_ns = time_total_ns / batch`). The calibration run is discarded. |
| Warm-up | Python and C: 1 run on the 64x64 corner (or n x n if n < 64). Java: **200 runs on 64x64 matrices** (about 52 M iterations) so that the JIT compiles the kernel before timing (see the report, Section 3.5). Never part of the sample; recorded in `warmup_runs`/`warmup_n`. |
| Threads | Single-threaded kernel in all three (`threads = 1`). The JVM adds its own threads (JIT, GC, ...): `os_threads_max` and `jvm_threads` in the CSV. `OMP/OPENBLAS/MKL_NUM_THREADS=1` in the children. |
| C | `gcc -std=c11 -O2 -ffp-contract=off` (no FMA: same rounding as Python and Java). `noinline` kernel. |
| Java | Temurin 17, `-XX:+UseSerialGC -Xms2g -Xmx2g` (one GC thread, fixed heap: no GC inside the timed region, `gc_count_timed = 0`). Default JIT (tiered C1/C2). |
| Python | CPython 3.11.9, no options. Garbage collector disabled inside the timed region (as `timeit` does). Kernel inside a function (local variables). |
| Statistics | Median and inter-quartile range (IQR). All raw measurements are kept. |

### Data structures (unavoidable differences between languages)

| Language | Structure | Storage of A, B and C |
|---|---|---|
| Python | list of lists of `float` objects | about 32 n² B per matrix: 24 B per `float` object + 8 B per list pointer |
| Java | `double[][]`: array of references to `double[]` rows | about 8 n² B + 16 B header per row + outer array (4 B per compressed reference) |
| C | one contiguous row-major block of `n²` doubles, accessed as `double (*)[n]` | exactly 8 n² B |

## 5. Correctness

**Before measuring**, 36 tests run (`tests/`): known answers (1x1, 2x2 and 3x3 by hand, identity on
both sides, zero matrix), non-commutativity, exact integers, comparison with NumPy, binary I/O,
deterministic generator, batching, input errors and the runner's memory measurement. The C and Java
workers are tested as black boxes in their 6 configurations, checking that their `C` is
**bit-identical** to the Python kernel's (same summation order, no FMA) and that their checksum
matches `math.fsum`.

**During the campaign**, on repetition 1 of each (configuration, n) the worker dumps `C` after timing
and a separate process compares it with NumPy's `A @ B` (`rtol = 1e-9`, `atol = 1e-12`); the SHA-256
of the dump (`c_sha256`) is stored to check bit-identity across languages. On later repetitions the
`checksum` (exact sum of all elements, same method as Python's `math.fsum`) must match
repetition 1. Any failure aborts the campaign.

**Tolerances:** NumPy adds the products in a different order, so tiny differences (around 1e-15
relative) are expected. `rtol = 1e-9` is much larger than that but still catches any logic error (a
wrong index gives relative errors of order 1).

## 6. Memory

Three different quantities that are **not** compared as if they were the same:

- **`proc_peak_wset_bytes` (main metric for comparing languages):** peak working set of the whole
  process over its lifetime, read by the runner with `GetProcessMemoryInfo` on the child's handle when
  it ends. Same tool and definition for the three languages. Includes the interpreter, the JVM or the
  C runtime.
- **`inproc_peak_bytes` / `inproc_baseline_bytes`:** what each worker measures in-process, right after
  the multiplication and before loading data. Python and C: peak working set (same OS counter,
  `peak_wset`). Java: **peak used heap** (`MemoryPoolMXBean`), which excludes the JVM and includes
  not-yet-collected garbage.
- **Models** (estimates, not measurements): `array_model_bytes = 3 x 8 x n²` and `object_model_bytes`
  (actual structure of each language: Python `float` objects, Java `double[]` rows).

Known limitation: with a fixed 2 GB heap Java's GC never runs, so for small `n` its peak includes batch
and warm-up garbage.

## 7. Sizes and budgets (declared before measuring)

- Sizes: `10 25 50 100 200 300 400 500 600 800 1000 1200 1500 2000`.
- **300 s** wall-clock per process (including start-up and data loading) and **4096 MB** of memory per
  process (checked against the measured peak on Windows; enforced with `RLIMIT_AS` on Linux).
- A timeout, OOM or error is recorded as a **result** (no invented time) and that configuration does not
  try larger sizes; the others continue.
- `analyze.py` reports separately the largest size within the limits and the largest size attempted.

## 8. CSV schema (`raw.csv`)

One row per run (process). The older campaign `01_*` uses the previous schema (Python only;
`analyze.py` understands it).

| Column | Meaning |
|---|---|
| `timestamp_utc`, `language`, `variant`, `method`, `n`, `dtype`, `threads`, `repetition` | Identify the configuration (`variant = base` in the main comparison) |
| `reps_planned`, `run_order` | Planned repetitions for that size; global execution order |
| `status` | `OK`, `timeout`, `OOM`, `killed`, `error`, `WRONG_RESULT`, `mem_budget_exceeded` |
| `time_ns`, `time_total_ns`, `batch` | Time **per multiplication** (ns), total timed and number of multiplications in the batch |
| `warmup_runs`, `warmup_n` | Warm-up runs (excluded from the sample) and their size |
| `proc_peak_wset_bytes` | Peak working set of the process, measured by the runner (Section 6) |
| `inproc_peak_bytes`, `inproc_baseline_bytes`, `inproc_metric` | Memory measured inside the worker (Section 6) |
| `array_model_bytes`, `object_model_bytes` | Storage models |
| `checksum`, `verified`, `max_abs_err`, `max_rel_err`, `c_sha256` | Correctness (`verified`: `yes` = compared with NumPy, `checksum` = equal to repetition 1) |
| `os_threads_max`, `jvm_threads`, `gc_count_timed`, `gc_ms_timed` | OS threads of the process (sampled), Java threads, GC inside the timed region |
| `timeout_s`, `mem_budget_mb`, `wall_s`, `returncode` | Budgets, total process duration and exit code |
| `runtime_version`, `runtime_flags`, `note` | Runtime/compiler version, flags and notes |

## 9. Results

See [`results/LEEME.md`](results/LEEME.md) (Spanish). The main comparison is
`results/04_comparativa_3_lenguajes/`. The separate experiments (C and JVM flags in `05`, fine sweep of n
in `07`) are not mixed with it. Campaign `01` is an earlier
Python-only pilots kept for the record.

## 10. Known limitations

- One machine (AMD Ryzen 7 4800H laptop, Windows 11, plugged in). Conclusions are limited to these
  implementations, settings, inputs and hardware.
- The laptop is not an isolated environment: OS activity, frequency scaling (boost) and temperature add
  noise, hence fresh processes, interleaved order, median and IQR.
- External reading of the peak memory exists only on Windows; on Linux/macOS the in-process measurement
  (VmHWM / ru_maxrss) is used and `proc_peak_wset_bytes` stays empty.
- The code is portable (Linux/macOS) but has only been run on Windows.

## 11. Use of generative AI

A generative AI assistant (Claude Code, Anthropic) was used to clarify concepts, to help with the code and to help with writing the report; see the disclosure in Section 8 of the report. The author is responsible for the work.
