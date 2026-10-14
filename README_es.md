# Práctica 1 · Multiplicación de matrices básica en Python, Java y C

Estudio empírico del algoritmo de **triple bucle** (`i-j-k`) para multiplicar matrices densas
`n × n` de `float64`, implementado en **Python, Java y C** con el mismo orden de bucles, el mismo
tipo numérico y **los mismos ficheros de entrada**. Mide **tiempo** y **memoria** para una
secuencia creciente de tamaños, valida la corrección frente a NumPy y produce los datos en bruto
(CSV), las tablas, las gráficas y el informe.

**Pregunta de investigación:** ¿cómo crecen el tiempo de ejecución y la memoria con el tamaño `n`
para la multiplicación por triple bucle implementada en Python, Java y C, en esta máquina y con
estos ajustes, y cómo se compara ese crecimiento con el modelo cúbico `O(n³)`?

El informe (en inglés) está en [`informe/informe.md`](informe/informe.md) y en PDF, `informe/Assignment01_OltraRodriguez_Report.pdf`; la versión en castellano, en `informe/informe_es.md`. README en inglés: [`README.md`](README.md).

---

## 1. Requisitos

| Herramienta | Versión usada | Para qué |
|---|---|---|
| CPython | 3.11.9 | kernel de Python, runner, tests y análisis |
| GCC (MinGW-w64) | 11.2.0, `x86_64-w64-mingw32` | compilar el kernel de C |
| JDK (Eclipse Temurin) | 17.0.20.1 | compilar y ejecutar el kernel de Java |
| numpy, matplotlib, psutil | 2.4.6, 3.11.2, 7.2.2 | verificación, gráficas, memoria/hilos |
| markdown + Chrome/Edge | 3.x | solo para generar el PDF del informe |

```bash
py -3.11 -m pip install -r requirements.txt      # en Linux/macOS: python3 -m pip ...
```

`gcc`, `javac` y `java` deben estar en el `PATH`. En Windows usa `py -3.11` (o `python` si
apunta a la 3.11); en Linux/macOS, `python3`.

## 2. Reproducir el estudio completo

Desde la raíz del proyecto:

```bash
# 1) Campaña principal: compila, ejecuta los tests, genera los datos y mide Python, Java y C
py -3.11 scripts/run_benchmarks.py --out results/04_comparativa_3_lenguajes/raw.csv

# 2) Experimentos aparte (flags de C y de la JVM), intercalados con las configuraciones base
py -3.11 scripts/run_benchmarks.py --configs c c-O0 c-O3native java java-warmup5x256 java-Xint \
    --sizes 10 50 100 200 400 600 800 1000 --out results/05_extra_variantes/raw.csv

# 3) Barrido fino de n en C y Java (datos propios en data_barrido/, ~0,5 GB, no versionados)
py -3.11 scripts/run_benchmarks.py --configs c java --reps 3 --data-dir data_barrido \
    --sizes 400 450 500 512 550 600 650 700 750 800 850 900 950 1000 1024 1050 1100 1150 \
            1200 1250 1300 1350 1400 1450 1500 1536 1550 1600 \
    --out results/07_extra_barrido_fino/raw.csv

# 4) Disposición en memoria del kernel de Python (3.11 y 3.13)
py -3.11 scripts/probe_python_layout.py --out results/06_extra_python_layout/raw.csv
py -3.13 scripts/probe_python_layout.py --out results/06_extra_python_layout/raw.csv
py -3.11 scripts/probe_python_layout.py --layouts filas columnas traspuesta \
    --sizes 400 600 800 900 1000 1100 1200 --out results/06_extra_python_layout/diag_traspuesta.csv
py -3.13 scripts/probe_python_layout.py --layouts filas columnas traspuesta \
    --sizes 400 600 800 900 1000 1100 1200 --out results/06_extra_python_layout/diag_traspuesta.csv

# 5) Tablas y gráficas
py -3.11 scripts/analyze.py --csv results/04_comparativa_3_lenguajes/raw.csv \
    --out-dir results/04_comparativa_3_lenguajes
py -3.11 scripts/analyze.py --csv results/05_extra_variantes/raw.csv \
    --out-dir results/05_extra_variantes
py -3.11 scripts/analyze.py --csv results/07_extra_barrido_fino/raw.csv \
    --out-dir results/07_extra_barrido_fino
py -3.11 scripts/plot_extras.py                    # figuras del barrido y de 06

# 6) PDF del informe (opcional)
py -3.11 informe/build_pdf.py
```

`run_benchmarks.py` hace todo en orden: documenta el entorno, **compila** C y Java
(`scripts/build.py`), **ejecuta todos los tests antes de medir**, genera/verifica las matrices de
entrada y lanza las repeticiones. Cada medida se escribe en el CSV en cuanto termina; si se
interrumpe, se continúa sin duplicar filas:

```bash
py -3.11 scripts/run_benchmarks.py --out results/04_comparativa_3_lenguajes/raw.csv --resume
```

Nunca sobrescribe un CSV existente (usa `--resume`, otro `--out`, o bórralo tú). Pasos sueltos:

```bash
py -3.11 scripts/build.py                          # compila C (3 variantes) y Java
py -3.11 -m unittest discover -s tests -v          # 36 tests (Python, C, Java y runner)
py -3.11 src/python/datagen.py                     # genera data/A_n.bin y data/B_n.bin
py -3.11 src/python/datagen.py --verify            # comprueba el SHA-256 del manifest
py -3.11 scripts/env_info.py --out-dir results/x   # documenta el entorno
```

Ejecutar un worker a mano (todos tienen la misma interfaz y escriben una línea JSON):

```bash
py -3.11 src/python/matmul.py --n 200 --a data/A_200.bin --b data/B_200.bin
build/c/matmul.exe            --n 200 --a data/A_200.bin --b data/B_200.bin
java -XX:+UseSerialGC -Xms2g -Xmx2g -cp build/java MatMul --n 200 --a data/A_200.bin --b data/B_200.bin
```

## 3. Estructura

```
practica1/
├── README.md, requirements.txt
├── data/
│   ├── manifest.json            semilla, formato y SHA-256 de cada matriz (versionado)
│   └── A_<n>.bin, B_<n>.bin     entradas compartidas (se regeneran; no se versionan)
├── src/
│   ├── python/matmul.py         kernel Python + worker de UNA medición
│   ├── python/datagen.py        generador determinista de las entradas compartidas
│   ├── python/matrix_io.py      E/S del formato binario compartido
│   ├── python/memtools.py       pico de memoria residente (Python)
│   ├── python/verify.py         comparación con NumPy (solo verificación)
│   ├── c/matmul.c               kernel C + worker (mismo protocolo)
│   └── java/MatMul.java         kernel Java + worker (mismo protocolo)
├── build/build_info.json        compiladores, versiones y flags exactos (binarios: se regeneran)
├── scripts/
│   ├── build.py                 compila C (-O2, -O0, -O3 -march=native) y Java
│   ├── run_benchmarks.py        campaña: build -> tests -> datos -> mediciones -> CSV
│   ├── env_info.py              CPU, núcleos, RAM, SO, versiones, flags, hilos, energía
│   ├── analyze.py               summary.csv/.md y plots/*.png
│   ├── probe_python_layout.py   experimento aparte: disposición en memoria de B en Python
│   └── plot_extras.py           figuras del barrido fino (07) y del experimento 06
├── tests/
│   ├── test_matmul.py           24 tests del kernel Python, E/S, datos, verificador
│   └── test_workers.py          12 tests de los workers C/Java (6 configuraciones) y del runner
├── results/                     una carpeta por campaña (ver results/LEEME.md)
└── informe/                     informe.md, informe.pdf y build_pdf.py
```

## 4. Decisiones de medición

| Aspecto | Decisión (igual en los tres lenguajes salvo donde se indica) |
|---|---|
| Algoritmo | Triple bucle `i-j-k`: `C[i][j] += A[i][k] * B[k][j]`. Sin librerías de multiplicación. Trabajo `n³` multiplicaciones y sumas. |
| Tipo numérico | `float64` / `double`. `C` empieza en `0.0`. |
| Entradas | Los **mismos ficheros binarios** (float64 little-endian, row-major, sin cabecera) para los tres lenguajes. Uniformes en [0, 1). Semilla base 2026; `A_n`: `2026·1000003 + 2n`, `B_n`: +1. Generador: `random.Random` de CPython (Mersenne Twister). |
| Temporizador | Monótono: Python `time.perf_counter_ns()`, Java `System.nanoTime()`, C `QueryPerformanceCounter` (Windows) / `clock_gettime(CLOCK_MONOTONIC)`. |
| Frontera de medición | **Solo el kernel**: la función que reserva `C`, la inicializa a 0.0 y ejecuta los bucles. Lectura de ficheros, arranque del runtime, calentamiento, checksum y volcado quedan **fuera**. |
| Proceso | Un **proceso nuevo por repetición**, para cada lenguaje. |
| Orden | Intercalado: en cada (n, repetición) se ejecutan los tres lenguajes, rotando el orden en cada repetición. |
| Repeticiones | 5 por tamaño. Si la 1.ª tarda > 120 s, 3 en total (queda en `reps_planned`). |
| Tareas muy cortas | Si una multiplicación dura < 100 ms se mide un **lote** y se registra el tiempo **por multiplicación** (`time_ns = time_total_ns / batch`). La ejecución de calibración se descarta. |
| Calentamiento | Python y C: 1 ejecución sobre la esquina 64×64 (o n×n si n < 64). Java: **200 ejecuciones sobre matrices 64×64** (≈ 52 M iteraciones), para que C2 compile el kernel antes de medir (ver §6). Nunca entra en la muestra; queda en `warmup_runs`/`warmup_n`. |
| Hilos | El kernel es de un solo hilo en los tres lenguajes (`threads = 1`). La JVM crea además hilos propios (JIT, GC, ...): `os_threads_max` y `jvm_threads` en el CSV. `OMP/OPENBLAS/MKL_NUM_THREADS=1` en los hijos. |
| C | `gcc -std=c11 -O2 -ffp-contract=off` (sin FMA: mismos redondeos que Python y Java). Kernel `noinline`. |
| Java | Temurin 17, `-XX:+UseSerialGC -Xms2g -Xmx2g` (un solo hilo de GC, heap fijo: no hay GC durante la región medida, `gc_count_timed = 0`). JIT por defecto (tiered C1/C2). |
| Python | CPython 3.11.9, sin opciones. Recolector de basura desactivado en la región medida (como `timeit`). Kernel dentro de una función (variables locales). |
| Estadísticos | Mediana y rango intercuartílico (IQR). Se conservan todas las medidas en bruto. |

### Estructuras de datos (diferencias inevitables entre lenguajes)

| Lenguaje | Estructura | Almacenamiento de A, B y C |
|---|---|---|
| Python | lista de listas de objetos `float` | ≈ 32 n² B por matriz: 24 B por objeto `float` + 8 B por puntero en la lista |
| Java | `double[][]`: array de referencias a filas `double[]` | ≈ 8 n² B + 16 B de cabecera por fila + array externo (4 B por referencia comprimida) |
| C | un bloque contiguo row-major de `n²` doubles, accedido como `double (*)[n]` | exactamente 8 n² B |

## 5. Corrección

**Antes de medir** se ejecutan 36 tests (`tests/`): casos con respuesta conocida (1×1, 2×2 y 3×3 a
mano, identidad por ambos lados, matriz cero), no conmutatividad, enteros exactos, comparación con
NumPy, E/S binaria, generador determinista, lotes, errores de entrada y la medición de memoria del
runner. Los workers de C y Java se prueban como cajas negras en sus 6 configuraciones, y se
comprueba que su `C` es **idéntica bit a bit** a la del kernel de Python (mismo orden de sumas, sin
FMA) y que su checksum coincide con `math.fsum`.

**Durante la campaña**, en la repetición 1 de cada (configuración, n) el worker vuelca `C` después de
medir y un proceso aparte la compara con `A @ B` de NumPy (`rtol = 1e-9`, `atol = 1e-12`); se guarda
el SHA-256 del volcado (`c_sha256`) para comprobar la identidad bit a bit entre lenguajes. En las
demás repeticiones el `checksum` (suma correctamente redondeada, algoritmo de `math.fsum`, portado a
C y Java) debe coincidir con el de la repetición 1. Un fallo aborta la campaña.

**Tolerancias:** la suma ingenua de `n` productos acumula un error relativo de como mucho ~`n·u`
(`u = 2⁻⁵³ ≈ 1,1·10⁻¹⁶`), ~2·10⁻¹³ para `n = 2000`; BLAS suma en otro orden, así que se esperan
diferencias de ese orden. `rtol = 1e-9` deja cuatro órdenes de margen y detecta cualquier error de
lógica (un índice mal puesto da errores relativos de orden 1).

## 6. Memoria

Tres magnitudes distintas, que **no** se comparan entre sí como si fueran la misma:

- **`proc_peak_wset_bytes` (métrica principal para comparar lenguajes):** pico del working set del
  proceso completo durante toda su vida, leído por el runner con `GetProcessMemoryInfo` sobre el
  handle del hijo cuando termina. Misma herramienta y definición en los tres lenguajes. Incluye el
  intérprete, la JVM o el runtime de C.
- **`inproc_peak_bytes` / `inproc_baseline_bytes`:** lo que mide cada worker dentro de sí mismo,
  justo después de multiplicar y antes de cargar datos. Python y C: pico del working set (el mismo
  contador del SO, `peak_wset`). Java: **pico de heap usado** (`MemoryPoolMXBean`), que no incluye la
  JVM y sí incluye basura aún no recogida.
- **Modelos** (estimaciones, no medidas): `array_model_bytes = 3·8·n²` y `object_model_bytes`
  (estructura real de cada lenguaje: objetos `float` de Python, filas `double[]` de Java).

Limitación conocida: con un heap fijo de 2 GB el GC de Java no llega a ejecutarse, de modo que con
`n` pequeño su pico incluye la basura de los lotes y del calentamiento.

## 7. Tamaños y presupuestos (declarados antes de medir)

- Tamaños: `10 25 50 100 200 300 400 500 600 800 1000 1200 1500 2000`.
- **300 s** de reloj por proceso (incluye arranque y carga de datos) y **4096 MB** de memoria por
  proceso (en Windows se comprueba con el pico medido; en Linux se aplica con `RLIMIT_AS`).
- Un timeout, OOM o error se registra como **resultado** (sin tiempo inventado) y esa configuración
  no intenta tamaños mayores; las demás siguen.
- `analyze.py` informa por separado del mayor tamaño dentro de los límites y del mayor intentado.

## 8. Esquema del CSV (`raw.csv`)

Una fila por ejecución (proceso). Las campañas antiguas `01_*` y `03_*` usan el esquema anterior
(solo Python; `analyze.py` lo entiende).

| Columna | Significado |
|---|---|
| `timestamp_utc`, `language`, `variant`, `method`, `n`, `dtype`, `threads`, `repetition` | Identifican la configuración (`variant = base` en la comparación principal) |
| `reps_planned`, `run_order` | Repeticiones previstas para ese tamaño; orden global de ejecución |
| `status` | `OK`, `timeout`, `OOM`, `killed`, `error`, `WRONG_RESULT`, `mem_budget_exceeded` |
| `time_ns`, `time_total_ns`, `batch` | Tiempo **por multiplicación** (ns), total medido y nº de multiplicaciones del lote |
| `warmup_runs`, `warmup_n` | Calentamientos (excluidos de la muestra) y su tamaño |
| `proc_peak_wset_bytes` | Pico del working set del proceso, medido por el runner (§6) |
| `inproc_peak_bytes`, `inproc_baseline_bytes`, `inproc_metric` | Memoria medida dentro del worker (§6) |
| `array_model_bytes`, `object_model_bytes` | Modelos de almacenamiento |
| `checksum`, `verified`, `max_abs_err`, `max_rel_err`, `c_sha256` | Corrección (`verified`: `yes` = comparado con NumPy, `checksum` = igual que la rep. 1) |
| `os_threads_max`, `jvm_threads`, `gc_count_timed`, `gc_ms_timed` | Hilos del SO del proceso (muestreados), hilos Java, GC durante la región medida |
| `timeout_s`, `mem_budget_mb`, `wall_s`, `returncode` | Presupuestos, duración total del proceso y código de salida |
| `runtime_version`, `runtime_flags`, `note` | Versión del runtime/compilador, flags y observaciones |

## 9. Resultados

Ver [`results/LEEME.md`](results/LEEME.md). La comparación principal es
`results/04_comparativa_3_lenguajes/`. Los experimentos aparte (flags de C y de la JVM en `05`,
disposición en memoria en Python en `06`, barrido fino de n en `07`) no se mezclan con ella.

## 10. Limitaciones conocidas

- Una sola máquina (portátil AMD Ryzen 7 4800H, Windows 11, enchufado). Las conclusiones se limitan
  a estas implementaciones, ajustes, entradas y hardware.
- El portátil no es un entorno aislado: el SO, el escalado de frecuencia (turbo) y la temperatura
  introducen ruido. Por eso se usan procesos nuevos, orden intercalado, mediana e IQR.
- La lectura externa del pico de memoria solo existe en Windows; en Linux/macOS se usa la medida
  interna (VmHWM / ru_maxrss) y la columna `proc_peak_wset_bytes` queda vacía.
- El código es portable (Linux/macOS), pero solo se ha ejecutado en Windows.
