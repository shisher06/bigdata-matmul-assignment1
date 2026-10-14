# Multiplicación de matrices básica en Python, Java y C

<div class="meta" markdown="1">
**Big Data · Individual Assignment 1** — Grado en Ciencia e Ingeniería de Datos, ULPGC, curso 2026-2027<br>
**Autor:** Guillermo Oltra Rodriguez ·
**Material reproducible:** carpeta `practica1/` (código, datos, medidas en bruto, scripts de análisis; ver `README.md`)
</div>

> **Resumen.** Se compara el algoritmo de triple bucle `i-j-k` para multiplicar matrices densas
> `n × n` de `float64`, implementado del mismo modo en Python 3.11, Java 17 y C (gcc 11, `-O2`),
> con las mismas entradas y en la misma máquina (Ryzen 7 4800H, Windows 11). Se midieron 14
> tamaños entre 10 y 2000, con un proceso nuevo por medida, los lenguajes intercalados y 5
> repeticiones. Los tres programas producen **la misma matriz bit a bit**. Python es unas
> **100 veces** más lento que C (102–117× para 100 ≤ n ≤ 800) por el coste del intérprete. Java,
> con el JIT bien calentado, iguala a C mientras los datos caben en caché (1,1×) y es 1,5–2,7 veces
> más lento con n grande. El tiempo crece aproximadamente como n³, pero con pendientes de 3,2–3,6,
> por escalones de la jerarquía de memoria: el acceso a `B` por columnas deja de caber en la L3 y,
> con n potencia de 2, provoca conflictos de caché que hacen a C 6–7 veces más lento, hasta el punto
> de que con n = 1024 Java le gana. La memoria tiene un coste fijo del runtime (C ≈ 4 MB,
> Python ≈ 23 MB, JVM ≈ 60 MB) más una parte proporcional a n², que en Python es 5 veces mayor. Con
> el presupuesto de 300 s, Python llega a n = 1000 y Java y C a n = 2000. Tres experimentos aparte
> (flags del compilador y de la JVM, barrido fino de n y disposición en memoria en Python) explican
> esos efectos.

## 1. Pregunta de investigación e hipótesis

**Pregunta.** ¿Cómo crecen el tiempo de ejecución y la memoria de la multiplicación densa por
triple bucle cuando aumenta el tamaño `n` de las matrices `n × n`, implementada del mismo modo en
Python, Java y C, en una misma máquina y con unos ajustes documentados? ¿En qué medida sigue ese
crecimiento el modelo algorítmico `O(n³)`?

- **Factores que se varían:** el tamaño `n` (14 valores entre 10 y 2000) y la implementación
  (Python, Java, C).
- **Respuestas que se miden:** el tiempo por multiplicación (mediana e IQR de 5 repeticiones) y el
  pico de memoria del proceso (más dos métricas internas, §2.7).
- **Lo que se mantiene fijo:** el algoritmo y el orden de bucles (`i-j-k`), el tipo numérico
  (`float64`), los ficheros de entrada, la máquina, la sesión de medida y la frontera del cronómetro.

**Hipótesis** (escritas antes de la campaña principal; las cifras de partida proceden de la campaña
piloto de Python, `results/01_oficial_py311`, y de una prueba de humo con `n ≤ 300`):

- **H1 — crecimiento cúbico.** Para `n ≥ 100` el tiempo crecerá aproximadamente como `n³`
  (pendiente log-log ≈ 3) en los tres lenguajes. En C y Java se espera una pendiente algo mayor que 3
  cuando la matriz `B` (8n² bytes) deje de caber en la caché: el orden `i-j-k` recorre `B` por
  columnas, con un salto de 8n bytes entre accesos consecutivos.
- **H2 — coste del intérprete.** Python será unas 100 veces más lento que C, porque cada iteración
  interna ejecuta varias instrucciones de bytecode y manipula objetos `float` (≈ 70 ns por
  iteración en la campaña piloto, frente a ≈ 1 ns esperado en C).
- **H3 — JIT.** Tras el calentamiento, Java estará en el mismo orden de magnitud que C (factor 1–3),
  por las comprobaciones de límites de los arrays y la doble indirección de `double[][]`.
- **H4 — memoria.** Con `n` pequeño dominará un coste fijo del runtime (C: pocos MB; Python: ~20 MB;
  la JVM: decenas de MB). La parte que crece con `n²` será ≈ 3·8n² B en C y Java y ≈ 3·32n² B en
  Python (un objeto `float` de 24 B más un puntero de 8 B por elemento).

## 2. Método

### 2.1 Implementaciones

Las tres implementaciones ejecutan exactamente la misma sentencia en el mismo orden de bucles. No
usan ninguna librería de multiplicación: NumPy solo se usa fuera del cronómetro, para verificar.

```python
# Python (src/python/matmul.py)
def matmul_ijk(A, B, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
```

```java
// Java (src/java/MatMul.java)
static double[][] matmulIjk(double[][] A, double[][] B, int n) {
    double[][] C = new double[n][n];            // la JVM la inicializa a 0.0
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}
```

```c
/* C (src/c/matmul.c): bloque contiguo row-major visto como matriz 2-D (puntero a VLA, C99) */
__attribute__((noinline)) static double *matmul_ijk(const double *a_, const double *b_, int n) {
    double *c_ = malloc(sizeof(double) * (size_t)n * (size_t)n);
    const double (*A)[n] = (const double (*)[n])a_;
    const double (*B)[n] = (const double (*)[n])b_;
    double (*C)[n] = (double (*)[n])c_;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) C[i][j] = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return c_;
}
```

Las estructuras de datos son las idiomáticas de cada lenguaje. Es una diferencia inevitable, que se
documenta y forma parte de lo que se compara:

| Lenguaje | Estructura | Almacenamiento por matriz | Ejecución |
|:---|:---|:---|:---|
| Python | lista de listas de objetos `float` | ≈ 32 n² B (objeto de 24 B + puntero de 8 B) | CPython 3.11.9, intérprete con especialización adaptativa |
| Java | `double[][]`: array de referencias a filas `double[]` independientes | ≈ 8 n² B + 16 B por fila + 4 B por referencia | Temurin 17.0.20.1, JIT por niveles C1/C2 |
| C | un único bloque contiguo row-major de `n²` doubles | 8 n² B exactos | gcc 11.2.0 `-std=c11 -O2 -ffp-contract=off` |

`-ffp-contract=off` impide que gcc fusione la multiplicación y la suma en una instrucción FMA. Así el
redondeo es el mismo que en Python y Java, y las tres implementaciones hacen *exactamente* la misma
aritmética (se comprueba en §3.1). Ni Java ni CPython generan FMA. El kernel de C es `noinline`,
para que sea una función real como en los otros dos lenguajes.

### 2.2 Entradas

Los tres programas leen **los mismos ficheros binarios** `data/A_n.bin` y `data/B_n.bin`
(`float64` little-endian, por filas, sin cabecera), generados una sola vez con
`src/python/datagen.py`. Los valores son uniformes en [0, 1) (Mersenne Twister de CPython). La
semilla base es 2026; la de `A_n` es `2026·1000003 + 2n` y la de `B_n` es esa + 1. El SHA-256 de cada
fichero queda en `data/manifest.json` y se verifica antes de cada campaña. Así se evita que los
generadores aleatorios de cada lenguaje produzcan matrices distintas.

### 2.3 Entorno

| Elemento | Valor |
|:---|:---|
| Máquina | Portátil, AMD Ryzen 7 4800H (Zen 2), 8 núcleos físicos / 16 lógicos, 2,9 GHz base (turbo hasta 4,2 GHz) |
| Cachés (fabricante) | L1d 32 KiB y L2 512 KiB por núcleo; L3 2 × 4 MiB (4 MiB por grupo de 4 núcleos) |
| Memoria | 15,4 GiB |
| Sistema operativo | Windows 11 Home 26H2 (build 26300) |
| Energía | Enchufado (95 % de batería), plan de energía por defecto |
| Python | CPython 3.11.9 (MSC v.1938, 64 bits), sin opciones; GC desactivado en la región medida |
| Java | OpenJDK 17.0.20.1 (Temurin), `-XX:+UseSerialGC -Xms2g -Xmx2g`, JIT por defecto |
| C | gcc 11.2.0 (MinGW-w64, `x86_64-w64-mingw32`), `-std=c11 -O2 -ffp-contract=off` |
| Hilos | 1 hilo de cómputo en los tres. `OMP/OPENBLAS/MKL_NUM_THREADS=1` en los procesos hijo |
| Librerías auxiliares | NumPy 2.4.6 (solo verificación y análisis), matplotlib 3.11.2, psutil 7.2.2 |

Todo lo anterior se registra automáticamente en `results/*/environment.md` y en
`build/build_info.json`. Los ajustes de la JVM se eligieron para que su comportamiento sea
reproducible: un solo hilo de GC y un heap fijo de 2 GB, con el que no hay ninguna recolección
dentro de la región medida (`gc_count_timed = 0` en todas las filas). Fuera de lo indicado, todo
se mantuvo con sus valores por defecto.

### 2.4 Qué incluye el cronómetro

Se mide **solo el kernel**: la llamada que reserva `C`, la inicializa a 0.0 y ejecuta los tres
bucles. La reserva y la inicialización de `C` están **dentro** en los tres lenguajes. Quedan
**fuera** el arranque del intérprete o de la JVM, la lectura de los ficheros, el calentamiento, el
checksum y el volcado de `C`. Los temporizadores son monótonos: `time.perf_counter_ns()`,
`System.nanoTime()` y `QueryPerformanceCounter`. El resultado se consume siempre, con un checksum
de todos los elementos (y en C y Java, además, una variable `volatile` en los lotes), para que el
compilador no pueda eliminar el cálculo.

### 2.5 Protocolo de medición

- **Un proceso nuevo por repetición** y lenguaje: cada muestra empieza con el runtime en el mismo
  estado y tiene su propio pico de memoria.
- **Orden intercalado.** Para cada `n` y cada repetición se ejecutan los tres lenguajes uno tras
  otro, rotando el orden en cada repetición. Así una deriva lenta de la máquina (temperatura, tareas
  de fondo) no favorece sistemáticamente a ningún lenguaje.
- **5 repeticiones** por configuración. Si la primera supera 120 s, se hacen 3 en total por coste:
  solo ocurrió en Python con `n = 1000`.
- **Tareas muy cortas.** Si una multiplicación dura menos de 100 ms, se mide un lote de `b`
  multiplicaciones en una sola región cronometrada y se registra `tiempo/b`. El tamaño del lote se
  calibra con una ejecución previa que se descarta.
- **Calentamiento** (nunca entra en la muestra). En Python y C, 1 ejecución sobre la esquina 64×64,
  que basta para la especialización del bytecode de CPython 3.11. En Java, 200 ejecuciones sobre
  matrices 64×64 (ver §2.6).
- **Estadísticos.** Mediana y rango intercuartílico (Q1–Q3, interpolación lineal). Se conservan
  todas las medidas en bruto (`raw.csv`, una fila por proceso) y ninguna se ha excluido.

### 2.6 Calentamiento de Java: un efecto del JIT detectado al validar el protocolo

La primera versión calentaba Java con 5 ejecuciones de 256×256. Con `n = 1000` los tiempos eran
**bimodales**: ≈ 4,0 s en 5 de 6 ejecuciones y ≈ 2,4 s en la otra. Con `-XX:+PrintCompilation` y
`-XX:+LogCompilation` se vio la causa. C2 compilaba `matmulIjk` *durante la primera llamada larga*,
cuando el perfil aún no había visto salir nunca el bucle exterior. En la llamada siguiente esa rama
«imposible» se ejecutaba, saltaba un *uncommon trap* (`reason='unstable_if'`, bytecode 13: la
condición `i < n`), el código C2 se descartaba (*made not entrant*) y la medición se hacía con código
menos optimizado. Calentar con **muchas llamadas cortas** (200 × 64³ ≈ 52 M iteraciones, unos
60 ms) da un perfil completo. Con ese calentamiento hubo 0 desoptimizaciones en 5 de 5 ejecuciones y
tiempos estables de ≈ 2,4 s. La campaña `05` mide las dos variantes para cuantificar el efecto (§3.6).

### 2.7 Memoria: tres magnitudes distintas

1. **Pico del working set del proceso completo** (`proc_peak_wset_bytes`), leído por el runner con
   la API del SO `GetProcessMemoryInfo` sobre el proceso hijo cuando termina. La herramienta y la
   definición son las mismas en los tres lenguajes, así que **es la métrica que se compara entre
   lenguajes**. Incluye el intérprete, la JVM o el runtime de C.
2. **Medida interna del worker, justo después de multiplicar** (`inproc_*`). En Python y C es el
   pico del working set (el mismo contador del SO), y su diferencia con el valor previo a cargar los
   datos aísla lo que añaden A, B y C. En Java es el **pico de heap usado** (`MemoryPoolMXBean`), otra
   magnitud: no incluye la JVM y sí la basura aún no recogida.
3. **Modelos de almacenamiento** (estimaciones, no medidas): 3·8n² B para tres matrices densas, y el
   tamaño de la estructura real de Python (`sys.getsizeof`) y de Java (cabeceras de los arrays).

Estas magnitudes no se comparan entre sí como si fueran la misma.

### 2.8 Corrección

Antes de cada campaña se ejecutan automáticamente **36 tests** (`tests/`). Cubren casos con
respuesta conocida (1×1, 2×2 y 3×3 calculados a mano, identidad por ambos lados, matriz cero), la no
conmutatividad (que detecta índices intercambiados), enteros pequeños (resultado exacto), la
comparación con NumPy, la E/S binaria, el generador determinista, los lotes, los errores de entrada
y la medición de memoria del runner. Los workers de C y Java se prueban como cajas negras en sus seis
configuraciones (incluidas las de §3.6).

Durante la campaña, en la repetición 1 de cada (lenguaje, `n`) el worker vuelca `C` *después* de
medir y un proceso aparte la compara con `A @ B` de NumPy. La tolerancia es `rtol = 1e-9` y
`atol = 1e-12`. La suma ingenua de `n` productos acumula un error relativo de como mucho ~`n·u`
(`u = 2⁻⁵³`), ≈ 2·10⁻¹³ con `n = 2000`. NumPy (BLAS) suma en otro orden, así que se esperan
diferencias de ese orden. El margen es de cuatro órdenes de magnitud y aun así detecta cualquier
error de índices, que daría errores relativos de orden 1. Además se guarda el SHA-256 del volcado,
para comprobar si los tres lenguajes producen **los mismos bits**. En las repeticiones 2–5 el
checksum (suma correctamente redondeada, algoritmo de `math.fsum` portado a C y Java) debe coincidir
con el de la repetición 1.

### 2.9 Tamaños y presupuestos (declarados antes de medir)

- **Secuencia de tamaños:** 10, 25, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000.
- **Presupuesto de tiempo:** 300 s de reloj por proceso, incluidos el arranque y la carga de datos.
- **Presupuesto de memoria:** 4096 MB por proceso, comprobado con el pico medido.
- **Política de parada:** un timeout, un OOM o un error se registran como **resultado**, sin tiempo
  inventado, y ese lenguaje no prueba tamaños mayores; los demás siguen.

## 3. Resultados

Los datos están en `results/04_comparativa_3_lenguajes/`: `raw.csv` (194 procesos, uno por fila:
193 correctos y 1 timeout), `summary.md` (tablas completas por lenguaje, con Q1, Q3, lote,
calentamiento, memoria e hilos), `campaign.log` y `plots/`. La campaña duró 27 minutos
(14/10/2026, 16:56–17:24 UTC), en una sola sesión y con el portátil enchufado.

### 3.1 Corrección

- Los 36 tests pasaron antes de medir.
- En la repetición 1 de cada (lenguaje, `n`) la `C` coincide con NumPy dentro de la tolerancia: el
  **error relativo máximo fue 5,6·10⁻¹⁵**, lejos de `rtol = 1e-9`.
- **La matriz `C` es idéntica bit a bit en Python, Java y C para los 14 tamaños** (mismo SHA-256 del
  volcado). Las tres implementaciones hacen, por tanto, exactamente las mismas operaciones de coma
  flotante en el mismo orden: la comparación de tiempos es de verdad una comparación de *cómo
  ejecuta cada runtime el mismo cálculo*.
- En las repeticiones 2–5 el checksum coincidió siempre con el de la repetición 1.

### 3.2 Tiempo de ejecución

Mediana del tiempo por multiplicación (5 repeticiones; 3 en Python con `n = 1000`). Entre paréntesis,
la dispersión IQR/mediana cuando supera el 3 %.

| n | Python | Java | C (-O2) | Python / C | Java / C |
|---:|---:|---:|---:|---:|---:|
| 10 | 82,2 µs | 1,09 µs | 0,441 µs | 186× | 2,46× |
| 25 | 1,13 ms | 11,1 µs | 6,34 µs | 178× | 1,75× |
| 50 | 8,53 ms | 81,0 µs | 59,5 µs | 143× | 1,36× |
| 100 | 67,6 ms | 661 µs | 599 µs | 113× | 1,10× |
| 200 | 0,539 s | 5,74 ms | 5,28 ms | 102× | 1,09× |
| 300 | 1,90 s | 27,5 ms | 18,6 ms | 102× | 1,48× |
| 400 | 4,70 s | 70,6 ms (4,2 %) | 45,1 ms | 104× | 1,56× |
| 500 | 9,87 s | 154 ms (25 %) | 89,3 ms | 110× | 1,72× |
| 600 | 18,3 s | 259 ms (9,3 %) | 156 ms | 117× | 1,66× |
| 800 | 49,4 s | 1,19 s | 0,472 s | 105× | 2,51× |
| 1000 | 181,6 s (3 reps) | 2,46 s (3,6 %) | 0,934 s (4,1 %) | 194× | 2,63× |
| 1200 | **timeout (> 300 s)** | 4,20 s | 1,61 s | — | 2,60× |
| 1500 | no se intenta | 8,24 s | 4,60 s | — | 1,79× |
| 2000 | no se intenta | 23,4 s | 10,8 s | — | 2,17× |

<figure markdown="1">
![Tiempo frente a n](../results/04_comparativa_3_lenguajes/plots/time_vs_n.png)
<figcaption>Figura 1. Tiempo por multiplicación frente a n (escalas logarítmicas; mediana y barras
IQR, casi siempre más pequeñas que el marcador). Las líneas discontinuas son la predicción ∝ n³
anclada en n = 100.</figcaption>
</figure>

**Variabilidad.** El IQR relativo es menor del 2 % en 31 de las 39 configuraciones
(`plots/variability_vs_n.png`) y no supera el 4,2 % en ninguna, salvo Java con `n = 500` (25 %) y
`n = 600` (9 %). En esos dos casos la mayoría de las repeticiones se agrupan (142–154 ms y
247–259 ms), pero hay repeticiones sueltas hasta un 45 % más lentas (225 y 331 ms). Esa zona
coincide con la transición de la Figura 2, y C, intercalado con Java en esas mismas rondas, no
muestra dispersión (IQR < 3 %). Por eso se atribuye a la propia ejecución de Java (código JIT o
colocación de las filas en el heap, que cambian de un proceso a otro) más que a ruido de la máquina.
Python, al ser mucho más lento, es el más estable en términos relativos (IQR ≤ 1,8 %).

**Límites.** Con el presupuesto declarado (300 s por proceso), el mayor tamaño completado por
Python es **n = 1000**. Con `n = 1200` agotó el tiempo en la primera repetición; se registró como
*timeout*, sin tiempo asignado, y no se probaron tamaños mayores. Java y C completaron toda la
secuencia hasta **n = 2000** sin timeouts ni fallos de memoria. Python con `n = 1000` tiene 3
repeticiones en lugar de 5 por la regla de coste declarada (la primera tardó 184 s > 120 s); aun
así su IQR es del 1,1 % (180,8–182,9 s), así que la incertidumbre de la mediana es pequeña frente a
los efectos que se discuten.

### 3.3 Crecimiento frente al modelo cúbico

Pendiente del ajuste por mínimos cuadrados de log(tiempo) frente a log(n), ± error estándar:

| Tramo | Python | Java | C |
|---|---:|---:|---:|
| 100 ≤ n ≤ 600 | — | 3,39 ± 0,06 | **3,11 ± 0,01** |
| 100 ≤ n ≤ 800 | **3,16 ± 0,04** | — | — |
| 100 ≤ n ≤ 1000 | 3,32 ± 0,12 | 3,59 ± 0,10 | 3,19 ± 0,04 |
| 100 ≤ n ≤ 2000 | — | 3,57 ± 0,06 | 3,28 ± 0,05 |

Las tres pendientes están cerca de 3, pero todas son **algo mayores**, y la desviación no es
uniforme: se concentra en saltos concretos. La Figura 2 lo muestra dividiendo el tiempo entre n³
(nanosegundos por iteración interna), que sería una recta horizontal si el modelo fuese exacto:

<figure markdown="1">
![Tiempo / n³](../results/04_comparativa_3_lenguajes/plots/time_over_n3.png)
<figcaption>Figura 2. Coste por iteración interna (tiempo / n³) de cada implementación, con su
propia escala vertical.</figcaption>
</figure>

- **C:** ≈ 0,4–0,7 ns por iteración hasta `n = 600`, con una subida suave. Luego hay **dos
  escalones**: 0,72 → 0,92 ns entre `n = 600` y `800`, y 0,93 → 1,36 ns entre `n = 1200` y `1500`.
  Entre escalones el coste es casi constante (0,92–0,93 de 800 a 1200; 1,35–1,36 de 1500 a 2000).
- **Java:** ≈ 0,65–0,72 ns hasta `n = 200` (prácticamente igual que C), una subida a ≈ 1,0–1,2 ns
  entre 300 y 600, un **salto a ≈ 2,3–2,5 ns entre 600 y 800**, una meseta hasta 1500 y 2,9 ns con
  `n = 2000`.
- **Python:** ≈ 67–70 ns entre 50 y 300 (en `n = 10` el coste fijo de la llamada pesa más), una
  subida gradual hasta 96 ns con `n = 800` y un **salto a 182 ns con `n = 1000`** (exponente local
  5,8 entre 800 y 1000). Este salto ya apareció en la campaña piloto (137–186 s) y aquí se repite
  con 3 de 3 repeticiones entre 180 y 184 s, así que no es ruido.

### 3.4 Comparación entre lenguajes

<figure markdown="1">
![Factor de lentitud](../results/04_comparativa_3_lenguajes/plots/slowdown_vs_n.png)
<figcaption>Figura 3. Tiempo de Python y Java dividido entre el de C (-O2) para el mismo n.</figcaption>
</figure>

- **Python / C:** entre **102× y 117×** para `100 ≤ n ≤ 800`. Es mayor con `n` pequeño (186× con
  `n = 10`, porque C apenas tiene coste fijo por llamada) y con `n = 1000` (194×, por la anomalía de
  Python).
- **Java / C:** **1,1×** con `n = 100–200`, cuando todo cabe en la caché; entre **1,5× y 2,6×** con
  `n ≥ 300`. Con `n` muy pequeño (10–25) Java también es más lento (1,8–2,5×): la reserva de
  `double[n][n]` (n + 1 objetos) pesa frente a muy pocas operaciones.

### 3.5 Memoria

<figure markdown="1">
![Memoria](../results/04_comparativa_3_lenguajes/plots/memory_vs_n.png)
<figcaption>Figura 4. (a) Pico del working set del proceso completo, medido por el runner igual en
los tres lenguajes. (b) Memoria añadida dentro del proceso (pico tras multiplicar − valor antes de
cargar datos): working set en Python y C, heap usado en Java; con los modelos de almacenamiento.</figcaption>
</figure>

| n | Python | Java | C | 3·8n² |
|---:|---:|---:|---:|---:|
| 10 | 23,2 MB | 100,1 MB | 3,8 MB | 0,002 MB |
| 100 | 24,1 MB | 66,3 MB | 4,1 MB | 0,24 MB |
| 300 | 32,5 MB | 59,5 MB | 6,5 MB | 2,2 MB |
| 600 | 64,8 MB | 63,3 MB | 12,4 MB | 8,6 MB |
| 1000 | 143,2 MB | 78,6 MB | 27,7 MB | 24,0 MB |
| 2000 | — | 150,1 MB | 99,7 MB | 96,0 MB |

*Pico del working set del proceso completo (mediana de 5 procesos; MB = 10⁶ B).*

- **Coste fijo del runtime** (n pequeño): C ≈ 3,8 MB, Python ≈ 23 MB y la JVM ≈ 60 MB (mínimo, con
  `n = 300`). En Java el pico es *mayor* con `n = 10–50` (80–100 MB) que con `n = 300`. La causa
  son los lotes: con `n = 10` se ejecutan ≈ 48 000 multiplicaciones seguidas, cada una crea una `C`
  nueva y, con un heap fijo de 2 GB, el GC nunca llega a ejecutarse. La basura se acumula y toca
  páginas nuevas. El heap usado lo confirma: ≈ 46 MB por encima del inicio con `n = 10`, frente a
  11 MB con `n = 100`.
- **Parte que crece con n²**. En C, lo que añaden los datos coincide con el modelo casi
  exactamente: 96,2 MB medidos frente a 96,0 MB con `n = 2000`, y 24,2 frente a 24,0 con
  `n = 1000`. En Python, la memoria añadida es ≈ 1,25 veces su propio modelo de objetos: 121,6 MB
  frente a 96,2 MB con `n = 1000`. Eso es **unas 5 veces** lo que ocupa la misma matriz en C, por el
  objeto de 24 B (32 B en el asignador) más el puntero de 8 B por elemento, más el búfer temporal
  de lectura y la fragmentación del asignador. En Java, el crecimiento entre `n = 300` y
  `n = 2000` (+90,6 MB) coincide con el aumento del modelo `double[][]` (+93,9 MB), es decir, el
  mismo coste por elemento que C más una cabecera de 16 B por fila.
- Por tanto, **con `n ≤ 600` la memoria la domina el runtime** (Java > Python > C), mientras que
  **con `n` grande la domina la representación de los datos** (Python > Java ≈ C + JVM). Los dos
  órdenes se cruzan hacia `n ≈ 600`.

### 3.6 Experimentos aparte: compilador y JIT

Estos resultados son **separados** de la comparación controlada, como pide el enunciado: cambian los
ajustes de un lenguaje para entender su efecto. Se midieron en otra campaña
(`results/05_extra_variantes/`, 240 procesos, 5 repeticiones), intercalando cada variante con las
configuraciones base de C y Java como referencia. Las bases reproducen los valores de la campaña
principal con diferencias menores del 2 % (C con n = 1000: 0,925 s frente a 0,934 s; Java: 2,47 s
frente a 2,46 s). Todas las variantes producen la misma `C` bit a bit.

| n | C -O0 | C -O2 (base) | C -O3 -march=native | Java -Xint | Java, calentamiento 5×256³ | Java (base) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3,31 µs (7,4×) | 0,450 µs | 0,496 µs (1,10×) | 24,9 µs (55×) | 1,35 µs (3,0×) | 1,10 µs (2,4×) |
| 100 | 3,16 ms (5,2×) | 0,604 ms | 0,613 ms (1,01×) | 22,2 ms (37×) | 0,862 ms (1,4×) | 0,677 ms (1,1×) |
| 200 | 24,7 ms (4,7×) | 5,30 ms | 5,26 ms (0,99×) | 174 ms (33×) | 14,7 ms (2,8×) | 5,77 ms (1,1×) |
| 400 | 203 ms (4,5×) | 45,1 ms | 45,3 ms (1,00×) | 1,40 s (31×) | 169 ms (3,7×) | 69,0 ms (1,5×) |
| 1000 | 4,16 s (4,5×) | 0,925 s | 0,935 s (1,01×) | 25,5 s (28×) | 4,00 s (4,3×) | 2,47 s (2,7×) |

*Mediana del tiempo por multiplicación; entre paréntesis, el factor respecto a C -O2 medido en la
misma campaña. Tabla completa (n = 10, 50, 100, 200, 400, 600, 800, 1000) en
`results/05_extra_variantes/summary.md`.*

<figure markdown="1">
![Variantes](../results/05_extra_variantes/plots/slowdown_vs_n.png)
<figcaption>Figura 5. Factor respecto a C -O2 de cada variante (campaña 05).</figcaption>
</figure>

- **Optimización del compilador de C.** Sin optimizar (`-O0`), C es 4,5–7,4 veces más lento: guarda
  y recarga `i`, `j`, `k` y la suma en memoria en cada iteración. Con `-O3 -march=native` no gana
  **nada** respecto a `-O2` (±1 %). La razón es que el bucle interno es una reducción en coma
  flotante (`C[i][j] += …`) y el compilador no puede vectorizarla sin reordenar las sumas (lo que
  exigiría `-ffast-math` y cambiaría el resultado), ni tampoco cambiar el orden de los bucles. La
  ventaja de C en la comparación principal viene, por tanto, de compilar a código nativo, no de
  optimizaciones agresivas.
- **JIT de Java.** Con `-Xint` la JVM solo interpreta su bytecode y es 28–55 veces más lenta que C.
  Comparado con su propio JIT (Java base) eso es 10–34 veces: casi todo el rendimiento de Java en la
  comparación principal lo aporta el compilador C2. Aun así, el intérprete de la JVM (≈ 22 ns por
  iteración con n = 400) es unas 3,4 veces más rápido que CPython (≈ 73 ns): el bytecode de Java
  tiene tipos estáticos y opera con `double` sin envolver, mientras que CPython comprueba tipos y
  crea objetos `float` en cada operación.
- **Calentamiento de Java.** Con el calentamiento que se usó al principio (5 llamadas de 256×256),
  Java es 1,4–4,4 veces más lento que C, frente a 1,1–2,7 veces con el calentamiento definitivo. Con
  n = 1000, 4,00 s frente a 2,47 s (1,62×), el mismo efecto de desoptimización descrito en §2.6.
  Este experimento confirma que **la forma de calentar la JVM es un factor de primer orden**: un
  protocolo de calentamiento descuidado habría cambiado la conclusión de la comparación Java–C.

### 3.7 Experimento aparte: barrido fino de n en C y Java

Para localizar los escalones de la Figura 2 se midieron C y Java con n = 400, 450, …, 1600 (paso
50), más 512, 1024 y 1536 (`results/07_extra_barrido_fino/`, 168 procesos, 3 repeticiones, mismo
protocolo y verificación; `C` idéntica entre lenguajes en todos los tamaños). Los datos de entrada
se generan con el mismo generador y la misma regla de semillas.

<figure markdown="1">
![Barrido fino](../results/07_extra_barrido_fino/plots/sweep_ns_per_iter.png)
<figcaption>Figura 6. Coste por iteración interna de C y Java en el barrido fino. Las bandas
grises marcan los n múltiplos de 512.</figcaption>
</figure>

- **Potencias de dos.** Con n = 512, 1024 y 1536, C pasa a **5,5–6,2 ns por iteración**, 6–7 veces
  más que con sus vecinos (0,73 ns con n = 500, 0,96 ns con 1000). Java **no** muestra ese pico
  (1,3–2,7 ns), así que con n = 1024 Java es 2,3 veces *más rápido* que C, y con n = 1000 es 2,6
  veces *más lento*. La causa es la disposición en memoria (§4.3).
- **Escalones de C.** ≈ 0,71–0,81 ns hasta n = 750; ≈ 0,92–1,02 ns entre 800 y 1200; por encima de
  1250, valores irregulares entre 1,2 y 1,9 ns que no crecen de forma monótona con n.
- **Java** es más irregular entre procesos: rango de hasta el 27 % entre repeticiones para
  450 ≤ n ≤ 750, frente a < 10 % en C. Pasa de ≈ 1,2–1,4 ns (n ≤ 550) a ≈ 1,6–2,6 ns.

- **Java también depende del alineamiento.** Los n múltiplos de 100 (800, 900, …, 1600) cuestan
  sistemáticamente ≈ 2,4–2,6 ns, y los múltiplos impares de 50 (850, 950, …, 1550) solo ≈ 1,6–2,2 ns.
  La mayoría de los tamaños de la campaña principal son múltiplos de 100, así que el factor Java/C de
  §3.4 está en el lado alto del rango que se observa en el barrido (1,0–2,6× fuera de las potencias
  de dos).

### 3.8 Experimento aparte: el salto de Python con n ≈ 1000

Para explicar el salto de Python (182 ns por iteración con n = 1000, frente a 96 con n = 800) se
escribió `scripts/probe_python_layout.py`. Ejecuta el kernel de la campaña sobre las primeras filas
de C (las filas son independientes, así que el coste por iteración es representativo; ≈ 2·10⁷
iteraciones por medida, 3 repeticiones alternadas) con tres disposiciones de B que contienen
exactamente los mismos valores y dan la misma C bit a bit:

1. **por filas:** los objetos `float` se crean fila a fila, como al leer los ficheros en la campaña;
2. **por columnas:** los objetos `float` se crean columna a columna, de modo que `B[k][j]` y
   `B[k+1][j]` quedan contiguos en memoria (separados 32 B);
3. **diagnóstico, B traspuesta:** el kernel lee `Bt[j][k]`. Hace las mismas operaciones en el mismo
   orden, pero recorre la memoria de forma secuencial. Es solo un diagnóstico, no una
   implementación que se compare.

Se ejecutó con CPython 3.11.9 y 3.13.5 (`results/06_extra_python_layout/`):

| n | 3.11 por filas | 3.11 por columnas | 3.11 traspuesta | 3.13 por filas | 3.13 por columnas | 3.13 traspuesta |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 75 | 72 | 70 | 87 | 85 | 83 |
| 800 | 92 | 76 | 74 | 100 | 90 | 86 |
| 900 | 98 | 78 | 72 | 102 | 91 | 85 |
| 1000 | **173** | **124** | 71 | 109 | 93 | 87 |
| 1100 | 134 | 97 | 71 | 108 | 92 | 87 |
| 1200 | 121 | 91 | 76 | **159** | 89 | 84 |

*ns por iteración interna (mediana de 3). Archivo `diag_traspuesta.csv`. Una primera tanda sin la
variante traspuesta (`raw.csv`, también con n = 960, 1024 y 1040) dio el mismo patrón: en 3.11, 174 ns
por filas con n = 1000; en 3.13, ningún salto con n = 1000 y 139 ns con n = 1200.*

<figure markdown="1">
![Disposición de B en Python](../results/06_extra_python_layout/plots/python_layout_diag.png)
<figcaption>Figura 7. Coste por iteración del kernel de Python según la disposición de B, en
CPython 3.11.9 y 3.13.5.</figcaption>
</figure>

- **Con acceso secuencial (B traspuesta) el coste es constante**: 70–76 ns en 3.11 y 83–87 ns en
  3.13, para todos los n. El intérprete, por tanto, cuesta lo mismo por iteración con cualquier
  tamaño. El salto lo produce el **recorrido de B por columnas**.
- **No se debe a dónde caen los objetos `float`**. Ningún salto entre `B[k][j]` y `B[k+1][j]` es
  múltiplo de 4 KiB (0 % en todas las disposiciones), y con los `float` contiguos («por columnas»)
  el salto se reduce pero no desaparece (124 y 169 ns con n = 1000 en 3.11, en las dos tandas). Queda el otro acceso que
  también va a saltos en ambas disposiciones: el array de punteros de cada fila (`B[k]`), separado
  ≈ 8n bytes del siguiente.
- **Es errático y no es exclusivo de una versión.** En 3.11 aparece con n = 1000 (y de forma variable
  con n = 900: 152 ns en una tanda y 98 ns en otra); en 3.13 no aparece con n = 1000 pero sí con
  n = 1200 (159 ns). La magnitud depende de cómo coloca el asignador los miles de objetos y arrays,
  no solo de n. Con n = 1000, CPython 3.11 cae en el caso desfavorable de forma reproducible (3 de 3
  procesos en la campaña, entre 180 y 184 s). Por eso la campaña `03`, con 3.13, midió 104 s.

Una explicación compatible con estos datos, aunque **no verificada** con contadores hardware, es el
alcance de la TLB. Por filas, cada recorrido de columna toca ≈ 2n páginas distintas de 4 KiB (n de
objetos `float` y n de arrays de punteros), ≈ 2000 con n = 1000, que es del orden de las 2048
entradas de la TLB de segundo nivel de Zen 2. Al superarla, cada acceso puede requerir una consulta
de la tabla de páginas, y el intérprete, que tarda ≈ 70 ns por iteración, no puede solapar esas
esperas como hace el código nativo. Para la pregunta de este estudio basta lo que sí está medido: el
coste por iteración de Python depende fuertemente de la disposición en memoria cuando n ≳ 900, y
el exponente 5,8 entre n = 800 y 1000 no es una propiedad del algoritmo.


## 4. Discusión

### 4.1 Python: el coste lo pone el intérprete, no la aritmética (H2 se confirma)

Las tres implementaciones hacen las mismas operaciones de coma flotante, como prueba la igualdad bit
a bit (§3.1). La diferencia de tiempo es, por tanto, el coste de *ejecutar* esas operaciones. En
CPython 3.11, cada iteración interna `C[i][j] += A[i][k] * B[k][j]` son **25 instrucciones de
bytecode** (`dis`):

- 7 indexaciones de lista (`BINARY_SUBSCR` y `STORE_SUBSCR`);
- 2 operaciones (`BINARY_OP *` y `+=`), y cada una crea un objeto `float` nuevo en el heap;
- cargas de variables, `COPY`/`SWAP` y el salto del bucle;
- en cada acceso, actualización de contadores de referencias y comprobación de tipos.

Con ≈ 67–70 ns por iteración (n = 50–300) salen unos 2,7 ns, es decir ≈ 8–11 ciclos, por
instrucción de bytecode. C hace lo mismo en ≈ 0,4–0,7 ns (2–3 ciclos): unas pocas instrucciones de
máquina sobre `double` sin envolver. De ahí el factor ≈ 100: **constante con n** (102–117× entre
100 y 800), porque los dos lenguajes ejecutan las mismas n³ iteraciones y lo que cambia es el coste
fijo de cada una. Con n muy pequeño el factor sube (186×) porque C, con todo en L1, es todavía más
rápido por iteración.

La representación también influye: cada elemento de una lista es un puntero a un objeto `float`
separado (32 B en el asignador). Leer `B[k][j]` exige, por tanto, seguir dos punteros (la fila y el
objeto) en lugar de calcular una dirección, y A, B y C ocupan unas 5 veces más memoria que en C
(§3.5), lo que agota antes las cachés. Esto explica la subida gradual del coste por iteración de
Python a partir de n ≈ 300 (Figura 2): sus tres matrices ocupan ≈ 96n² B, unos 8,6 MB con
n = 300, más que los 4 MiB de L3 de un grupo de núcleos.

### 4.2 Java frente a C: el JIT iguala a C mientras los datos caben en caché (H3 se confirma con matices)

Con n = 100–200, cuando A, B y C caben en L2 o L3, Java tarda solo 1,1 veces lo que C: el código
que genera C2 para este bucle es tan bueno como el de `gcc -O2`. El experimento con `-Xint` (§3.6)
muestra que esa velocidad se debe por completo al JIT: sin él, Java es 10–34 veces más lento. La
diferencia aparece al crecer n (1,5–2,7×) y coincide con los escalones de memoria de §4.3. La
representación `double[][]` explica una parte plausible: para leer `B[k][j]` Java carga la
referencia a la fila `B[k]` y comprueba el índice contra `B[k].length`, un campo de la cabecera de
la fila que está en otra línea de caché. El recorrido por columnas toca así ≈ 2 líneas por
iteración en lugar de 1, lo que duplica la huella en caché y en la TLB. Además, las filas son
objetos que la JVM coloca en el heap de forma que cambia entre procesos, lo que encaja con la mayor
dispersión de Java con n = 400–750. Estas explicaciones son compatibles con los datos, pero no se
han medido con contadores hardware.

### 4.3 Desviaciones del modelo cúbico: la jerarquía de memoria (H1 se confirma con matices)

El trabajo es exactamente n³ iteraciones en los tres lenguajes (mismos bits en el resultado), así
que una pendiente mayor que 3 solo puede venir de que **cada iteración cuesta más al crecer n**.
El orden `i-j-k` explica por qué. La fila `A[i][·]` se recorre de forma secuencial y `C[i][j]` es
fija en el bucle interno, pero `B[0][j], B[1][j], …, B[n−1][j]` es un recorrido **por columnas**:
en C, cada acceso está 8n bytes más allá del anterior, en una línea de caché distinta y, con
n ≥ 512, en una página distinta. Las líneas se reaprovechan para las 7 columnas siguientes solo si
siguen en caché. Los datos encajan con la jerarquía del Ryzen 7 4800H:

- **Mientras todo cabe en caché** (C con n ≤ 600) el modelo cúbico se cumple bien: pendiente
  3,11 ± 0,01, con una subida suave del coste (0,4 → 0,7 ns), que el procesador oculta en gran
  parte ejecutando fuera de orden.
- **Primer escalón de C, entre n = 700 y 800**: B ocupa 8n² bytes, 3,9 MB con n = 700 y 5,1 MB con
  n = 800. Es justo donde B deja de caber en los **4 MiB de L3** de un grupo de núcleos. A partir de
  ahí, cada fila i de C obliga a traer B entera desde la DRAM.
- **Potencias de dos (n = 512, 1024, 1536)**: el salto entre filas, 8n bytes, es múltiplo de 4 KiB.
  En la L1 (64 conjuntos de 64 B: el conjunto lo fijan los bits 6–11 de la dirección) **todos** los
  elementos de una columna caen en el mismo conjunto, que solo tiene 8 vías. Con n = 1024 el salto
  es de 8 KiB y en la L2 (1024 conjuntos: periodo de 64 KiB) solo se usan 8 de los 1024 conjuntos.
  El resultado es un fallo de caché casi en cada acceso y un coste 6–7 veces mayor. Java se libra
  porque cada fila `double[]` empieza con una cabecera de 16 B: el salto real es 8n + 16 bytes, que
  no es múltiplo de 4 KiB, y los accesos se reparten entre los conjuntos. Es un ejemplo claro de que
  **la disposición en memoria puede pesar más que el lenguaje**: con n = 1024 el orden entre Java y
  C se invierte.
- **Con n > 1250 (C)** el coste es irregular (1,2–1,9 ns) y depende de n de forma no monótona. Es
  compatible con conflictos en la L2 o la TLB que dependen del alineamiento del salto (cada
  recorrido de columna toca n páginas distintas), más que con un límite de capacidad.
- **Python** acusa la memoria antes y de forma más gradual (67 → 96 ns entre n = 300 y 800). Sus
  matrices ocupan unas 5 veces más, y cada acceso sigue dos punteros (fila y objeto `float`). §3.8
  analiza el salto adicional de 3.11 con n ≈ 1000.

En resumen, el modelo `O(n³)` describe bien el **trabajo**, pero el **tiempo** solo lo sigue mientras
el coste por iteración es constante, es decir, mientras los datos se mantienen en el mismo nivel de
la jerarquía de memoria. Las pendientes de 3,2–3,6 medidas en el rango completo no indican un
algoritmo distinto: son la suma de escalones de la jerarquía de memoria sobre un crecimiento cúbico.

### 4.4 Memoria (H4 se confirma)

Los resultados separan con claridad las dos componentes previstas:

- **Coste fijo del runtime** (≈ 3,8 MB en C, ≈ 23 MB en CPython y ≈ 60 MB en la JVM con heap fijo de
  2 GB). Domina con n ≤ 600. Es memoria que el proceso usa para existir: el intérprete y sus módulos,
  la JVM con su código, el JIT, metaspace y las pilas de sus ≈ 20 hilos, o la CRT de C.
- **Coste proporcional a n²** de la representación elegida. El modelo predice C y Java casi
  exactamente (8 B por elemento, más 16 B por fila en Java). Python necesita ≈ 32 B por elemento más
  sobrecargas del asignador. Con n = 1000 son 121,6 MB frente a 24,2 MB en C (≈ 5×), que es justo la
  diferencia entre un `double` (8 B) y un puntero a un objeto `float` (8 + 32 B).

Dos matices sobre la medida. (1) El working set es memoria *residente*: no incluye memoria
reservada y nunca tocada (los 2 GB del heap de Java no aparecen). (2) En Java el pico con n pequeño
lo infla la basura de los lotes, porque con el heap fijo de 2 GB el GC no llega a ejecutarse. Es un
efecto real de esa configuración (la JVM cambia memoria por velocidad), no del algoritmo, y por eso
el mínimo de la JVM se toma con n = 300.


## 5. Amenazas a la validez y limitaciones

- **Una sola máquina, y portátil.** No se fijó la frecuencia de la CPU (Windows no lo permite sin
  herramientas del fabricante): el turbo y la temperatura cambian el coste por ciclo. Para
  mitigarlo se usó un proceso nuevo por medida, orden intercalado y rotado entre lenguajes, mediana
  e IQR, y el portátil estuvo enchufado. La dispersión resultante es baja (IQR < 2 % en 31 de 39
  configuraciones). Los valores absolutos y los factores entre lenguajes pueden cambiar en otra
  máquina o con otro ajuste de energía.
- **Ruido de fondo.** El sistema no estaba aislado: servicios de Windows, sincronización de OneDrive
  de la carpeta del proyecto y el propio runner, que muestrea los hilos del hijo. Afecta por igual a
  los tres lenguajes gracias al intercalado, pero puede explicar valores sueltos.
- **Lotes para n pequeño.** Con n ≤ 100 (Python), n ≤ 400 (Java) o n ≤ 500 (C) se mide un lote de multiplicaciones
  seguidas, con las cachés y el predictor de saltos calientes. Es el tiempo por multiplicación en
  régimen estable, no el de una llamada aislada.
- **Python con n = 1000: 3 repeticiones**, por la regla de coste declarada. Su IQR es del 1,1 % y el
  efecto que se discute es de 2×, mucho mayor que esa incertidumbre.
- **Métricas de memoria distintas.** El pico de working set del proceso es comparable entre
  lenguajes; la medida interna de Java (heap) no lo es y no se compara con las otras. El working set
  no cuenta memoria reservada sin tocar.
- **Ajustes fijados, no explorados.** Los factores de Java dependen de la JVM (GC, heap,
  calentamiento) y los de C del compilador y sus flags; §3.6 muestra lo grande que puede ser ese
  efecto. Se eligió una configuración razonable y documentada, no la mejor posible.
- **Entradas.** Valores uniformes en [0, 1), sin ceros ni números subnormales; con otros valores
  (p. ej. subnormales) los tiempos de coma flotante podrían cambiar.
- **Explicaciones de microarquitectura.** Los escalones de caché y la anomalía de Python 3.11 se
  interpretan a partir de los tiempos, los tamaños de las estructuras y experimentos dirigidos, pero
  sin contadores hardware (fallos de caché o de TLB). Esos contadores no están disponibles en esta
  máquina sin herramientas de AMD. Son interpretaciones compatibles con la evidencia, no
  mediciones directas.

## 6. Conclusiones

Las conclusiones se limitan a estas implementaciones (triple bucle `i-j-k`, `float64`), estos
ajustes (CPython 3.11.9; Temurin 17 con `-XX:+UseSerialGC -Xms2g -Xmx2g` y el calentamiento
descrito; gcc 11.2 `-O2 -ffp-contract=off`), estas entradas y este portátil.

1. **Corrección.** Las tres implementaciones calculan exactamente lo mismo: `C` idéntica bit a bit
   en todos los tamaños, a ≤ 5,6·10⁻¹⁵ de NumPy. Las diferencias de tiempo reflejan solo cómo
   ejecuta cada runtime el mismo cálculo.
2. **Python frente a C (H2).** Python es ≈ 100 veces más lento (102–117× para 100 ≤ n ≤ 800). La
   causa es el coste fijo de interpretar 25 instrucciones de bytecode con objetos `float` en cada
   iteración (≈ 70 ns), no la aritmética. Ese factor es estable mientras el acceso a memoria no
   domina.
3. **Java frente a C (H3).** Con el JIT bien calentado, Java está a 1,1× de C mientras los datos
   caben en caché y a 1,5–2,7× con n grande. No existe un orden universal ni siquiera entre estos dos:
   con n = 512, 1024 o 1536, C sufre conflictos de caché que Java evita gracias a las cabeceras de
   sus filas, y Java es hasta 2,3 veces más rápido.
4. **Crecimiento (H1).** El tiempo sigue aproximadamente n³ (pendiente 3,11 ± 0,01 en C mientras
   todo cabe en caché), pero en el rango completo crece algo más deprisa (3,2–3,6). No es un
   algoritmo distinto: el coste por iteración sube a escalones cuando `B`, recorrida por columnas,
   supera cada nivel de la jerarquía de memoria (L3 hacia n ≈ 724 en C). En Python, con n ≳ 900, el
   coste pasa a depender de la disposición en memoria de B (hasta 2× más).
5. **Memoria (H4).** Hay un coste fijo del runtime (C ≈ 3,8 MB, Python ≈ 23 MB, JVM ≈ 60 MB), que
   domina con n ≤ 600. La parte que crece con n² sigue el modelo de cada estructura: ≈ 24n² B en C y
   en Java, y unas 5 veces más en Python (objetos `float` y punteros).
6. **Límites.** Con 300 s por proceso, el mayor tamaño completado es n = 1000 en Python (timeout con
   n = 1200) y n = 2000 en Java (23 s) y C (11 s). No hubo fallos de memoria con el presupuesto de
   4096 MB: el pico máximo fue de 150 MB.
7. **Método.** Dos decisiones de protocolo cambian el resultado tanto como el lenguaje: el
   calentamiento de la JVM (1,6× con n = 1000) y la elección de los tamaños (las potencias de dos
   multiplican por 6–7 el tiempo de C). Ambas se detectaron gracias a validar el protocolo con
   diagnósticos (registro del JIT, barrido fino) antes de sacar conclusiones.

## 7. Reproducibilidad

Todo el estudio se regenera desde la carpeta `practica1/` (instrucciones completas, versiones y
esquema del CSV en `README.md`):

```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 scripts/run_benchmarks.py --out results/04_comparativa_3_lenguajes/raw.csv
py -3.11 scripts/run_benchmarks.py --configs c c-O0 c-O3native java java-warmup5x256 java-Xint \
    --sizes 10 50 100 200 400 600 800 1000 --out results/05_extra_variantes/raw.csv
py -3.11 scripts/run_benchmarks.py --configs c java --reps 3 --data-dir data_barrido \
    --sizes 400 450 500 512 550 600 650 700 750 800 850 900 950 1000 1024 1050 1100 1150 \
            1200 1250 1300 1350 1400 1450 1500 1536 1550 1600 \
    --out results/07_extra_barrido_fino/raw.csv
py -3.11 scripts/probe_python_layout.py --out results/06_extra_python_layout/raw.csv
py -3.13 scripts/probe_python_layout.py --out results/06_extra_python_layout/raw.csv
py -3.11 scripts/probe_python_layout.py --layouts filas columnas traspuesta \
    --sizes 400 600 800 900 1000 1100 1200 --out results/06_extra_python_layout/diag_traspuesta.csv
py -3.13 scripts/probe_python_layout.py --layouts filas columnas traspuesta \
    --sizes 400 600 800 900 1000 1100 1200 --out results/06_extra_python_layout/diag_traspuesta.csv
py -3.11 scripts/analyze.py --csv results/04_comparativa_3_lenguajes/raw.csv \
    --out-dir results/04_comparativa_3_lenguajes      # ídem para 05 y 07
py -3.11 scripts/plot_extras.py                       # figuras 6 y 7
```

El barrido fino usa su propia carpeta de datos (`data_barrido/`, ≈ 0,5 GB, no versionada), con las
mismas reglas de generación y de semillas, solo que con más tamaños.

| Elemento | Dónde |
|:---|:---|
| Código de los kernels | `src/python/matmul.py`, `src/java/MatMul.java`, `src/c/matmul.c` |
| Compilación y flags exactos | `scripts/build.py`, `build/build_info.json` |
| Entradas compartidas | `src/python/datagen.py` + `data/manifest.json` (semillas y SHA-256) |
| Tests de corrección (36) | `tests/test_matmul.py`, `tests/test_workers.py` |
| Runner (protocolo, presupuestos, verificación) | `scripts/run_benchmarks.py` |
| Medidas en bruto (una fila por proceso) | `results/*/raw.csv` |
| Entorno de cada campaña | `results/*/environment.md` y `.json` |
| Tablas y gráficas | `scripts/analyze.py` → `results/*/summary.md`, `summary.csv`, `plots/` |
| Experimento de disposición en memoria (Python) | `scripts/probe_python_layout.py` → `results/06_extra_python_layout/` |
| Figuras de los experimentos aparte | `scripts/plot_extras.py` → `results/06_*/plots/`, `results/07_*/plots/` |
| Este informe | `informe/informe.md` → `informe/informe.pdf` (`informe/build_pdf.py`) |

Cada fila de los CSV lleva el tamaño, el lenguaje, la variante, el tipo numérico, el número de
hilos, la repetición, el tamaño del lote y del calentamiento, la versión del runtime y sus flags.
Ninguna medida se ha eliminado. Las campañas `01` y `03` (solo Python, anteriores a este diseño)
se conservan como históricas; su relación con este estudio se explica en `results/LEEME.md`.
