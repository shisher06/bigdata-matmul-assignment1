# Resumen de resultados

Estadísticos sobre las ejecuciones con `status = OK`: mediana y rango intercuartílico (IQR = Q3 - Q1). Tiempos por multiplicación. Memoria en MB decimales (10⁶ B). Cada ejecución es un proceso nuevo.

## Comparación entre lenguajes (mediana del tiempo por multiplicación)

| n | Python | Java | C (-O2) | Python / C | Java / C |
|---:|---:|---:|---:|---:|---:|
| 10 | 82.2 µs | 1.09 µs | 0.441 µs | 186x | 2.46x |
| 25 | 1.13 ms | 11.1 µs | 6.34 µs | 178x | 1.75x |
| 50 | 8.53 ms | 81 µs | 59.5 µs | 143x | 1.36x |
| 100 | 67.6 ms | 661 µs | 599 µs | 113x | 1.1x |
| 200 | 539 ms | 5.74 ms | 5.28 ms | 102x | 1.09x |
| 300 | 1.899 s | 27.5 ms | 18.6 ms | 102x | 1.48x |
| 400 | 4.7 s | 70.6 ms | 45.1 ms | 104x | 1.56x |
| 500 | 9.867 s | 154 ms | 89.3 ms | 110x | 1.72x |
| 600 | 18.31 s | 259 ms | 156 ms | 117x | 1.66x |
| 800 | 49.4 s | 1.187 s | 472 ms | 105x | 2.51x |
| 1000 | 181.6 s | 2.457 s | 934 ms | 194x | 2.63x |
| 1200 | - | 4.196 s | 1.613 s | - | 2.6x |
| 1500 | - | 8.236 s | 4.597 s | - | 1.79x |
| 2000 | - | 23.45 s | 10.81 s | - | 2.17x |

«-» = sin medida válida para ese tamaño (ver *Límites*).

## Memoria: pico del working set del proceso completo (MB)

Medido por el runner con `GetProcessMemoryInfo` sobre el proceso hijo al terminar (misma herramienta y definición en los tres lenguajes; incluye intérprete/JVM).

| n | Python | Java | C (-O2) | modelo 3·8n² |
|---:|---:|---:|---:|---:|
| 10 | 23.16 | 100.1 | 3.76 | 0.0024 |
| 25 | 23.31 | 100.4 | 3.83 | 0.015 |
| 50 | 23.61 | 80.27 | 3.793 | 0.06 |
| 100 | 24.11 | 66.31 | 4.067 | 0.24 |
| 200 | 26.84 | 61.05 | 5.071 | 0.96 |
| 300 | 32.46 | 59.48 | 6.517 | 2.16 |
| 400 | 40.46 | 60.8 | 8.815 | 3.84 |
| 500 | 51.51 | 62.49 | 11.7 | 6 |
| 600 | 64.82 | 63.27 | 12.38 | 8.64 |
| 800 | 98.7 | 69.91 | 19.1 | 15.36 |
| 1000 | 143.2 | 78.58 | 27.74 | 24 |
| 1200 | - | 89.28 | 38.29 | 34.56 |
| 1500 | - | 107.8 | 57.73 | 54 |
| 2000 | - | 150.1 | 99.73 | 96 |

## Corrección

- Matriz C idéntica bit a bit (mismo SHA-256 del volcado) en todos los lenguajes medidos: n = 10, 25, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000.
- Error relativo máximo frente a NumPy (repetición 1, todas las configuraciones): 5.56e-15 (tolerancia rtol = 1e-9).
- Repeticiones 2..k: checksum igual al de la repetición 1 en todas las filas `OK`.

## Python - `python/base`

Runtime: CPython 3.11.9; flags: `-`; métrica interna de memoria: `peak_wset`. «calent.» = ejecuciones x tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno - base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 82.2 µs | 81.5 µs | 82.4 µs | 1.13 | 82.19 | 1231 | 2x10 | 23.16 | 21.7 | 0.02048 | 0.01183 | 4 |
| 25 | 5 | 1.13 ms | 1.12 ms | 1.13 ms | 0.752 | 72.08 | 93 | 2x25 | 23.31 | 21.84 | 0.1352 | 0.06514 | 4 |
| 50 | 5 | 8.53 ms | 8.53 ms | 8.54 ms | 0.128 | 68.26 | 12 | 2x50 | 23.61 | 22.14 | 0.5325 | 0.2498 | 4 |
| 100 | 5 | 67.6 ms | 67.4 ms | 68 ms | 0.842 | 67.58 | 2 | 2x64 | 24.11 | 23.67 | 1.999 | 0.9796 | 4 |
| 200 | 5 | 539 ms | 539 ms | 549 ms | 1.82 | 67.43 | 1 | 1x64 | 26.84 | 26.81 | 5.202 | 3.879 | 4 |
| 300 | 5 | 1.899 s | 1.894 s | 1.9 s | 0.316 | 70.35 | 1 | 1x64 | 32.46 | 32.46 | 10.85 | 8.698 | 4 |
| 400 | 5 | 4.7 s | 4.7 s | 4.7 s | 0.00795 | 73.44 | 1 | 1x64 | 40.46 | 40.39 | 18.73 | 15.44 | 4 |
| 500 | 5 | 9.867 s | 9.866 s | 9.869 s | 0.0207 | 78.93 | 1 | 1x64 | 51.51 | 51.49 | 29.88 | 24.1 | 4 |
| 600 | 5 | 18.31 s | 18.29 s | 18.31 s | 0.126 | 84.75 | 1 | 1x64 | 64.82 | 64.78 | 43.13 | 34.68 | 4 |
| 800 | 5 | 49.4 s | 49.04 s | 49.86 s | 1.66 | 96.48 | 1 | 1x64 | 98.7 | 98.68 | 77.04 | 61.6 | 6 |
| 1000 | 3 (!) | 181.6 s | 180.8 s | 182.9 s | 1.14 | 181.6 | 1 | 1x64 | 143.2 | 143.2 | 121.6 | 96.19 | 6 |

(!) = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1200**
- n = 1200, repetición 1: **timeout** (cancelado a los 300 s); límite de tiempo 300 s, presupuesto de memoria 4096 MB. Sin tiempo asignado.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n >= 100, 8 tamaños): **3.323** +- 0.119 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10 a 25: 2.86; 25 a 50: 2.92; 50 a 100: 2.99; 100 a 200: 3.00; 200 a 300: 3.10; 300 a 400: 3.15; 400 a 500: 3.32; 500 a 600: 3.39; 600 a 800: 3.45; 800 a 1000: 5.83.

## Java - `java/base`

Runtime: OpenJDK 64-Bit Server VM 17.0.20.1+1; flags: `-XX:+UseSerialGC -Xms2g -Xmx2g`; métrica interna de memoria: `heap_peak_used`. «calent.» = ejecuciones x tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno - base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 1.09 µs | 1.09 µs | 1.09 µs | 0.127 | 1.086 | 47620 | 201x64 | 100.1 | 103.1 | 45.81 | 0.003048 | 20 |
| 25 | 5 | 11.1 µs | 11.1 µs | 11.2 µs | 1.18 | 0.7111 | 8404 | 201x64 | 100.4 | 91.63 | 45.81 | 0.01655 | 20 |
| 50 | 5 | 81 µs | 80.5 µs | 81 µs | 0.592 | 0.6477 | 1247 | 201x64 | 80.27 | 80.17 | 34.36 | 0.06305 | 20 |
| 100 | 5 | 661 µs | 660 µs | 668 µs | 1.18 | 0.6613 | 145 | 201x64 | 66.31 | 57.27 | 11.45 | 0.246 | 20 |
| 200 | 5 | 5.74 ms | 5.73 ms | 5.75 ms | 0.276 | 0.7172 | 18 | 201x64 | 61.05 | 68.72 | 11.45 | 0.972 | 21 |
| 300 | 5 | 27.5 ms | 27.5 ms | 27.6 ms | 0.415 | 1.02 | 4 | 201x64 | 59.48 | 80.17 | 34.36 | 2.178 | 21 |
| 400 | 5 | 70.6 ms | 69.2 ms | 72.2 ms | 4.22 | 1.103 | 2 | 201x64 | 60.8 | 68.72 | 11.46 | 3.864 | 21 |
| 500 | 5 | 154 ms | 148 ms | 186 ms | 24.6 | 1.228 | 1 | 201x64 | 62.49 | 68.72 | 22.91 | 6.03 | 21 |
| 600 | 5 | 259 ms | 249 ms | 273 ms | 9.34 | 1.199 | 1 | 200x64 | 63.27 | 68.72 | 11.45 | 8.676 | 21 |
| 800 | 5 | 1.187 s | 1.185 s | 1.208 s | 1.97 | 2.319 | 1 | 200x64 | 69.91 | 91.64 | 45.83 | 15.41 | 21 |
| 1000 | 5 | 2.457 s | 2.436 s | 2.523 s | 3.56 | 2.457 | 1 | 200x64 | 78.58 | 91.63 | 34.37 | 24.06 | 21 |
| 1200 | 5 | 4.196 s | 4.185 s | 4.197 s | 0.283 | 2.428 | 1 | 200x64 | 89.28 | 91.65 | 45.84 | 34.63 | 21 |
| 1500 | 5 | 8.236 s | 8.159 s | 8.238 s | 0.953 | 2.44 | 1 | 200x64 | 107.8 | 114.6 | 68.78 | 54.09 | 21 |
| 2000 | 5 | 23.45 s | 23.09 s | 23.8 s | 3 | 2.931 | 1 | 200x64 | 150.1 | 171.9 | 126.1 | 96.12 | 22 |

(!) = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **2000**
- Mayor tamaño intentado: **2000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n >= 100, 11 tamaños): **3.570** +- 0.057 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10 a 25: 2.54; 25 a 50: 2.87; 50 a 100: 3.03; 100 a 200: 3.12; 200 a 300: 3.87; 300 a 400: 3.27; 400 a 500: 3.48; 500 a 600: 2.87; 600 a 800: 5.29; 800 a 1000: 3.26; 1000 a 1200: 2.94; 1200 a 1500: 3.02; 1500 a 2000: 3.64.

## C (-O2) - `c/base`

Runtime: gcc 11.2.0; flags: `-std=c11 -Wall -Wextra -ffp-contract=off -O2`; métrica interna de memoria: `peak_wset`. «calent.» = ejecuciones x tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno - base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 0.441 µs | 0.441 µs | 0.442 µs | 0.199 | 0.4411 | 166667 | 2x10 | 3.76 | 3.592 | 0.1065 | 0.0024 | 2 |
| 25 | 5 | 6.34 µs | 6.33 µs | 6.44 µs | 1.71 | 0.4055 | 15625 | 2x25 | 3.83 | 3.67 | 0.1802 | 0.015 | 2 |
| 50 | 5 | 59.5 µs | 59.4 µs | 59.6 µs | 0.422 | 0.4757 | 1553 | 2x50 | 3.793 | 3.69 | 0.2007 | 0.06 | 2 |
| 100 | 5 | 599 µs | 598 µs | 599 µs | 0.284 | 0.5988 | 162 | 2x64 | 4.067 | 3.903 | 0.4178 | 0.24 | 2 |
| 200 | 5 | 5.28 ms | 5.28 ms | 5.28 ms | 0.137 | 0.6595 | 20 | 2x64 | 5.071 | 4.911 | 1.421 | 0.96 | 2 |
| 300 | 5 | 18.6 ms | 18.5 ms | 18.6 ms | 0.559 | 0.6872 | 6 | 2x64 | 6.517 | 6.472 | 2.986 | 2.16 | 2 |
| 400 | 5 | 45.1 ms | 45 ms | 45.5 ms | 0.938 | 0.7052 | 3 | 2x64 | 8.815 | 8.815 | 5.325 | 3.84 | 2 |
| 500 | 5 | 89.3 ms | 88.6 ms | 89.4 ms | 0.853 | 0.7143 | 2 | 2x64 | 11.7 | 11.7 | 8.208 | 6 | 2 |
| 600 | 5 | 156 ms | 155 ms | 159 ms | 2.8 | 0.7233 | 1 | 1x64 | 12.38 | 12.34 | 8.847 | 8.64 | 2 |
| 800 | 5 | 472 ms | 471 ms | 482 ms | 2.53 | 0.9227 | 1 | 1x64 | 19.1 | 19.05 | 15.56 | 15.36 | 2 |
| 1000 | 5 | 934 ms | 933 ms | 972 ms | 4.13 | 0.9343 | 1 | 1x64 | 27.74 | 27.7 | 24.21 | 24 | 2 |
| 1200 | 5 | 1.613 s | 1.606 s | 1.617 s | 0.698 | 0.9333 | 1 | 1x64 | 38.29 | 38.25 | 34.76 | 34.56 | 2 |
| 1500 | 5 | 4.597 s | 4.561 s | 4.644 s | 1.8 | 1.362 | 1 | 1x64 | 57.73 | 57.69 | 54.2 | 54 | 2 |
| 2000 | 5 | 10.81 s | 10.76 s | 10.87 s | 0.989 | 1.351 | 1 | 1x64 | 99.73 | 99.69 | 96.2 | 96 | 2 |

(!) = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **2000**
- Mayor tamaño intentado: **2000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n >= 100, 11 tamaños): **3.279** +- 0.045 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10 a 25: 2.91; 25 a 50: 3.23; 50 a 100: 3.33; 100 a 200: 3.14; 200 a 300: 3.10; 300 a 400: 3.09; 400 a 500: 3.06; 500 a 600: 3.07; 600 a 800: 3.85; 800 a 1000: 3.06; 1000 a 1200: 2.99; 1200 a 1500: 4.69; 1500 a 2000: 2.97.
