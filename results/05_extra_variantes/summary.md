# Resumen de resultados

Estadísticos sobre las ejecuciones con `status = OK`: mediana y rango intercuartílico (IQR = Q3 − Q1). Tiempos por multiplicación. Memoria en MB decimales (10⁶ B). Cada ejecución es un proceso nuevo.

## Comparación entre lenguajes (mediana del tiempo por multiplicación)

| n | Java | C (-O2) | Java / C |
|---:|---:|---:|---:|
| 10 | 1.1 µs | 0.45 µs | 2.45× |
| 50 | 83.5 µs | 60.2 µs | 1.39× |
| 100 | 676 µs | 604 µs | 1.12× |
| 200 | 5.77 ms | 5.3 ms | 1.09× |
| 400 | 69 ms | 45.1 ms | 1.53× |
| 600 | 254 ms | 155 ms | 1.64× |
| 800 | 1.16 s | 452 ms | 2.57× |
| 1000 | 2.471 s | 925 ms | 2.67× |

«—» = sin medida válida para ese tamaño (ver *Límites*).

## Memoria: pico del working set del proceso completo (MB)

Medido por el runner con `GetProcessMemoryInfo` sobre el proceso hijo al terminar (misma herramienta y definición en los tres lenguajes; incluye intérprete/JVM).

| n | Java | C (-O2) | modelo 3·8n² |
|---:|---:|---:|---:|
| 10 | 102.2 | 3.76 | 0.0024 |
| 50 | 79.87 | 3.793 | 0.06 |
| 100 | 66.31 | 4.067 | 0.24 |
| 200 | 61.04 | 5.071 | 0.96 |
| 400 | 60.82 | 8.81 | 3.84 |
| 600 | 63.25 | 12.38 | 8.64 |
| 800 | 69.87 | 19.1 | 15.36 |
| 1000 | 78.72 | 27.74 | 24 |

## Corrección

- Matriz C idéntica bit a bit (mismo SHA-256 del volcado) en todos los lenguajes medidos: n = 10, 50, 100, 200, 400, 600, 800, 1000.
- Error relativo máximo frente a NumPy (repetición 1, todas las configuraciones): 4.19e-15 (tolerancia rtol = 1e-9).
- Repeticiones 2..k: checksum igual al de la repetición 1 en todas las filas `OK`.

## Java · `java/base`

Runtime: OpenJDK 64-Bit Server VM 17.0.20.1+1; flags: `-XX:+UseSerialGC -Xms2g -Xmx2g`; métrica interna de memoria: `heap_peak_used`. «calent.» = ejecuciones × tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno − base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 1.1 µs | 1.09 µs | 1.12 µs | 2.97 | 1.102 | 47620 | 201×64 | 102.2 | 91.63 | 45.81 | 0.003048 | 20 |
| 50 | 5 | 83.5 µs | 81.3 µs | 83.9 µs | 3.15 | 0.6678 | 1222 | 201×64 | 79.87 | 80.17 | 22.91 | 0.06305 | 20 |
| 100 | 5 | 676 µs | 674 µs | 680 µs | 0.925 | 0.6765 | 145 | 201×64 | 66.31 | 68.72 | 11.45 | 0.246 | 20 |
| 200 | 5 | 5.77 ms | 5.75 ms | 5.79 ms | 0.679 | 0.7216 | 18 | 201×64 | 61.04 | 68.72 | 22.91 | 0.972 | 21 |
| 400 | 5 | 69 ms | 68.9 ms | 69.4 ms | 0.79 | 1.079 | 2 | 201×64 | 60.82 | 68.72 | 22.91 | 3.864 | 21 |
| 600 | 5 | 254 ms | 248 ms | 256 ms | 2.87 | 1.176 | 1 | 200×64 | 63.25 | 80.17 | 22.91 | 8.676 | 21 |
| 800 | 5 | 1.16 s | 1.152 s | 1.175 s | 2.01 | 2.265 | 1 | 200×64 | 69.87 | 91.64 | 34.37 | 15.41 | 21 |
| 1000 | 5 | 2.471 s | 2.412 s | 2.568 s | 6.31 | 2.471 | 1 | 200×64 | 78.72 | 80.18 | 34.37 | 24.06 | 21 |

⚠ = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n ≥ 100, 6 tamaños): **3.577** ± 0.121 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10→50: 2.69; 50→100: 3.02; 100→200: 3.09; 200→400: 3.58; 400→600: 3.21; 600→800: 5.28; 800→1000: 3.39.

## C (-O2) · `c/base`

Runtime: gcc 11.2.0; flags: `-std=c11 -Wall -Wextra -ffp-contract=off -O2`; métrica interna de memoria: `peak_wset`. «calent.» = ejecuciones × tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno − base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 0.45 µs | 0.445 µs | 0.46 µs | 3.26 | 0.4502 | 200000 | 2×10 | 3.76 | 3.596 | 0.1065 | 0.0024 | 2 |
| 50 | 5 | 60.2 µs | 59.7 µs | 61.1 µs | 2.26 | 0.4815 | 1544 | 2×50 | 3.793 | 3.686 | 0.2007 | 0.06 | 2 |
| 100 | 5 | 604 µs | 602 µs | 611 µs | 1.55 | 0.604 | 137 | 2×64 | 4.067 | 3.908 | 0.4178 | 0.24 | 2 |
| 200 | 5 | 5.3 ms | 5.27 ms | 5.35 ms | 1.43 | 0.6621 | 19 | 2×64 | 5.071 | 4.911 | 1.421 | 0.96 | 2 |
| 400 | 5 | 45.1 ms | 45 ms | 45.3 ms | 0.565 | 0.7049 | 3 | 2×64 | 8.81 | 8.81 | 5.325 | 3.84 | 2 |
| 600 | 5 | 155 ms | 154 ms | 155 ms | 0.454 | 0.7165 | 1 | 1×64 | 12.38 | 12.34 | 8.847 | 8.64 | 2 |
| 800 | 5 | 452 ms | 444 ms | 456 ms | 2.68 | 0.883 | 1 | 1×64 | 19.1 | 19.05 | 15.56 | 15.36 | 2 |
| 1000 | 5 | 925 ms | 925 ms | 929 ms | 0.494 | 0.9251 | 1 | 1×64 | 27.74 | 27.7 | 24.21 | 24 | 2 |

⚠ = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n ≥ 100, 6 tamaños): **3.173** ± 0.037 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10→50: 3.04; 50→100: 3.33; 100→200: 3.13; 200→400: 3.09; 400→600: 3.04; 600→800: 3.73; 800→1000: 3.21.

## C (-O0) · `c/O0`

Runtime: gcc 11.2.0; flags: `-std=c11 -Wall -Wextra -ffp-contract=off -O0`; métrica interna de memoria: `peak_wset`. «calent.» = ejecuciones × tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno − base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 3.31 µs | 3.21 µs | 3.35 µs | 4.07 | 3.314 | 30304 | 2×10 | 3.764 | 3.6 | 0.1065 | 0.0024 | 2 |
| 50 | 5 | 375 µs | 373 µs | 379 µs | 1.64 | 2.997 | 264 | 2×50 | 3.797 | 3.695 | 0.2007 | 0.06 | 2 |
| 100 | 5 | 3.16 ms | 3.1 ms | 3.18 ms | 2.45 | 3.162 | 33 | 2×64 | 4.071 | 3.912 | 0.4178 | 0.24 | 2 |
| 200 | 5 | 24.7 ms | 24.7 ms | 25.5 ms | 3.23 | 3.085 | 4 | 2×64 | 5.075 | 4.915 | 1.421 | 0.96 | 2 |
| 400 | 5 | 203 ms | 202 ms | 204 ms | 1.01 | 3.177 | 1 | 1×64 | 7.578 | 7.537 | 4.043 | 3.84 | 2 |
| 600 | 5 | 713 ms | 706 ms | 713 ms | 0.992 | 3.303 | 1 | 1×64 | 12.38 | 12.34 | 8.847 | 8.64 | 2 |
| 800 | 5 | 2.095 s | 2.065 s | 2.116 s | 2.41 | 4.092 | 1 | 1×64 | 19.1 | 19.06 | 15.56 | 15.36 | 2 |
| 1000 | 5 | 4.162 s | 4.147 s | 4.176 s | 0.697 | 4.162 | 1 | 1×64 | 27.74 | 27.7 | 24.21 | 24 | 2 |

⚠ = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n ≥ 100, 6 tamaños): **3.121** ± 0.048 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10→50: 2.94; 50→100: 3.08; 100→200: 2.96; 200→400: 3.04; 400→600: 3.10; 600→800: 3.74; 800→1000: 3.08.

## C (-O3 -march=native) · `c/O3native`

Runtime: gcc 11.2.0; flags: `-std=c11 -Wall -Wextra -ffp-contract=off -O3 -march=native`; métrica interna de memoria: `peak_wset`. «calent.» = ejecuciones × tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno − base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 0.496 µs | 0.494 µs | 0.499 µs | 0.835 | 0.4962 | 166667 | 2×10 | 3.76 | 3.596 | 0.1065 | 0.0024 | 2 |
| 50 | 5 | 60.4 µs | 60.2 µs | 60.5 µs | 0.587 | 0.4831 | 1513 | 2×50 | 3.793 | 3.686 | 0.2007 | 0.06 | 2 |
| 100 | 5 | 613 µs | 612 µs | 614 µs | 0.291 | 0.6129 | 132 | 2×64 | 4.067 | 3.908 | 0.4178 | 0.24 | 2 |
| 200 | 5 | 5.26 ms | 5.26 ms | 5.31 ms | 1.03 | 0.6579 | 19 | 2×64 | 5.071 | 4.911 | 1.421 | 0.96 | 2 |
| 400 | 5 | 45.3 ms | 45.2 ms | 45.3 ms | 0.217 | 0.7075 | 3 | 2×64 | 8.815 | 8.815 | 5.325 | 3.84 | 2 |
| 600 | 5 | 156 ms | 156 ms | 156 ms | 0.318 | 0.7205 | 1 | 1×64 | 12.38 | 12.34 | 8.847 | 8.64 | 2 |
| 800 | 5 | 458 ms | 458 ms | 473 ms | 3.24 | 0.8944 | 1 | 1×64 | 19.1 | 19.05 | 15.56 | 15.36 | 2 |
| 1000 | 5 | 935 ms | 922 ms | 942 ms | 2.17 | 0.935 | 1 | 1×64 | 27.74 | 27.7 | 24.21 | 24 | 2 |

⚠ = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n ≥ 100, 6 tamaños): **3.175** ± 0.039 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10→50: 2.98; 50→100: 3.34; 100→200: 3.10; 200→400: 3.10; 400→600: 3.05; 600→800: 3.75; 800→1000: 3.20.

## Java (-Xint, no JIT) · `java/Xint`

Runtime: OpenJDK 64-Bit Server VM 17.0.20.1+1; flags: `-XX:+UseSerialGC -Xms2g -Xmx2g -Xint`; métrica interna de memoria: `heap_peak_used`. «calent.» = ejecuciones × tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno − base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 24.9 µs | 24.7 µs | 25.3 µs | 2.45 | 24.91 | 3522 | 2×64 | 44.28 | 22.91 | 0 | 0.003048 | 15 |
| 50 | 5 | 2.85 ms | 2.82 ms | 2.89 ms | 2.58 | 22.82 | 37 | 2×64 | 41.61 | 22.91 | 0 | 0.06305 | 15 |
| 100 | 5 | 22.2 ms | 22.1 ms | 22.4 ms | 1.41 | 22.2 | 5 | 2×64 | 41.47 | 22.91 | 0 | 0.246 | 15 |
| 200 | 5 | 174 ms | 173 ms | 174 ms | 0.339 | 21.71 | 1 | 1×64 | 41.82 | 22.91 | 0 | 0.972 | 15 |
| 400 | 5 | 1.397 s | 1.394 s | 1.418 s | 1.7 | 21.83 | 1 | 1×64 | 44.77 | 22.91 | 0 | 3.864 | 15 |
| 600 | 5 | 5.102 s | 5.078 s | 5.119 s | 0.813 | 23.62 | 1 | 1×64 | 49.63 | 22.91 | 0 | 8.676 | 15 |
| 800 | 5 | 12.85 s | 12.79 s | 12.89 s | 0.763 | 25.09 | 1 | 1×64 | 56.36 | 34.37 | 11.46 | 15.41 | 15 |
| 1000 | 5 | 25.46 s | 25.16 s | 25.59 s | 1.69 | 25.46 | 1 | 1×64 | 65.06 | 45.83 | 22.92 | 24.06 | 15 |

⚠ = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n ≥ 100, 6 tamaños): **3.065** ± 0.023 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10→50: 2.95; 50→100: 2.96; 100→200: 2.97; 200→400: 3.01; 400→600: 3.19; 600→800: 3.21; 800→1000: 3.07.

## Java (warm-up 5×256³) · `java/warmup5x256`

Runtime: OpenJDK 64-Bit Server VM 17.0.20.1+1; flags: `-XX:+UseSerialGC -Xms2g -Xmx2g`; métrica interna de memoria: `heap_peak_used`. «calent.» = ejecuciones × tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno − base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 1.35 µs | 1.33 µs | 1.35 µs | 1.62 | 1.351 | 10990 | 6×256 | 63.48 | 57.27 | 11.45 | 0.003048 | 21 |
| 50 | 5 | 96.5 µs | 94.3 µs | 96.6 µs | 2.38 | 0.7723 | 262 | 6×256 | 56.36 | 45.81 | 0 | 0.06305 | 21 |
| 100 | 5 | 862 µs | 853 µs | 868 µs | 1.75 | 0.8617 | 39 | 6×256 | 54.42 | 45.81 | 0 | 0.246 | 20 |
| 200 | 5 | 14.7 ms | 14.6 ms | 16.7 ms | 14.3 | 1.831 | 5 | 6×256 | 53.71 | 57.27 | 11.45 | 0.972 | 21 |
| 400 | 5 | 169 ms | 167 ms | 169 ms | 1.1 | 2.634 | 1 | 6×256 | 55.15 | 57.27 | 8.8e-05 | 3.864 | 21 |
| 600 | 5 | 623 ms | 617 ms | 633 ms | 2.54 | 2.882 | 1 | 5×256 | 60.12 | 68.72 | 22.91 | 8.676 | 21 |
| 800 | 5 | 2.007 s | 1.995 s | 2.028 s | 1.66 | 3.921 | 1 | 5×256 | 66.79 | 57.27 | 11.46 | 15.41 | 21 |
| 1000 | 5 | 4.003 s | 3.986 s | 4.017 s | 0.77 | 4.003 | 1 | 5×256 | 75.62 | 80.19 | 34.38 | 24.06 | 21 |

⚠ = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n ≥ 100, 6 tamaños): **3.641** ± 0.068 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10→50: 2.65; 50→100: 3.16; 100→200: 4.09; 200→400: 3.52; 400→600: 3.22; 600→800: 4.07; 800→1000: 3.09.
