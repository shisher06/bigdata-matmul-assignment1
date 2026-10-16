# Resumen de resultados

Estadísticos sobre las ejecuciones con `status = OK`: mediana y rango intercuartílico (IQR = Q3 - Q1). Tiempos por multiplicación. Memoria en MB decimales (10⁶ B). Cada ejecución es un proceso nuevo.

## Python - `python/base`

Runtime: 3.11.9; flags: `-`; métrica interna de memoria: `peak_wset`. «calent.» = ejecuciones x tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno - base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 81.6 µs | 81.3 µs | 81.8 µs | 0.637 | 81.58 | 1224 | 2x10 | - | 22.47 | 0.02458 | 0.01183 | - |
| 25 | 5 | 1.1 ms | 1.1 ms | 1.11 ms | 0.457 | 70.55 | 93 | 2x25 | - | 22.56 | 0.1556 | 0.06514 | - |
| 50 | 5 | 8.45 ms | 8.43 ms | 8.5 ms | 0.74 | 67.62 | 12 | 2x50 | - | 23 | 0.5448 | 0.2498 | - |
| 100 | 5 | 67 ms | 66.7 ms | 67.4 ms | 1.03 | 66.96 | 2 | 2x64 | - | 24.64 | 2.146 | 0.9796 | - |
| 200 | 5 | 540 ms | 536 ms | 542 ms | 1.11 | 67.47 | 1 | 1x64 | - | 27.62 | 5.194 | 3.879 | - |
| 300 | 5 | 1.871 s | 1.852 s | 1.873 s | 1.12 | 69.29 | 1 | 1x64 | - | 33.35 | 10.94 | 8.698 | - |
| 400 | 5 | 4.57 s | 4.547 s | 4.582 s | 0.78 | 71.4 | 1 | 1x64 | - | 41.12 | 18.73 | 15.44 | - |
| 500 | 5 | 9.98 s | 9.93 s | 10.02 s | 0.882 | 79.84 | 1 | 1x64 | - | 52.19 | 29.84 | 24.1 | - |
| 600 | 5 | 18.15 s | 18.1 s | 18.16 s | 0.34 | 84.05 | 1 | 1x64 | - | 65.5 | 42.91 | 34.68 | - |
| 800 | 5 | 46.01 s | 45.37 s | 46.33 s | 2.07 | 89.87 | 1 | 1x64 | - | 99.39 | 76.9 | 61.6 | - |
| 1000 | 3 (!) | 178.2 s | 157.6 s | 182 s | 13.7 | 178.2 | 1 | 1x64 | - | 144 | 121.5 | 96.19 | - |

(!) = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n >= 100, 8 tamaños): **3.310** +- 0.120 (error estándar); el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 10 a 25: 2.84; 25 a 50: 2.94; 50 a 100: 2.99; 100 a 200: 3.01; 200 a 300: 3.07; 300 a 400: 3.10; 400 a 500: 3.50; 500 a 600: 3.28; 600 a 800: 3.23; 800 a 1000: 6.07.
