# Resumen de resultados

Estadísticos sobre las ejecuciones con `status = OK`: mediana y rango intercuartílico (IQR = Q3 − Q1). Tiempos por multiplicación. Memoria en MB decimales (10⁶ B). Cada ejecución es un proceso nuevo.

## Python · `python/base`

Runtime: 3.13.5; flags: `—`; métrica interna de memoria: `unavailable`. «calent.» = ejecuciones × tamaño del calentamiento (incluye la calibración del lote).

| n | reps | mediana | Q1 | Q3 | IQR/med (%) | ns/n³ | lote | calent. | pico proceso (MB) | interno (MB) | interno − base (MB) | modelo almac. (MB) | hilos SO |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 900 | 5 | 69.36 s | 69.33 s | 70.37 s | 1.49 | 95.14 | 1 | 1×64 | — | — | — | 77.93 | — |
| 1000 | 5 | 104.4 s | 103.8 s | 104.6 s | 0.708 | 104.4 | 1 | 1×64 | — | — | — | 96.19 | — |

⚠ = menos de 5 repeticiones (justificado por coste; ver el informe).

### Límites

- Mayor tamaño probado dentro de los límites: **1000**
- Mayor tamaño intentado: **1000**
- Sin timeouts ni fallos de memoria.

### Crecimiento frente al modelo cúbico

- Pendiente log-log de la mediana (n ≥ 100, 2 tamaños): **3.878**; el modelo algorítmico predice 3.
- Exponente local entre tamaños consecutivos: 900→1000: 3.88.
