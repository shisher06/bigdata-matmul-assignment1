# Carpetas de resultados

| Carpeta | Qué es | Uso |
|---|---|---|
| `04_comparativa_3_lenguajes` | **Campaña principal.** Python 3.11.9, Java 17 y C (gcc -O2), n = 10 a 2000, 5 repeticiones, un proceso por repetición, lenguajes intercalados en la misma sesión. Incluye `campaign.log`. | **Resultados del informe.** |
| `05_extra_variantes` | Experimentos aparte con flags: C `-O0` y `-O3 -march=native`; Java `-Xint` (sin JIT) y con el calentamiento antiguo (5 × 256³). Incluye las configuraciones base de C y Java intercaladas como referencia. n = 10 a 1000. | Discusión de efectos de compilador y runtime. **No se mezcla** con la comparación principal. |
| `06_extra_python_layout` | Kernel de Python sobre las primeras filas de C con tres disposiciones de B (por filas, `float` creados por columnas, y B traspuesta como diagnóstico), en CPython 3.11.9 y 3.13.5. Explica el salto de Python con n ≈ 1000. `raw.csv` (1.ª tanda), `diag_traspuesta.csv` (2.ª tanda con la traspuesta), logs y `plots/`. | Informe §3.8. |
| `07_extra_barrido_fino` | C y Java con n = 400…1600 en pasos de 50, más 512, 1024 y 1536; 3 repeticiones. Localiza los escalones de caché y el efecto de las potencias de dos. Sus datos de entrada se generan en `data_barrido/` (no versionado). | Informe §3.7. |
| `01_oficial_py311` | Primera campaña, solo Python 3.11, n = 10 a 1000. En n = 1000 solo hay 3 repeticiones y son inestables (137 → 178 → 186 s; ~2× el coste por iteración de n = 800), probablemente por ruido externo o temperatura. | Histórico / piloto. Sustituida por `04`. |
| `03_extra_py313_n900_n1000` | Comprobación de n = 900 y 1000 con Python 3.13 (sin `psutil`: memoria `-1`, entorno incompleto). Mostró que n = 1000 tarda ~104 s de forma estable, es decir, que la lentitud de `01` era ruido. | Histórico. No mezclar: otra versión de Python. |

La numeración salta de `01` a `03`: no hay carpeta `02`.

Cada carpeta tiene su `raw.csv` (medidas en bruto, una fila por proceso), `environment.md/.json`
(máquina, versiones, flags) y, una vez analizada, `summary.csv`, `summary.md` y `plots/`.
Para volver a generar tablas y gráficas:

```
py -3.11 scripts/analyze.py --csv results/04_comparativa_3_lenguajes/raw.csv --out-dir results/04_comparativa_3_lenguajes
py -3.11 scripts/analyze.py --csv results/05_extra_variantes/raw.csv --out-dir results/05_extra_variantes
py -3.11 scripts/analyze.py --csv results/07_extra_barrido_fino/raw.csv --out-dir results/07_extra_barrido_fino
py -3.11 scripts/analyze.py --csv results/01_oficial_py311/raw.csv --out-dir results/01_oficial_py311
py -3.11 scripts/plot_extras.py
```
