# Carpetas de resultados

| Carpeta | Qué es | Uso |
|---|---|---|
| `04_comparativa_3_lenguajes` | **Campaña principal.** Python 3.11.9, Java 17 y C (gcc -O2), n = 10 a 2000, 5 repeticiones, un proceso por repetición, lenguajes intercalados en la misma sesión. Incluye `campaign.log`. | **Resultados del informe.** |
| `05_extra_variantes` | Experimentos aparte con flags: C `-O0` y `-O3 -march=native`; Java `-Xint` (sin JIT) y con el calentamiento antiguo (5 ejecuciones de 256x256). Incluye las configuraciones base de C y Java intercaladas como referencia. n = 10 a 1000. | Discusión de efectos de compilador y runtime. **No se mezcla** con la comparación principal. |
| `07_extra_barrido_fino` | C y Java con n = 400...1600 en pasos de 50, más 512, 1024 y 1536; 3 repeticiones. Localiza los escalones de caché y el efecto de las potencias de dos. Sus datos de entrada se generan en `data_barrido/` (no versionado). | Informe, apartado 4.7. |
| `01_oficial_py311` | Primera campaña, solo Python 3.11, n = 10 a 1000. En n = 1000 solo hay 3 repeticiones (137, 178 y 186 s; el coste por iteración es el doble que en n = 800). Ese salto se repite en `04` (informe, apartados 4.3 y 5.1). | Histórico / piloto. Sustituida por `04`. |

La numeración no es continua: no hay carpetas `02`, `03` ni `06`.

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
