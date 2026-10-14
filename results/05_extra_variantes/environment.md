# Entorno de ejecución

- Fecha (UTC): 2026-10-14T17:29:52+00:00
- Sistema operativo: Windows 11 Home 26H2 (build 26300.9711) (platform: Windows-10-10.0.26300-SP0)
- CPU: AMD Ryzen 7 4800H with Radeon Graphics
- Núcleos físicos / lógicos: 8 / 16
- RAM total: 15.4 GiB
- Python: CPython 3.11.9 (MSC v.1938 64 bit (AMD64))
- GIL activado: None
- Temporizador: time.perf_counter_ns (monótono=True, resolución=1e-07 s)
- C: gcc.EXE (GCC) 11.2.0 (x86_64-w64-mingw32); flags: O2: `-std=c11 -Wall -Wextra -ffp-contract=off -O2`; O0: `-std=c11 -Wall -Wextra -ffp-contract=off -O0`; O3native: `-std=c11 -Wall -Wextra -ffp-contract=off -O3 -march=native` (O2 = comparación principal)
- Java: openjdk version "17.0.20.1" 2026-08-18 | OpenJDK Runtime Environment Temurin-17.0.20.1+1 (build 17.0.20.1+1) | OpenJDK 64-Bit Server VM Temurin-17.0.20.1+1 (build 17.0.20.1+1, mixed mode, sharing); javac 17.0.20.1; flags JVM: `-XX:+UseSerialGC -Xms2g -Xmx2g`
- numpy (solo verificación y análisis): 2.4.6
- matplotlib (solo gráficas): 3.11.2
- psutil (opcional): 7.2.2
- Variables de hilos en el shell: ninguna definida
- battery_percent: 95
- power_plugged: True
- cpu_freq_mhz_current: 2900.0
- cpu_freq_mhz_max: 2900.0

## Campaña

- Tamaños: [10, 50, 100, 200, 400, 600, 800, 1000]; repeticiones: 5; límite de tiempo: 300 s por proceso; presupuesto de memoria: 4096 MB por proceso
- `c`: `build/c/matmul.exe`
- `c-O0`: `build/c/matmul_O0.exe`
- `c-O3native`: `build/c/matmul_O3native.exe`
- `java`: `java.EXE -XX:+UseSerialGC -Xms2g -Xmx2g -cp build/java MatMul`
- `java-warmup5x256`: `java.EXE -XX:+UseSerialGC -Xms2g -Xmx2g -cp build/java MatMul --warmup-runs 5 --warmup-n 256`
- `java-Xint`: `java.EXE -XX:+UseSerialGC -Xms2g -Xmx2g -Xint -cp build/java MatMul --warmup-runs 1 --warmup-n 64`

> Los tres kernels (Python, Java, C) se ejecutan en un solo hilo (threads = 1). La JVM crea además hilos propios (JIT, GC, señales...): su número queda en os_threads_max / jvm_threads del CSV.
> El runner fija OMP/OPENBLAS/MKL_NUM_THREADS=1 en los procesos hijo; NumPy solo se usa para verificar, fuera del cronómetro.
