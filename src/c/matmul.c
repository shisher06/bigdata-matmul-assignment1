/*
 * Kernel de la Práctica 1 (C) y *worker* que hace UNA medición.
 *
 * Algoritmo: multiplicación densa ingenua, triple bucle en orden i-j-k:
 *
 *     for i: for j: for k: C[i][j] += A[i][k] * B[k][j]
 *
 * Sin BLAS ni ninguna multiplicación de librería (trabajo O(n³)).
 * Estructura de datos: UN bloque contiguo de n·n doubles por matriz, orden por filas
 * (row-major). Se accede como matriz 2-D mediante un puntero a array de longitud
 * variable (C99), así la sentencia del bucle es la misma que en Python y Java.
 *
 * Qué cronometra el worker (medición del *kernel*), igual que en Python y Java:
 *   DENTRO: la llamada a matmul_ijk (reserva e inicializa C a 0.0 y ejecuta los bucles).
 *   FUERA:  lectura de ficheros, calentamiento, calibración, checksum, volcado de C.
 *   Temporizador monótono: QueryPerformanceCounter (Windows) o
 *   clock_gettime(CLOCK_MONOTONIC) (POSIX).
 *
 * Calentamiento: --warmup-runs ejecuciones del kernel sobre una submatriz de
 * --warmup-n x --warmup-n (o de n si es menor). Nunca forman parte de la muestra.
 *
 * Tareas muy cortas: si el coste estimado de una multiplicación es menor que
 * --batch-target-ms se hace una ejecución de calibración (se descarta y cuenta como
 * un calentamiento más) y se mide un LOTE dentro de una sola región cronometrada.
 * time_ns es siempre el tiempo POR multiplicación (time_total_ns / batch).
 *
 * Memoria: pico del working set del proceso (PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize,
 * el mismo contador que psutil llama peak_wset) en Windows; VmHWM en Linux; ru_maxrss
 * en otros Unix. Se lee justo después de multiplicar, como en Python.
 *
 * Checksum: suma correctamente redondeada de todos los elementos de C (algoritmo de
 * Shewchuk, el mismo que math.fsum de CPython), así que coincide bit a bit con el de
 * Python si C coincide bit a bit.
 *
 * Salida: la última línea de stdout es un objeto JSON. Códigos de salida:
 * 0 = OK, 2 = error de entrada/salida o de argumentos, 3 = memoria insuficiente.
 *
 * Compilación (la hace scripts/build.py):
 *   gcc -std=c11 -O2 -ffp-contract=off -o build/c/matmul.exe src/c/matmul.c -lpsapi
 *
 * Uso:
 *   build/c/matmul.exe --n 200 --a data/A_200.bin --b data/B_200.bin
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <time.h>
#endif

#define LANGUAGE "c"
#define METHOD "triple_loop_ijk"
#define DTYPE "float64"

#ifndef BUILD_FLAGS
#define BUILD_FLAGS "desconocidos"
#endif

#define EXIT_IO 2
#define EXIT_OOM 3

/* ------------------------------------------------------------------------- */
/* Temporizador monótono y memoria                                           */
/* ------------------------------------------------------------------------- */
static int64_t now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    int64_t q = t.QuadPart, f = freq.QuadPart;
    return (q / f) * 1000000000LL + (q % f) * 1000000000LL / f;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

/* Pico de memoria residente del proceso, en bytes; *metric recibe el nombre de la fuente. */
static int64_t peak_rss_bytes(const char **metric) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        *metric = "peak_wset";
        return (int64_t)pmc.PeakWorkingSetSize;
    }
#else
    FILE *fh = fopen("/proc/self/status", "r");
    if (fh) {
        char line[256];
        long long kb = -1;
        while (fgets(line, sizeof line, fh)) {
            if (strncmp(line, "VmHWM:", 6) == 0) {
                kb = atoll(line + 6);
                break;
            }
        }
        fclose(fh);
        if (kb >= 0) {
            *metric = "VmHWM";
            return (int64_t)kb * 1024;
        }
    }
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        *metric = "ru_maxrss";
#ifdef __APPLE__
        return (int64_t)ru.ru_maxrss;
#else
        return (int64_t)ru.ru_maxrss * 1024;
#endif
    }
#endif
    *metric = "unavailable";
    return -1;
}

/* ------------------------------------------------------------------------- */
/* Kernel                                                                     */
/* ------------------------------------------------------------------------- */
/* C = A x B (n x n), triple bucle i-j-k. Reserva C y la inicializa a 0.0.
 * noinline: el kernel queda como una función real (como en Python y Java) y el
 * compilador no puede fusionarlo con el código que lo rodea. */
__attribute__((noinline)) static double *matmul_ijk(const double *a_, const double *b_, int n) {
    double *c_ = malloc(sizeof(double) * (size_t)n * (size_t)n);
    if (c_ == NULL) return NULL;
    const double (*A)[n] = (const double (*)[n])a_;
    const double (*B)[n] = (const double (*)[n])b_;
    double (*C)[n] = (double (*)[n])c_;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return c_;
}

/* Suma correctamente redondeada (Shewchuk), traducción de math.fsum de CPython
 * para valores finitos. */
static double checksum_fsum(const double *x, size_t len) {
    double p[128];
    int np = 0;
    for (size_t idx = 0; idx < len; idx++) {
        double v = x[idx];
        int i = 0;
        for (int j = 0; j < np; j++) {
            double y = p[j];
            if (fabs(v) < fabs(y)) {
                double t = v;
                v = y;
                y = t;
            }
            double hi = v + y;
            double lo = y - (hi - v);
            if (lo != 0.0) p[i++] = lo;
            v = hi;
        }
        np = i;
        if (v != 0.0) {
            if (np >= (int)(sizeof p / sizeof p[0])) return NAN; /* imposible con doubles finitos */
            p[np++] = v;
        }
    }
    double hi = 0.0, lo = 0.0;
    if (np > 0) {
        hi = p[--np];
        while (np > 0) {
            double v = hi, y = p[--np];
            hi = v + y;
            lo = y - (hi - v);
            if (lo != 0.0) break;
        }
        if (np > 0 && ((lo < 0.0 && p[np - 1] < 0.0) || (lo > 0.0 && p[np - 1] > 0.0))) {
            double y = lo * 2.0;
            double v = hi + y;
            if (y == v - hi) hi = v;
        }
    }
    return hi;
}

/* ------------------------------------------------------------------------- */
/* E/S en el formato compartido (float64 little-endian, row-major, sin cabecera) */
/* ------------------------------------------------------------------------- */
static int host_is_little_endian(void) {
    const uint16_t one = 1;
    return *(const uint8_t *)&one == 1;
}

static void byteswap_all(double *v, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint8_t *b = (uint8_t *)&v[i];
        for (int k = 0; k < 4; k++) {
            uint8_t t = b[k];
            b[k] = b[7 - k];
            b[7 - k] = t;
        }
    }
}

/* Devuelve 0 si va bien; en caso de error escribe el motivo en err. */
static int read_matrix(const char *path, int n, double **out, char *err, size_t errlen) {
    size_t count = (size_t)n * (size_t)n;
    FILE *fh = fopen(path, "rb");
    if (!fh) {
        snprintf(err, errlen, "no se puede abrir %s", path);
        return EXIT_IO;
    }
#ifdef _WIN32
    _fseeki64(fh, 0, SEEK_END);
    long long size = (long long)_ftelli64(fh);
    _fseeki64(fh, 0, SEEK_SET);
#else
    fseek(fh, 0, SEEK_END);
    long long size = (long long)ftell(fh);
    fseek(fh, 0, SEEK_SET);
#endif
    if (size != (long long)(8 * count)) {
        snprintf(err, errlen, "%s tiene %lld bytes; para n=%d se esperaban %lld", path, size, n,
                 (long long)(8 * count));
        fclose(fh);
        return EXIT_IO;
    }
    double *m = malloc(sizeof(double) * count);
    if (!m) {
        fclose(fh);
        snprintf(err, errlen, "sin memoria al leer %s", path);
        return EXIT_OOM;
    }
    size_t got = fread(m, sizeof(double), count, fh);
    fclose(fh);
    if (got != count) {
        free(m);
        snprintf(err, errlen, "lectura incompleta de %s", path);
        return EXIT_IO;
    }
    if (!host_is_little_endian()) byteswap_all(m, count);
    *out = m;
    return 0;
}

static int write_matrix(const char *path, const double *m, int n) {
    size_t count = (size_t)n * (size_t)n;
    FILE *fh = fopen(path, "wb");
    if (!fh) return EXIT_IO;
    size_t put;
    if (host_is_little_endian()) {
        put = fwrite(m, sizeof(double), count, fh);
    } else { /* fila a fila, para no duplicar la matriz en memoria */
        put = 0;
        double *row = malloc(sizeof(double) * (size_t)n);
        if (!row) {
            fclose(fh);
            return EXIT_OOM;
        }
        for (int i = 0; i < n; i++) {
            memcpy(row, m + (size_t)i * n, sizeof(double) * (size_t)n);
            byteswap_all(row, (size_t)n);
            put += fwrite(row, sizeof(double), (size_t)n, fh);
        }
        free(row);
    }
    int bad = fclose(fh) != 0 || put != count;
    return bad ? EXIT_IO : 0;
}

/* ------------------------------------------------------------------------- */
/* Worker                                                                     */
/* ------------------------------------------------------------------------- */
static volatile double sink; /* consume un elemento de cada resultado del lote */

static int emit_error(const char *status, int n, const char *note, int code) {
    fprintf(stderr, "%s: %s\n", status, note);
    printf("{\"status\": \"%s\", \"n\": %d, \"note\": \"", status, n);
    for (const char *ch = note; *ch; ch++) { /* escapa \ y " (rutas de Windows) */
        if (*ch == '\\' || *ch == '"') putchar('\\');
        putchar(*ch);
    }
    printf("\"}\n");
    fflush(stdout);
    return code;
}

/* Copia la esquina m x m superior izquierda de una matriz n x n. */
static double *corner(const double *src, int n, int m) {
    double *dst = malloc(sizeof(double) * (size_t)m * (size_t)m);
    if (!dst) return NULL;
    for (int i = 0; i < m; i++) memcpy(dst + (size_t)i * m, src + (size_t)i * n, sizeof(double) * (size_t)m);
    return dst;
}

/* Región cronometrada: batch multiplicaciones. Devuelve C (la última) y el tiempo total. */
static double *timed(const double *A, const double *B, int n, long batch, int64_t *elapsed) {
    double *C = NULL;
    int64_t t0 = now_ns();
    for (long r = 0; r < batch; r++) {
        double *next = matmul_ijk(A, B, n);
        if (next == NULL) {
            free(C);
            return NULL;
        }
        sink = next[0];
        free(C);
        C = next;
    }
    *elapsed = now_ns() - t0;
    return C;
}

static void usage(void) {
    fprintf(stderr,
            "uso: matmul --n N --a A.bin --b B.bin [--dump-c C.bin] [--warmup-runs R]\n"
            "            [--warmup-n M] [--batch-target-ms MS]\n");
}

int main(int argc, char **argv) {
    int n = 0, warmup_runs_arg = 1, warmup_n = 64;
    double batch_target_ms = 100.0;
    const char *a_path = NULL, *b_path = NULL, *dump_path = NULL;

    for (int i = 1; i < argc; i++) {
        const char *opt = argv[i];
        if (i + 1 >= argc) {
            usage();
            return EXIT_IO;
        }
        const char *val = argv[++i];
        if (strcmp(opt, "--n") == 0) n = atoi(val);
        else if (strcmp(opt, "--a") == 0) a_path = val;
        else if (strcmp(opt, "--b") == 0) b_path = val;
        else if (strcmp(opt, "--dump-c") == 0) dump_path = val;
        else if (strcmp(opt, "--warmup-runs") == 0) warmup_runs_arg = atoi(val);
        else if (strcmp(opt, "--warmup-n") == 0) warmup_n = atoi(val);
        else if (strcmp(opt, "--batch-target-ms") == 0) batch_target_ms = atof(val);
        else {
            usage();
            return EXIT_IO;
        }
    }
    if (n < 1 || !a_path || !b_path) {
        usage();
        return EXIT_IO;
    }

    const char *metric = "unavailable";
    int64_t baseline = peak_rss_bytes(&metric); /* proceso antes de cargar datos */

    char err[512];
    double *A = NULL, *B = NULL;
    int rc = read_matrix(a_path, n, &A, err, sizeof err);
    if (rc == 0) rc = read_matrix(b_path, n, &B, err, sizeof err);
    if (rc != 0) return emit_error(rc == EXIT_OOM ? "OOM" : "error", n, err, rc);

    /* Calentamiento y, si la tarea es muy corta, calibración del lote. */
    long warmup_runs = 0, batch = 1;
    int warmup_size = 0;
    if (warmup_runs_arg > 0) {
        int m = n < warmup_n ? n : warmup_n;
        if (m < 1) m = 1;
        double *wa = corner(A, n, m), *wb = corner(B, n, m);
        if (!wa || !wb) return emit_error("OOM", n, "sin memoria en el calentamiento", EXIT_OOM);
        int64_t t0 = now_ns();
        for (int r = 0; r < warmup_runs_arg; r++) {
            double *w = matmul_ijk(wa, wb, m);
            if (!w) return emit_error("OOM", n, "sin memoria en el calentamiento", EXIT_OOM);
            sink = w[0];
            free(w);
        }
        int64_t warm_ns = now_ns() - t0;
        free(wa);
        free(wb);
        warmup_runs = warmup_runs_arg;
        warmup_size = m;
        if (batch_target_ms > 0) {
            double ratio = (double)n / m;
            double est_ns = (double)warm_ns / warmup_runs_arg * ratio * ratio * ratio;
            if (est_ns < batch_target_ms * 1e6) {
                int64_t cal_ns;
                double *cal = timed(A, B, n, 1, &cal_ns); /* calibración: se descarta */
                if (!cal) return emit_error("OOM", n, "MemoryError", EXIT_OOM);
                free(cal);
                warmup_runs += 1;
                double b = ceil(batch_target_ms * 1e6 / (double)(cal_ns > 0 ? cal_ns : 1));
                batch = b < 1 ? 1 : (long)b;
            }
        }
    } else if (batch_target_ms > 0) {
        int64_t cal_ns;
        double *cal = timed(A, B, n, 1, &cal_ns);
        if (!cal) return emit_error("OOM", n, "MemoryError", EXIT_OOM);
        free(cal);
        warmup_runs += 1;
        if (cal_ns < batch_target_ms * 1e6) {
            double b = ceil(batch_target_ms * 1e6 / (double)(cal_ns > 0 ? cal_ns : 1));
            batch = b < 1 ? 1 : (long)b;
        }
    }

    int64_t total_ns = 0;
    double *C = timed(A, B, n, batch, &total_ns);
    if (!C) return emit_error("OOM", n, "MemoryError", EXIT_OOM);
    int64_t peak = peak_rss_bytes(&metric); /* justo después de multiplicar */
    size_t count = (size_t)n * (size_t)n;
    double checksum = checksum_fsum(C, count);
    if (dump_path && write_matrix(dump_path, C, n) != 0)
        return emit_error("error", n, "no se pudo escribir el volcado de C", EXIT_IO);

    long long model = 3LL * 8LL * (long long)count;
    printf("{\"status\": \"OK\", \"language\": \"%s\", \"method\": \"%s\", \"n\": %d, "
           "\"dtype\": \"%s\", \"threads\": 1, \"batch\": %ld, \"time_total_ns\": %lld, "
           "\"time_ns\": %.17g, \"warmup_runs\": %ld, \"warmup_n\": %d, "
           "\"peak_rss_bytes\": %lld, \"baseline_rss_bytes\": %lld, \"memory_metric\": \"%s\", "
           "\"array_model_bytes\": %lld, \"object_model_bytes\": %lld, \"checksum\": \"%.17g\", "
           "\"runtime_version\": \"gcc %d.%d.%d\", \"runtime_flags\": \"%s\"}\n",
           LANGUAGE, METHOD, n, DTYPE, batch, (long long)total_ns, (double)total_ns / (double)batch,
           warmup_runs, warmup_size, (long long)peak, (long long)baseline, metric, model, model,
           checksum, __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__, BUILD_FLAGS);
    fflush(stdout);
    free(A);
    free(B);
    free(C);
    return 0;
}
