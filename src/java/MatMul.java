import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

/**
 * Kernel de la Práctica 1 (Java) y <i>worker</i> que hace UNA medición.
 *
 * <p>Algoritmo: multiplicación densa ingenua, triple bucle en orden i-j-k:
 *
 * <pre>
 *     for i: for j: for k: C[i][j] += A[i][k] * B[k][j]
 * </pre>
 *
 * <p>Sin librerías de multiplicación (trabajo O(n³)). Estructura de datos: {@code double[][]},
 * es decir, un array de referencias a filas; cada fila es un {@code double[]} contiguo. Las
 * filas son objetos independientes del heap (no hay garantía de que estén juntas en memoria).
 *
 * <p>Qué cronometra el worker (medición del <i>kernel</i>), igual que en Python y C:
 *
 * <ul>
 *   <li>DENTRO: la llamada a {@link #matmulIjk} (reserva C, que la JVM inicializa a 0.0, y
 *       ejecuta los bucles).
 *   <li>FUERA: lectura de ficheros, calentamiento, calibración, checksum, volcado de C.
 *   <li>Temporizador monótono: {@link System#nanoTime()}.
 * </ul>
 *
 * <p>Calentamiento (JIT): {@code --warmup-runs} ejecuciones del kernel sobre matrices de
 * {@code --warmup-n} x {@code --warmup-n} rellenadas con los valores de A y B (por defecto 200
 * llamadas de 64 x 64, unos 52 millones de iteraciones internas), para que HotSpot compile
 * {@code matmulIjk} con C2 antes de medir. Nunca forman parte de la muestra.
 *
 * <ul>
 *   <li>El tamaño NO depende de n: con n pequeño, calentar con n x n no basta para compilar.
 *   <li>Muchas llamadas cortas, no pocas largas: si C2 compila el método durante la primera
 *       llamada larga, el perfil aún no ha visto salir el bucle exterior; en la siguiente llamada
 *       salta un <i>uncommon trap</i> ({@code unstable_if}), el código C2 se descarta y la medición
 *       puede ejecutarse con código menos optimizado (comprobado con
 *       {@code -XX:+PrintCompilation} y {@code -XX:+LogCompilation}: 4,0 s frente a 2,4 s con n=1000).
 * </ul>
 *
 * <p>Tareas muy cortas: como en Python y C, si una multiplicación dura menos de
 * {@code --batch-target-ms} se mide un LOTE y {@code time_ns} es el tiempo POR multiplicación.
 *
 * <p>Memoria (dentro del proceso): pico de heap usado, suma de {@code getPeakUsage().getUsed()} de
 * los pools de tipo HEAP ({@link MemoryPoolMXBean}). NO es lo mismo que el RSS del proceso: no
 * incluye la propia JVM (código, metaspace, pilas, JIT). El pico de working set del proceso
 * completo lo mide el runner desde fuera, igual para los tres lenguajes.
 *
 * <p>Checksum: suma correctamente redondeada (algoritmo de Shewchuk, el de {@code math.fsum} de
 * CPython), así que coincide bit a bit con el de Python y C si C coincide bit a bit. Java no
 * contrae a+b*c en FMA (semántica IEEE estricta desde Java 17).
 *
 * <p>Salida: la última línea de stdout es un objeto JSON. Códigos de salida: 0 = OK, 2 = error de
 * entrada/salida o de argumentos, 3 = memoria insuficiente.
 *
 * <pre>
 * javac -d build/java src/java/MatMul.java
 * java -XX:+UseSerialGC -Xms2g -Xmx2g -cp build/java MatMul --n 200 --a data/A_200.bin --b data/B_200.bin
 * </pre>
 */
public final class MatMul {
    static final String LANGUAGE = "java";
    static final String METHOD = "triple_loop_ijk";
    static final String DTYPE = "float64";
    static final int EXIT_IO = 2;
    static final int EXIT_OOM = 3;

    /** Consume un elemento de cada resultado del lote (evita que el JIT descarte el trabajo). */
    static volatile double sink;

    private MatMul() {}

    // ----------------------------------------------------------------------------------------- //
    // Kernel
    // ----------------------------------------------------------------------------------------- //

    /** C = A x B (n x n), triple bucle i-j-k. La JVM inicializa C a 0.0. */
    static double[][] matmulIjk(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    /** Suma correctamente redondeada (Shewchuk), traducción de math.fsum de CPython. */
    static double checksumFsum(double[][] C) {
        double[] p = new double[128];
        int np = 0;
        for (double[] row : C) {
            for (double value : row) {
                double v = value;
                int i = 0;
                for (int j = 0; j < np; j++) {
                    double y = p[j];
                    if (Math.abs(v) < Math.abs(y)) {
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
                if (v != 0.0) p[np++] = v;
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

    // ----------------------------------------------------------------------------------------- //
    // E/S en el formato compartido (float64 little-endian, row-major, sin cabecera)
    // ----------------------------------------------------------------------------------------- //

    static double[][] readMatrix(Path path, int n) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            long expected = 8L * n * n;
            if (ch.size() != expected) {
                throw new IOException(path + " tiene " + ch.size() + " bytes; para n=" + n
                        + " se esperaban " + expected);
            }
            double[][] M = new double[n][n];
            ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) { // fila a fila: no hace falta un búfer de 8n² bytes
                buf.clear();
                while (buf.hasRemaining()) {
                    if (ch.read(buf) < 0) throw new IOException("lectura incompleta de " + path);
                }
                buf.flip();
                buf.asDoubleBuffer().get(M[i]);
            }
            return M;
        }
    }

    static void writeMatrix(Path path, double[][] M) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            int n = M.length;
            ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
            for (double[] row : M) {
                buf.clear();
                buf.asDoubleBuffer().put(row);
                buf.limit(8 * row.length);
                while (buf.hasRemaining()) ch.write(buf);
            }
        }
    }

    // ----------------------------------------------------------------------------------------- //
    // Memoria y GC (dentro del proceso)
    // ----------------------------------------------------------------------------------------- //

    /** Pico de heap usado desde el arranque de la JVM: suma de los pools de tipo HEAP. */
    static long heapPeakUsedBytes() {
        long total = 0;
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP && pool.getPeakUsage() != null) {
                total += pool.getPeakUsage().getUsed();
            }
        }
        return total;
    }

    static long[] gcCountAndMillis() {
        long count = 0, millis = 0;
        for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
            count += Math.max(0, gc.getCollectionCount());
            millis += Math.max(0, gc.getCollectionTime());
        }
        return new long[] {count, millis};
    }

    // ----------------------------------------------------------------------------------------- //
    // Worker
    // ----------------------------------------------------------------------------------------- //

    /**
     * Matriz m x m para el calentamiento, rellenada recorriendo cíclicamente los valores de M
     * (n x n). Así el JIT recibe el mismo calentamiento para cualquier n, también cuando n < m.
     */
    static double[][] warmupMatrix(double[][] M, int n, int m) {
        double[][] out = new double[m][m];
        long idx = 0, count = (long) n * n;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++, idx++) {
                long t = idx % count;
                out[i][j] = M[(int) (t / n)][(int) (t % n)];
            }
        }
        return out;
    }

    /** Resultado de la región cronometrada. */
    record Timed(double[][] C, long elapsedNs) {}

    static Timed timed(double[][] A, double[][] B, int n, long batch) {
        double[][] C = null;
        long t0 = System.nanoTime();
        for (long r = 0; r < batch; r++) {
            C = matmulIjk(A, B, n);
            sink = C[0][0];
        }
        long t1 = System.nanoTime();
        return new Timed(C, t1 - t0);
    }

    static String jsonEscape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    static int emitError(String status, int n, String note, int code) {
        System.err.println(status + ": " + note);
        System.out.println("{\"status\": \"" + status + "\", \"n\": " + n + ", \"note\": \""
                + jsonEscape(note) + "\"}");
        System.out.flush();
        return code;
    }

    static int run(String[] args) {
        int n = 0, warmupRunsArg = 200, warmupN = 64;
        double batchTargetMs = 100.0;
        String aPath = null, bPath = null, dumpPath = null;
        try {
            for (int i = 0; i < args.length; i += 2) {
                String opt = args[i], val = args[i + 1];
                switch (opt) {
                    case "--n" -> n = Integer.parseInt(val);
                    case "--a" -> aPath = val;
                    case "--b" -> bPath = val;
                    case "--dump-c" -> dumpPath = val;
                    case "--warmup-runs" -> warmupRunsArg = Integer.parseInt(val);
                    case "--warmup-n" -> warmupN = Integer.parseInt(val);
                    case "--batch-target-ms" -> batchTargetMs = Double.parseDouble(val);
                    default -> throw new IllegalArgumentException("opción desconocida " + opt);
                }
            }
        } catch (RuntimeException e) {
            System.err.println("uso: MatMul --n N --a A.bin --b B.bin [--dump-c C.bin] "
                    + "[--warmup-runs R] [--warmup-n M] [--batch-target-ms MS]");
            return EXIT_IO;
        }
        if (n < 1 || aPath == null || bPath == null) {
            System.err.println("faltan --n, --a o --b");
            return EXIT_IO;
        }

        long baseline = heapPeakUsedBytes(); // heap antes de cargar datos
        double[][] A, B, C;
        long batch = 1, warmupRuns = 0, totalNs, peak, gcCount, gcMillis;
        int warmupSize = 0;
        double checksum;
        try {
            A = readMatrix(Path.of(aPath), n);
            B = readMatrix(Path.of(bPath), n);

            if (warmupRunsArg > 0) {
                int m = Math.max(1, warmupN);
                double[][] wa = warmupMatrix(A, n, m), wb = warmupMatrix(B, n, m);
                long lastNs = 0;
                for (int r = 0; r < warmupRunsArg; r++) {
                    long t0 = System.nanoTime();
                    sink = matmulIjk(wa, wb, m)[0][0];
                    lastNs = System.nanoTime() - t0;
                }
                warmupRuns = warmupRunsArg;
                warmupSize = m;
                if (batchTargetMs > 0) {
                    // las primeras ejecuciones son interpretadas: la última (ya compilada)
                    // estima mejor el coste de una multiplicación completa
                    double ratio = (double) n / m;
                    double estNs = (double) lastNs * ratio * ratio * ratio;
                    if (estNs < batchTargetMs * 1e6) {
                        long calNs = timed(A, B, n, 1).elapsedNs(); // calibración: se descarta
                        warmupRuns += 1;
                        batch = Math.max(1, (long) Math.ceil(batchTargetMs * 1e6 / Math.max(calNs, 1)));
                    }
                }
            } else if (batchTargetMs > 0) {
                long calNs = timed(A, B, n, 1).elapsedNs();
                warmupRuns += 1;
                if (calNs < batchTargetMs * 1e6) {
                    batch = Math.max(1, (long) Math.ceil(batchTargetMs * 1e6 / Math.max(calNs, 1)));
                }
            }

            long[] gcBefore = gcCountAndMillis();
            Timed t = timed(A, B, n, batch);
            long[] gcAfter = gcCountAndMillis();
            C = t.C();
            totalNs = t.elapsedNs();
            peak = heapPeakUsedBytes(); // justo después de multiplicar
            gcCount = gcAfter[0] - gcBefore[0];
            gcMillis = gcAfter[1] - gcBefore[1];
            checksum = checksumFsum(C);
            if (dumpPath != null) writeMatrix(Path.of(dumpPath), C);
        } catch (OutOfMemoryError e) {
            return emitError("OOM", n, "OutOfMemoryError: " + e.getMessage(), EXIT_OOM);
        } catch (IOException e) {
            return emitError("error", n, e.toString(), EXIT_IO);
        }

        long count = (long) n * n;
        long arrayModel = 3L * 8L * count;
        // double[][]: array externo (cabecera 16 B + n referencias comprimidas de 4 B)
        // + n filas (cabecera 16 B + 8n B), por matriz, con la JVM de 64 bits por defecto.
        long objectModel = 3L * ((16 + 4L * n) + n * (16 + 8L * n));
        String flags = String.join(" ", ManagementFactory.getRuntimeMXBean().getInputArguments());
        String vm = System.getProperty("java.vm.name") + " " + System.getProperty("java.runtime.version");
        int threads = ManagementFactory.getThreadMXBean().getThreadCount();

        String json = String.format(Locale.ROOT,
                "{\"status\": \"OK\", \"language\": \"%s\", \"method\": \"%s\", \"n\": %d, "
                        + "\"dtype\": \"%s\", \"threads\": 1, \"batch\": %d, \"time_total_ns\": %d, "
                        + "\"time_ns\": %s, \"warmup_runs\": %d, \"warmup_n\": %d, "
                        + "\"peak_rss_bytes\": %d, \"baseline_rss_bytes\": %d, "
                        + "\"memory_metric\": \"heap_peak_used\", "
                        + "\"array_model_bytes\": %d, \"object_model_bytes\": %d, \"checksum\": \"%s\", "
                        + "\"runtime_version\": \"%s\", \"runtime_flags\": \"%s\", "
                        + "\"jvm_threads\": %d, \"gc_count_timed\": %d, \"gc_ms_timed\": %d}",
                LANGUAGE, METHOD, n, DTYPE, batch, totalNs, Double.toString((double) totalNs / batch),
                warmupRuns, warmupSize, peak, baseline, arrayModel, objectModel,
                Double.toString(checksum), jsonEscape(vm), jsonEscape(flags), threads, gcCount, gcMillis);
        System.out.println(json);
        System.out.flush();
        return 0;
    }

    public static void main(String[] args) {
        System.exit(run(args));
    }
}
