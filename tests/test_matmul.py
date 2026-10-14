"""Tests de corrección de la Práctica 1 (Python).

Se ejecutan con:  python -m unittest discover -s tests -v

Cubren: casos con respuesta conocida (identidad, cero, 2x2 y 3x3 a mano), no
conmutatividad (detecta índices intercambiados), comparación con NumPy,
lectura/escritura binaria, determinismo del generador de datos, el worker de
línea de comandos y el verificador.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "python"
sys.path.insert(0, str(SRC))

import datagen  # noqa: E402
import matrix_io  # noqa: E402
import memtools  # noqa: E402
from matmul import (  # noqa: E402
    array_model_bytes,
    checksum_fsum,
    matmul_ijk,
    python_object_model_bytes,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(n):
    return [[0.0] * n for _ in range(n)]


def random_matrix(n, seed):
    rnd = random.Random(seed)
    return [[rnd.random() for _ in range(n)] for _ in range(n)]


class TestKernelKnownAnswers(unittest.TestCase):
    def test_1x1(self):
        self.assertEqual(matmul_ijk([[3.0]], [[4.0]], 1), [[12.0]])

    def test_2x2_by_hand(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        self.assertEqual(matmul_ijk(A, B, 2), [[19.0, 22.0], [43.0, 50.0]])

    def test_3x3_by_hand_non_symmetric(self):
        A = [[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]]
        B = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 0.0, 1.0]]
        expected = [[7.0, 4.0, 7.0], [11.0, 5.0, 17.0], [20.0, 1.0, 8.0]]
        self.assertEqual(matmul_ijk(A, B, 3), expected)

    def test_not_commutative(self):
        """A@B != B@A: si los índices estuvieran intercambiados, este test falla."""
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[0.0, 1.0], [1.0, 0.0]]
        self.assertEqual(matmul_ijk(A, B, 2), [[2.0, 1.0], [4.0, 3.0]])
        self.assertEqual(matmul_ijk(B, A, 2), [[3.0, 4.0], [1.0, 2.0]])

    def test_identity_is_neutral(self):
        for n in (1, 5, 16):
            A = random_matrix(n, seed=n)
            self.assertEqual(matmul_ijk(A, identity(n), n), A)
            self.assertEqual(matmul_ijk(identity(n), A, n), A)

    def test_zero_matrix(self):
        for n in (1, 5, 16):
            A = random_matrix(n, seed=100 + n)
            self.assertEqual(matmul_ijk(A, zeros(n), n), zeros(n))
            self.assertEqual(matmul_ijk(zeros(n), A, n), zeros(n))

    def test_inputs_not_modified(self):
        A = random_matrix(8, 1)
        B = random_matrix(8, 2)
        A0 = [row[:] for row in A]
        B0 = [row[:] for row in B]
        matmul_ijk(A, B, 8)
        self.assertEqual(A, A0)
        self.assertEqual(B, B0)

    def test_integer_valued_floats_are_exact(self):
        """Con enteros pequeños en float64 el resultado es exacto (sin redondeo)."""
        rnd = random.Random(7)
        n = 12
        A = [[float(rnd.randint(-9, 9)) for _ in range(n)] for _ in range(n)]
        B = [[float(rnd.randint(-9, 9)) for _ in range(n)] for _ in range(n)]
        expected = [
            [float(sum(int(A[i][k]) * int(B[k][j]) for k in range(n))) for j in range(n)]
            for i in range(n)
        ]
        self.assertEqual(matmul_ijk(A, B, n), expected)


@unittest.skipIf(np is None, "NumPy no instalado (solo se usa como referencia)")
class TestAgainstNumpy(unittest.TestCase):
    def test_random_sizes(self):
        for n in (2, 7, 33, 64):
            A = random_matrix(n, seed=10 + n)
            B = random_matrix(n, seed=20 + n)
            C = np.array(matmul_ijk(A, B, n))
            R = np.array(A) @ np.array(B)
            np.testing.assert_allclose(C, R, rtol=1e-9, atol=1e-12)


class TestMatrixIO(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "M.bin"
            M = random_matrix(9, seed=3)
            matrix_io.write_matrix(path, M)
            self.assertEqual(path.stat().st_size, 8 * 9 * 9)
            self.assertEqual(matrix_io.read_matrix(path, 9), M)

    def test_wrong_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "M.bin"
            matrix_io.write_matrix(path, random_matrix(4, seed=3))
            with self.assertRaises(ValueError):
                matrix_io.read_matrix(path, 5)

    def test_little_endian_layout(self):
        """El formato compartido es float64 little-endian, row-major."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "M.bin"
            matrix_io.write_matrix(path, [[1.0, 2.0], [3.0, 4.0]])
            raw = path.read_bytes()
            import struct

            self.assertEqual(struct.unpack("<4d", raw), (1.0, 2.0, 3.0, 4.0))


class TestDatagen(unittest.TestCase):
    def test_deterministic_and_in_range(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            m1 = datagen.generate([6, 11], Path(t1), seed=42, log=lambda *_: None)
            m2 = datagen.generate([6, 11], Path(t2), seed=42, log=lambda *_: None)
            self.assertEqual(m1["files"], m2["files"])
            A = matrix_io.read_matrix(Path(t1) / "A_6.bin", 6)
            B = matrix_io.read_matrix(Path(t1) / "B_6.bin", 6)
            self.assertNotEqual(A, B)
            for row in A + B:
                for x in row:
                    self.assertTrue(0.0 <= x < 1.0)

    def test_manifest_verify_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            datagen.generate([5], tmp, seed=1, log=lambda *_: None)
            self.assertEqual(datagen.verify(tmp), [])
            (tmp / "A_5.bin").write_bytes(b"\x00" * (8 * 25))
            self.assertTrue(datagen.verify(tmp))
            # generate() repara el fichero corrupto reproduciendo el SHA-256 del manifest
            datagen.generate([5], tmp, seed=1, log=lambda *_: None)
            self.assertEqual(datagen.verify(tmp), [])

    def test_different_seed_requires_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            datagen.generate([5], tmp, seed=1, log=lambda *_: None)
            with self.assertRaises(ValueError):
                datagen.generate([5], tmp, seed=2, log=lambda *_: None)


class TestMemory(unittest.TestCase):
    def test_peak_rss_positive_and_monotonic(self):
        before, metric = memtools.peak_rss_bytes()
        self.assertGreater(before, 0)
        self.assertNotEqual(metric, "unavailable")
        block = bytearray(40 * 1024 * 1024)
        block[::4096] = b"\x01" * len(block[::4096])  # toca las páginas
        after, _ = memtools.peak_rss_bytes()
        self.assertGreaterEqual(after, before + 20 * 1024 * 1024)
        del block

    def test_models(self):
        self.assertEqual(array_model_bytes(10), 2400)
        M = random_matrix(10, 1)
        self.assertGreater(python_object_model_bytes(M, M, M), 3 * 8 * 100)


class TestWorkerPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        datagen.generate([8, 24], cls.dir, seed=5, log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_worker(self, n, *extra):
        cmd = [
            sys.executable, str(SRC / "matmul.py"), "--n", str(n),
            "--a", str(self.dir / f"A_{n}.bin"), "--b", str(self.dir / f"B_{n}.bin"), *extra,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_worker_output_fields(self):
        data = self.run_worker(24, "--batch-target-ms", "0")
        self.assertEqual(data["status"], "OK")
        self.assertEqual(data["batch"], 1)
        self.assertGreater(data["time_ns"], 0)
        self.assertEqual(data["threads"], 1)
        self.assertEqual(data["dtype"], "float64")
        self.assertGreater(data["peak_rss_bytes"], data["baseline_rss_bytes"] // 2)
        self.assertEqual(data["array_model_bytes"], 3 * 8 * 24 * 24)

    def test_tiny_task_is_batched_and_reported_per_multiplication(self):
        data = self.run_worker(8, "--batch-target-ms", "50")
        self.assertGreater(data["batch"], 1)
        self.assertAlmostEqual(data["time_ns"], data["time_total_ns"] / data["batch"], delta=1e-6)
        self.assertGreaterEqual(data["warmup_runs"], 2)  # warm-up + calibración

    def test_checksum_matches_reference_and_is_reproducible(self):
        first = self.run_worker(24, "--batch-target-ms", "0")
        second = self.run_worker(24, "--batch-target-ms", "0")
        self.assertEqual(first["checksum"], second["checksum"])  # bit a bit
        if np is not None:
            A = matrix_io.read_matrix(self.dir / "A_24.bin", 24)
            B = matrix_io.read_matrix(self.dir / "B_24.bin", 24)
            ref = float(np.sum(np.array(A) @ np.array(B)))
            self.assertAlmostEqual(float(first["checksum"]) / ref, 1.0, places=12)

    @unittest.skipIf(np is None, "NumPy no instalado")
    def test_dump_and_verify_cli(self):
        dump = self.dir / "C_24.bin"
        self.run_worker(24, "--batch-target-ms", "0", "--dump-c", str(dump))
        cmd = [
            sys.executable, str(SRC / "verify.py"), "--n", "24",
            "--a", str(self.dir / "A_24.bin"), "--b", str(self.dir / "B_24.bin"), "--c", str(dump),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["ok"])

        # Si corrompemos un elemento, el verificador debe rechazarlo (código 4)
        C = matrix_io.read_matrix(dump, 24)
        C[3][5] += 1e-3
        matrix_io.write_matrix(dump, C)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 4)
        self.assertFalse(json.loads(proc.stdout)["ok"])

    def test_missing_file_fails_cleanly(self):
        cmd = [sys.executable, str(SRC / "matmul.py"), "--n", "24",
               "--a", str(self.dir / "nope.bin"), "--b", str(self.dir / "B_24.bin")]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(proc.returncode, 0)


@unittest.skipIf(np is None, "NumPy no instalado")
class TestVerifyCompare(unittest.TestCase):
    def test_tolerances(self):
        import verify

        R = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(verify.compare(R * (1 + 1e-13), R)["ok"])
        self.assertFalse(verify.compare(R * (1 + 1e-6), R)["ok"])
        bad = R.copy()
        bad[0, 0] = float("nan")
        self.assertFalse(verify.compare(bad, R)["ok"])
        self.assertFalse(verify.compare(R[:1], R)["ok"])


class TestChecksum(unittest.TestCase):
    def test_fsum_is_exact_for_tricky_sums(self):
        # 1e16 + 1 - 1e16 pierde el 1 con suma ingenua; fsum lo conserva
        self.assertEqual(checksum_fsum([[1e16, 1.0], [-1e16, 0.0]]), 1.0)


if __name__ == "__main__":
    unittest.main()
