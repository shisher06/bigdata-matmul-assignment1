"""Medición del pico de memoria residente (RSS) del proceso actual.

Métrica usada en todo el proyecto: *pico de memoria residente del proceso*
(high-water mark del RSS), en bytes.

Fuentes, por orden de preferencia:

1. Linux:   campo ``VmHWM`` de ``/proc/self/status``.
            Se prefiere a ``ru_maxrss`` porque ``ru_maxrss`` hereda el RSS del
            proceso padre en el ``fork`` previo al ``exec`` y puede inflar el
            valor; ``VmHWM`` pertenece solo al proceso ya "exec-eado".
2. Unix:    ``resource.getrusage(RUSAGE_SELF).ru_maxrss``
            (kilobytes en Linux, bytes en macOS).
3. Windows: ``psutil.Process().memory_info().peak_wset``.

La función devuelve también el nombre de la fuente para dejarlo registrado.
"""

from __future__ import annotations

import sys


def _vmhwm_linux() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    return int(parts[1]) * 1024  # el valor viene en kB
    except OSError:
        return None
    return None


def _ru_maxrss() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == "darwin" else int(value) * 1024


def _peak_wset_windows() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    except Exception:  # psutil ausente o atributo inexistente
        return None


def peak_rss_bytes() -> tuple[int, str]:
    """Devuelve ``(bytes, nombre_de_la_métrica)`` del pico de RSS del proceso."""
    value = _vmhwm_linux()
    if value is not None:
        return value, "VmHWM"
    value = _ru_maxrss()
    if value is not None:
        return value, "ru_maxrss"
    value = _peak_wset_windows()
    if value is not None:
        return value, "peak_wset"
    return -1, "unavailable"


if __name__ == "__main__":
    print(peak_rss_bytes())
