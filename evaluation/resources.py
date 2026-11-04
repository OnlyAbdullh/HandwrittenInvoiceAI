"""Memory a run used -- peak RAM per process, VRAM per process -- with the
standard library only, so the benchmark's model venv can import it too.

Each measure returns None when it cannot be taken on this machine, never a
guess. Peak RAM is the operating system's own high-water mark (VmHWM on Linux,
PeakWorkingSetSize on Windows); elsewhere psutil's current RSS is the best
available and is reported as such. VRAM comes from nvidia-smi.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _linux_peak_mb(pid: int | None) -> float | None:
    try:
        for line in Path(f"/proc/{pid or 'self'}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError):
        return None
    return None


def _windows_peak_mb(pid: int | None) -> float | None:
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
        handle = kernel32.GetCurrentProcess() if pid is None else kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            return None
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ok = kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if pid is not None:
            kernel32.CloseHandle(handle)
        return round(counters.PeakWorkingSetSize / 2**20, 1) if ok else None
    except Exception:  # noqa: BLE001
        return None


def peak_rss_mb(pid: int | None = None) -> float | None:
    """Peak resident memory of a process (this one by default), in MB, if measurable."""
    if sys.platform.startswith("linux"):
        return _linux_peak_mb(pid)
    if sys.platform == "win32":
        return _windows_peak_mb(pid)
    try:
        import psutil

        return round(psutil.Process(pid).memory_info().rss / 2**20, 1)
    except Exception:  # noqa: BLE001 - psutil is optional
        return None


def gpu_mb_for_pid(pid: int) -> float | None:
    """VRAM a process holds now, from nvidia-smi, if there is an NVIDIA GPU."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run([smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def child_pids() -> list[int]:
    """Every descendant of this process (a llama-server, a model worker), when
    they can be listed."""
    try:
        import psutil

        return [c.pid for c in psutil.Process().children(recursive=True)]
    except Exception:  # noqa: BLE001, S110 - psutil is optional; fall through to /proc
        pass
    if not sys.platform.startswith("linux"):
        return []
    found: list[int] = []
    queue = [str(os.getpid())]
    while queue:
        pid = queue.pop()
        for children in Path(f"/proc/{pid}/task").glob("*/children"):
            try:
                kids = children.read_text().split()
            except OSError:
                continue
            found += [int(k) for k in kids]
            queue += kids
    return found


def process_tree_memory() -> dict:
    """Peak RAM of this process plus each child's, and the VRAM they hold now.

    A child's peak is only readable while it is alive, so call this before the
    pipeline (and its llama-server or model worker) is closed. Summing peaks
    overstates the true simultaneous peak slightly; it never understates it.
    """
    own = peak_rss_mb()
    children = {pid: peak_rss_mb(pid) for pid in child_pids()}
    known_ram = [v for v in (own, *children.values()) if v is not None]
    known_vram = [v for v in (gpu_mb_for_pid(pid) for pid in (os.getpid(), *children)) if v is not None]
    return {
        "peak_ram_mb": round(sum(known_ram), 1) if known_ram else None,
        "peak_ram_this_process_mb": own,
        "peak_ram_children_mb": {str(pid): mb for pid, mb in children.items()},
        "vram_mb": round(sum(known_vram), 1) if known_vram else None,
    }
