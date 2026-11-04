"""evaluation/resources.py: memory is measured, or reported unknown -- never guessed."""

from __future__ import annotations

import subprocess
import sys

import pytest

from evaluation import resources


@pytest.mark.skipif(not sys.platform.startswith(("linux", "win32")), reason="peak RAM needs /proc or Win32")
def test_this_process_has_a_peak():
    assert resources.peak_rss_mb() > 0


def test_a_running_child_is_found_and_counted():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        if child.pid not in resources.child_pids():
            pytest.skip("children cannot be listed on this platform without psutil")
        memory = resources.process_tree_memory()
        assert str(child.pid) in memory["peak_ram_children_mb"]
        assert memory["peak_ram_mb"] >= memory["peak_ram_this_process_mb"]
    finally:
        child.kill()
        child.wait()


def test_no_gpu_tool_means_unknown_vram(monkeypatch):
    monkeypatch.setattr(resources.shutil, "which", lambda name: None)
    assert resources.gpu_mb_for_pid(1) is None


def test_vram_is_read_from_nvidia_smi_for_that_process(monkeypatch):
    monkeypatch.setattr(resources.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(resources.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="111, 50\n4242, 2300\n"))
    assert resources.gpu_mb_for_pid(4242) == 2300.0
    assert resources.gpu_mb_for_pid(7) is None
