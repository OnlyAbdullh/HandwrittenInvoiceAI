"""tools/check_llamacpp_setup.py -- the pre-flight check the setup guide points at."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.llamacpp_support import write_model, write_projector

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("check_llamacpp_setup", ROOT / "tools" / "check_llamacpp_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reports_every_missing_piece_and_fails(tool, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INVOICE_AI_LLAMA_SERVER", str(tmp_path / "no" / "llama-server.exe"))
    monkeypatch.setenv("INVOICE_AI_NUQTA_MODEL", str(tmp_path / "missing.gguf"))
    monkeypatch.setenv("INVOICE_AI_NUQTA_MMPROJ", str(tmp_path / "missing-mmproj.gguf"))

    assert tool.main([]) == 1

    out = capsys.readouterr().out
    assert "llama-server not found" in out
    assert "Model file not found" in out


def test_passes_with_valid_files_and_a_runnable_server(tool, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INVOICE_AI_LLAMA_SERVER", sys.executable)  # runs; prints no build number
    monkeypatch.setenv("INVOICE_AI_NUQTA_MODEL", str(write_model(tmp_path / "m.gguf")))
    monkeypatch.setenv("INVOICE_AI_NUQTA_MMPROJ", str(write_projector(tmp_path / "mm.gguf")))

    assert tool.main([]) == 0

    out = capsys.readouterr().out
    assert "architecture 'qwen35'" in out
    assert "qwen3vl_merger" in out
    assert "All checks passed." in out


def test_refuses_a_config_that_does_not_use_llamacpp(tool, capsys):
    assert tool.main(["--config", "config/default_config.yaml"]) == 1
    assert "does not select the llamacpp recognizer" in capsys.readouterr().out


def test_service_resolves_a_relative_config_path_against_the_project(monkeypatch, tmp_path):
    from api.service_settings import ServiceSettings

    monkeypatch.chdir(tmp_path)
    settings = ServiceSettings(config_path="config/nuqta_config.yaml")
    assert Path(settings.config_path) == ROOT / "config" / "nuqta_config.yaml"
