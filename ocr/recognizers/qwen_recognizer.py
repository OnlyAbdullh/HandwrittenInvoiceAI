"""Qwen as a TextRecognizer, run by transformers.

A thin adapter over :class:`~ocr.engines.qwen_vlm_engine.QwenVLMEngine`, not a
reimplementation. That class owns the transports and their awkward parts
(worker startup, timeouts, thread safety, UTF-8 on Windows).

Batching, failure isolation and output validation are shared with every other
prompted VLM recognizer and live in
:class:`~ocr.recognizers.prompted_recognizer.PromptedRecognizer`. The
recognizer knows about *kinds of characters* and nothing about invoices.

The GGUF build of a model is read by the ``llamacpp`` recognizer instead; both
can be installed side by side and are chosen by ``ocr.recognizer.name``.
"""

from __future__ import annotations

from core.domain.roles import ContentKind
from ocr.engines.qwen_vlm_engine import PROMPTS, QwenVLMEngine
from ocr.recognizers.prompted_recognizer import (
    PromptedRecognizer,
    check_batch_settings,
    require_prompt_for_every_kind,
)
from ocr.recognizers.registry import recognizer_registry


@recognizer_registry.register("qwen")
class QwenRecognizer(PromptedRecognizer):
    name = "qwen"

    def __init__(
        self,
        default_kind: str = ContentKind.ANY.value,
        batch_size: int = 4,
        model_path: str = "OCR_Models/Arabic-Qwen3.5-OCR-v4",
        backend: str = "subprocess",
        host: str = "http://localhost:11434",
        model_name: str = "arabic-ocr",
        max_new_tokens: int = 64,
        worker_python: str = "qwen_venv/Scripts/python.exe",
        worker_script: str = "tools/qwen_worker.py",
        arch: str | None = None,
        timeout_s: float = 120.0,
        startup_timeout_s: float = 600.0,
    ):
        # Both checks run before the engine is built: building it starts the
        # worker process and loads the model.
        require_prompt_for_every_kind(PROMPTS, self.name)
        kind = check_batch_settings(self.name, default_kind, batch_size)

        engine = QwenVLMEngine(
            model_path=model_path,
            backend=backend,
            host=host,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            worker_python=worker_python,
            worker_script=worker_script,
            arch=arch,
            timeout_s=timeout_s,
            startup_timeout_s=startup_timeout_s,
        )
        super().__init__(engine, kind, batch_size)

    @property
    def engine(self) -> QwenVLMEngine:
        return self._backend
