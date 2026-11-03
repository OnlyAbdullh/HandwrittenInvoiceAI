"""The Qwen subprocess transport under concurrent use, against a fake worker.

The HTTP service runs request handlers on a thread pool and every request shares
one worker process. Without a lock around each write/read exchange, two
invoices processed at once could each receive the other's cell texts -- valid
JSON, plausible values, wrong invoice.
"""

from __future__ import annotations

import base64
import hashlib
import io
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

from core.exceptions import RecognitionTimeoutError
from ocr.engines.qwen_vlm_engine import QwenVLMEngine

FAKE_WORKER = Path(__file__).parent / "fixtures" / "fake_qwen_worker.py"


def make_engine(tmp_path, timeout_s: float = 10.0) -> QwenVLMEngine:
    return QwenVLMEngine(
        model_path=str(tmp_path),
        backend="subprocess",
        worker_python=sys.executable,
        worker_script=str(FAKE_WORKER),
        timeout_s=timeout_s,
        startup_timeout_s=30.0,
    )


def image(seed: int) -> Image.Image:
    return Image.new("RGB", (8, 8), color=(seed % 256, (seed * 7) % 256, (seed * 13) % 256))


def expected_text(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(base64.b64encode(buf.getvalue()).decode().encode()).hexdigest()[:16]


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(tmp_path)
    yield engine
    engine.close()


def test_concurrent_requests_each_get_their_own_answers(engine):
    errors: list[str] = []

    def worker(thread_index: int) -> None:
        for round_index in range(5):
            images = [image(thread_index * 100 + round_index * 10 + k) for k in range(3)]
            readings = engine.read_batch([(img, "number") for img in images])
            for img, reading in zip(images, readings):
                if reading.text != expected_text(img):
                    errors.append(f"thread {thread_index} got another request's answer")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors[:3]


def test_confidence_travels_with_the_text(engine):
    reading = engine.read(image(1), kind="number")
    assert reading.confidence == 0.9


def test_a_hung_worker_times_out_and_is_replaced(tmp_path):
    engine = make_engine(tmp_path, timeout_s=1.0)
    try:
        with pytest.raises(RecognitionTimeoutError):
            engine.read(image(1), kind="any")  # the fake hangs on "any"

        # The hung process was stopped; the next request starts a fresh one.
        reading = engine.read(image(2), kind="number")
        assert reading.text == expected_text(image(2))
    finally:
        engine.close()


def test_a_closed_engine_is_not_resurrected(tmp_path):
    engine = make_engine(tmp_path)
    engine.close()

    with pytest.raises(RuntimeError, match="closed"):
        engine.read(image(1), kind="number")
    assert not engine.is_ready()
