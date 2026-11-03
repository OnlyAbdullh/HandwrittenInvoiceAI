"""The llamacpp recognizer against a real llama-server and a real GGUF model.

Opt-in: skipped unless these are set, because they need a llama.cpp install
and model files no CI machine has::

    $env:INVOICE_AI_TEST_LLAMA_SERVER = "C:\\tools\\llama.cpp\\llama-server.exe"
    $env:INVOICE_AI_TEST_GGUF_MODEL   = "C:\\models\\nuqta\\nuqta-ocr-4b-Q4_K_M.gguf"
    $env:INVOICE_AI_TEST_GGUF_MMPROJ  = "C:\\models\\nuqta\\nuqta-ocr-4b-mmproj-f16.gguf"
    $env:INVOICE_AI_TEST_EXPECT_READING = "1"   # only for a real OCR model
    pytest tests/test_llamacpp_real_server.py -v

Without INVOICE_AI_TEST_EXPECT_READING the tests check everything except
whether the model reads well -- they pass with a tiny random-weight model of
the same architecture, which is how the plumbing was verified before the real
weights were available. With it, they also require the model to read a
cleanly rendered number correctly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import requests
from PIL import Image, ImageDraw, ImageFont

from core.domain.geometry import BoundingBox
from core.domain.ocr import RegionCrop, TextRegion
from core.domain.roles import ContentKind
from ocr.decoding_policy import DATE_CHARS, NUMBER_CHARS
from ocr.recognizers.registry import recognizer_registry

SERVER = os.environ.get("INVOICE_AI_TEST_LLAMA_SERVER")
MODEL = os.environ.get("INVOICE_AI_TEST_GGUF_MODEL")
MMPROJ = os.environ.get("INVOICE_AI_TEST_GGUF_MMPROJ")
EXPECT_READING = os.environ.get("INVOICE_AI_TEST_EXPECT_READING") == "1"

pytestmark = pytest.mark.skipif(
    not (SERVER and MODEL and MMPROJ),
    reason="set INVOICE_AI_TEST_LLAMA_SERVER, INVOICE_AI_TEST_GGUF_MODEL and INVOICE_AI_TEST_GGUF_MMPROJ",
)


@pytest.fixture(scope="module")
def recognizer():
    built = recognizer_registry.create(
        "llamacpp",
        model_path=MODEL,
        mmproj_path=MMPROJ,
        server_path=SERVER,
        gpu_layers=os.environ.get("INVOICE_AI_LLAMA_GPU_LAYERS", "auto"),
        timeout_s=300,
    )
    yield built
    built.close()


def rendered(text: str, kind: ContentKind) -> RegionCrop:
    """A clean, cell-sized crop of `text`, as the padded cropper would hand it over."""
    image = Image.new("L", (360, 120), 255)
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default(size=64)
    ImageDraw.Draw(image).text((20, 20), text, fill=0, font=font)
    return RegionCrop(region=TextRegion(bbox=BoundingBox(0, 0, 360, 120), content_kind=kind), image=np.asarray(image))


def test_the_server_is_up_with_image_support(recognizer):
    server = recognizer.server
    assert recognizer.is_ready()
    assert server.build_info

    session = requests.Session()
    session.trust_env = False
    # Private: nothing without the per-launch key gets an answer.
    assert session.get(f"{server.base_url}/props", timeout=10).status_code == 401
    props = session.get(f"{server.base_url}/props", headers=server._headers(), timeout=10).json()
    assert props["modalities"]["vision"] is True


def test_a_number_cell_is_read_within_its_character_set(recognizer):
    reading = recognizer._backend.read(Image.fromarray(rendered("12345", ContentKind.NUMBER).image), "number")

    assert all(ch in NUMBER_CHARS for ch in reading.text), reading.text
    assert reading.confidence is not None and 0.0 <= reading.confidence <= 1.0
    if EXPECT_READING:
        from string_matching.normalization import digit_sequence

        assert digit_sequence(reading.text) == "12345", reading


def test_a_date_cell_is_read_within_its_character_set(recognizer):
    reading = recognizer._backend.read(Image.fromarray(rendered("7/8/2026", ContentKind.DATE).image), "date")
    assert all(ch in DATE_CHARS for ch in reading.text), reading.text


def test_crops_go_through_validation_and_come_back_in_order(recognizer):
    crops = [rendered("12", ContentKind.NUMBER), rendered("00010", ContentKind.CODE), rendered("x", ContentKind.ANY)]
    readings = recognizer.read_all(crops)

    assert len(readings) == 3
    for reading in readings:
        assert isinstance(reading.text, str)
        assert reading.confidence is None or 0.0 <= reading.confidence <= 1.0
