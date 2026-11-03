"""Shared helpers for the llama.cpp backend tests.

`write_gguf` produces a real GGUF header -- magic, version, counts, typed
key/value pairs -- with no tensor data, which is all the file checks read. The
fake server stands in for llama-server; see tests/fixtures/fake_llama_server.py.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_llama_server.py"

_UINT32, _BOOL, _STRING, _ARRAY = 4, 7, 8, 9


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _value(value) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I", _BOOL) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", _UINT32) + struct.pack("<I", value)
    if isinstance(value, str):
        return struct.pack("<I", _STRING) + _string(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return struct.pack("<I", _ARRAY) + struct.pack("<IQ", _STRING, len(value)) + b"".join(
            _string(v) for v in value
        )
    if isinstance(value, list):
        return struct.pack("<I", _ARRAY) + struct.pack("<IQ", _UINT32, len(value)) + b"".join(
            struct.pack("<I", v) for v in value
        )
    raise TypeError(value)


def write_gguf(path: Path, metadata: list[tuple[str, object]], version: int = 3) -> Path:
    body = b"".join(_string(key) + _value(value) for key, value in metadata)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", version, 0, len(metadata)) + body)
    return path


def write_model(path: Path, architecture: str = "qwen35") -> Path:
    # Arrays ahead of general.name, the way a converted model lays out its
    # tokenizer, so the reader has to skip them.
    return write_gguf(
        path,
        [
            ("general.architecture", architecture),
            ("tokenizer.ggml.tokens", ["<|im_start|>", "ب", "١", "<|im_end|>"]),
            ("tokenizer.ggml.token_type", [1, 1, 1, 3]),
            ("general.name", "Nuqta Arabic OCR 4B (test header)"),
            ("general.file_type", 15),
        ],
    )


def write_projector(path: Path, vision: bool = True, min_pixels: int | None = None) -> Path:
    metadata = [
        ("general.architecture", "clip"),
        ("general.name", "Nuqta mmproj (test header)"),
        ("clip.has_vision_encoder", vision),
        ("clip.projector_type", "qwen3vl_merger"),
    ]
    if min_pixels is not None:
        metadata.append(("clip.vision.image_min_pixels", min_pixels))
    return write_gguf(path, metadata)


def fake_launcher() -> tuple[str, ...]:
    return (sys.executable, str(FAKE_SERVER))


def read_record(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
