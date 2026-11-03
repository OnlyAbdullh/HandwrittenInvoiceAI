"""Just enough of the GGUF format to check a model file before loading it.

A llama.cpp vision model is two files: the language model, and the ``mmproj``
that turns an image into tokens the language model can read. Point the
configuration at the wrong file -- the two swapped, a text-only model with no
projector, an HTML error page saved with a ``.gguf`` name, a download that
stopped half way -- and ``llama-server`` fails minutes into loading with a
message about tensors. Reading the header takes milliseconds and says which
file is wrong, and why.

Only the metadata section is read, and only until the keys that identify the
file have been seen. Tensor data is never touched.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
(header: magic, version, tensor count, key/value count; then the key/value
pairs, each a length-prefixed key, a type tag and a value).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from core.exceptions import ModelFileError

GGUF_MAGIC = b"GGUF"

# The architecture a projector file declares. Every llama.cpp mmproj, whatever
# the model family, is a "clip" file; the family is in `clip.projector_type`.
PROJECTOR_ARCHITECTURE = "clip"

# Value type tags (gguf.md, "GGUFValueType").
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL = range(8)
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12

_SCALARS = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h",
    _UINT32: "<I", _INT32: "<i", _FLOAT32: "<f", _BOOL: "<?",
    _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d",
}

# The keys worth reporting. Everything else -- above all the tokenizer's
# vocabulary, a 150k-entry string array -- is skipped without being decoded.
_WANTED = (
    "general.architecture",
    "general.name",
    "general.file_type",
    "clip.has_vision_encoder",
    "clip.projector_type",
    "clip.vision.projector_type",
    "clip.vision.image_min_pixels",
    "clip.vision.image_max_pixels",
)

# A sanity bound on any one string. Metadata strings are names and templates;
# a "length" past this means the file is not GGUF however it starts.
_MAX_STRING_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class GGUFInfo:
    path: Path
    version: int
    tensor_count: int
    size_bytes: int
    metadata: dict = field(default_factory=dict)

    @property
    def architecture(self) -> str | None:
        return self.metadata.get("general.architecture")

    @property
    def is_projector(self) -> bool:
        return self.architecture == PROJECTOR_ARCHITECTURE

    @property
    def projector_type(self) -> str | None:
        return self.metadata.get("clip.projector_type") or self.metadata.get(
            "clip.vision.projector_type"
        )

    @property
    def image_min_pixels(self) -> int | None:
        """The smallest image the projector's model was set up to read, if it says.

        A model that declares one (Nuqta's mmproj asks for about 1 megapixel)
        should be fed at that size; llama.cpp applies it unless told otherwise.
        """
        value = self.metadata.get("clip.vision.image_min_pixels")
        return int(value) if value else None

    @property
    def has_vision_encoder(self) -> bool | None:
        value = self.metadata.get("clip.has_vision_encoder")
        return None if value is None else bool(value)


class _Reader:
    def __init__(self, handle: BinaryIO, path: Path):
        self._handle = handle
        self._path = path

    def read(self, size: int) -> bytes:
        data = self._handle.read(size)
        if len(data) != size:
            raise ModelFileError(
                f"'{self._path}' ends inside its metadata header. The file is truncated -- "
                "most often an interrupted download. Download it again."
            )
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        length = self.unpack("<Q")
        if length > _MAX_STRING_BYTES:
            raise ModelFileError(f"'{self._path}' has a corrupt metadata header.")
        return self.read(length).decode("utf-8", errors="replace")

    def skip(self, size: int) -> None:
        self._handle.seek(size, 1)

    def value(self, value_type: int, keep: bool):
        if value_type in _SCALARS:
            return self.unpack(_SCALARS[value_type])
        if value_type == _STRING:
            if keep:
                return self.string()
            self.skip(self.unpack("<Q"))
            return None
        if value_type == _ARRAY:
            element_type = self.unpack("<I")
            count = self.unpack("<Q")
            if element_type in _SCALARS:
                self.skip(struct.calcsize(_SCALARS[element_type]) * count)
            else:
                for _ in range(count):
                    self.value(element_type, keep=False)
            return None
        raise ModelFileError(
            f"'{self._path}' has an unknown metadata value type {value_type}; it is either "
            "corrupt or written by a newer GGUF version than this check understands."
        )


def _identified(metadata: dict) -> bool:
    """Whether the keys that identify this kind of file have all been read."""
    architecture = metadata.get("general.architecture")
    if architecture is None or "general.name" not in metadata:
        return False
    # A projector's metadata is small and its image-size keys can come anywhere
    # in it, so it is always read to the end.
    return architecture != PROJECTOR_ARCHITECTURE


def read_gguf_info(path: str | Path) -> GGUFInfo:
    """The identifying metadata of a GGUF file.

    Raises FileNotFoundError when there is no such file, and ModelFileError
    when there is one but it is not a readable GGUF file.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Model file not found: {file}")
    if not file.is_file():
        raise ModelFileError(f"'{file}' is a directory; expected a single .gguf file.")

    size = file.stat().st_size

    with file.open("rb") as handle:
        magic = handle.read(4)
        if magic != GGUF_MAGIC:
            raise ModelFileError(
                f"'{file}' is not a GGUF file (it starts with {magic!r}, not b'GGUF'). "
                "If it was downloaded through a browser, the download may have saved an "
                "error page instead of the model."
            )

        reader = _Reader(handle, file)
        version = reader.unpack("<I")
        if version < 2:
            raise ModelFileError(
                f"'{file}' is GGUF version {version}, which current llama.cpp builds no longer "
                "load. Download a current conversion of the model."
            )

        tensor_count = reader.unpack("<Q")
        kv_count = reader.unpack("<Q")

        metadata: dict = {}
        for _ in range(kv_count):
            key = reader.string()
            value_type = reader.unpack("<I")
            keep = key in _WANTED
            value = reader.value(value_type, keep=keep)
            if keep:
                metadata[key] = value
            if _identified(metadata):
                # Everything identifying has been seen; the rest is tokenizer
                # and hyperparameters.
                break

    return GGUFInfo(
        path=file, version=version, tensor_count=tensor_count, size_bytes=size, metadata=metadata
    )


def check_vision_model_pair(model_path: str | Path, mmproj_path: str | Path) -> tuple[GGUFInfo, GGUFInfo]:
    """Both files of a llama.cpp vision model, checked against each other.

    Returns (model info, projector info). Raises FileNotFoundError or
    ModelFileError naming the file at fault.
    """
    model = read_gguf_info(model_path)
    projector = read_gguf_info(mmproj_path)

    if model.path.resolve() == projector.path.resolve():
        raise ModelFileError(
            f"model_path and mmproj_path both point at '{model.path}'. They are two different "
            "files: the language model, and the *mmproj* vision projector."
        )
    if model.is_projector and not projector.is_projector:
        raise ModelFileError(
            f"model_path and mmproj_path are swapped: '{model.path}' is the vision projector "
            f"and '{projector.path}' is the language model."
        )
    if model.is_projector:
        raise ModelFileError(
            f"model_path '{model.path}' is a vision projector (mmproj), not a language model."
        )
    if not projector.is_projector:
        raise ModelFileError(
            f"mmproj_path '{projector.path}' is not a vision projector: its architecture is "
            f"'{projector.architecture}'. The projector file is usually named '*mmproj*.gguf'."
        )
    if projector.has_vision_encoder is False:
        raise ModelFileError(
            f"mmproj_path '{projector.path}' has no vision encoder (it is an audio-only "
            "projector). Invoice OCR needs the image projector."
        )
    return model, projector
