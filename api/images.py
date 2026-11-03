"""Decoding uploads and encoding the diagnostic images that go back out."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np


class CorruptFileError(Exception):
    """The bytes could not be decoded as an image."""


class UnsupportedFormatError(Exception):
    """A format the service knows, but whose optional decoder is not installed."""


# A PDF page is rendered at this scale (1.0 = 72 DPI). 2.5 is ~180 DPI: enough
# for handwriting, small enough not to dominate request time.
PDF_RENDER_SCALE = 2.5


@dataclass
class LoadedDocument:
    image: np.ndarray
    filename: str
    page_count: int = 1
    page_used: int = 1


def _decode_heic(data: bytes, filename: str) -> np.ndarray:
    try:
        import pillow_heif
        from PIL import Image
    except ImportError as exc:
        raise UnsupportedFormatError(
            "HEIC/HEIF needs the optional 'pillow-heif' package (requirements-optional.txt)."
        ) from exc

    try:
        pillow_heif.register_heif_opener()
        import io

        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        raise CorruptFileError(f"'{filename}' could not be decoded as HEIC: {exc}") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _decode_pdf(data: bytes, filename: str, page: int) -> tuple[np.ndarray, int, int]:
    try:
        import pypdfium2
    except ImportError as exc:
        raise UnsupportedFormatError(
            "PDF needs the optional 'pypdfium2' package (requirements-optional.txt)."
        ) from exc

    try:
        pdf = pypdfium2.PdfDocument(data)
        count = len(pdf)
        used = min(max(page, 1), count)
        rgb = np.asarray(pdf[used - 1].render(scale=PDF_RENDER_SCALE).to_pil().convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        raise CorruptFileError(f"'{filename}' could not be rendered as a PDF: {exc}") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), count, used


def load_document(data: bytes, filename: str, page: int = 1) -> LoadedDocument:
    """Decode an uploaded document into a BGR image.

    PNG and JPEG decode through OpenCV, which applies the EXIF orientation, so a
    phone photo taken sideways arrives upright. HEIC (iPhone) and PDF are
    decoded when their optional packages are installed; `page` selects the PDF
    page and `page_count`/`page_used` report what was found and read.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension == "pdf":
        image, count, used = _decode_pdf(data, filename, page)
        return LoadedDocument(image=image, filename=filename, page_count=count, page_used=used)
    if extension in ("heic", "heif"):
        return LoadedDocument(image=_decode_heic(data, filename), filename=filename)

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise CorruptFileError(
            f"'{filename}' could not be decoded as an image. It may be truncated "
            "or not actually an image file."
        )

    return LoadedDocument(image=image, filename=filename)


def encode_png_base64(image: np.ndarray, max_width: int = 1200) -> str | None:
    """Base64 PNG of `image`, downscaled to `max_width`.

    Returns None rather than raising: a diagnostic image that fails to encode
    is never worth failing a good extraction over, and the caller turns a None
    into a `DEBUG_IMAGE_UNAVAILABLE` warning on an otherwise-200 response.

    Downscaling is never upscaling -- an image already narrower than the limit
    is encoded as it is.
    """
    try:
        if image is None or image.size == 0:
            return None

        prepared = _downscale(image, max_width)

        ok, buffer = cv2.imencode(".png", prepared)
        if not ok:
            return None

        return base64.b64encode(buffer.tobytes()).decode("ascii")
    except Exception:  # noqa: BLE001 - any encoding failure is "unavailable"
        return None


def _downscale(image: np.ndarray, max_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max_width <= 0 or width <= max_width:
        return image

    scale = max_width / width
    return cv2.resize(
        image,
        (max_width, max(1, int(round(height * scale)))),
        # INTER_AREA is the right filter for shrinking: it averages the pixels
        # that collapse together instead of point-sampling one of them, which
        # is what keeps thin Arabic strokes visible in the downscaled preview.
        interpolation=cv2.INTER_AREA,
    )
