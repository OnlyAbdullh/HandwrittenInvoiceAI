"""Importing this package self-registers every text recognizer that can be imported.

`echo` needs nothing and is always available -- it is what lets the flow
machinery be tested without a model. The rest are imported defensively, for the
same reason the heavy OCR engines are: a machine on the light path has no reason
to have torch installed, and an unconditional import would stop the service
starting there.
"""

from __future__ import annotations

import logging

from ocr.recognizers import echo_recognizer  # noqa: F401  (no dependencies)
from ocr.recognizers import routed_recognizer  # noqa: F401  (builds others; needs nothing itself)
from ocr.recognizers.registry import recognizer_registry

logger = logging.getLogger(__name__)

_OPTIONAL_RECOGNIZERS = (
    ("qwen_recognizer", "qwen", "transformers / torch"),
    # Needs only `requests` and Pillow in this process; the model itself runs
    # in the separately installed llama-server binary, which is looked for
    # when the recognizer is built, not when it is imported.
    ("llamacpp_recognizer", "llamacpp", "requests"),
)

for _module_name, _registered, _requirement in _OPTIONAL_RECOGNIZERS:
    try:
        __import__(f"ocr.recognizers.{_module_name}")
    except Exception as _exc:  # noqa: BLE001 - any import failure disables it
        recognizer_registry.mark_unavailable(_registered, f"needs {_requirement} ({_exc})")
        logger.info(
            "Text recognizer '%s' unavailable (needs %s): %s",
            _module_name,
            _requirement,
            _exc,
        )
