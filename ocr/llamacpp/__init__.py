"""Running a GGUF vision-language model through llama.cpp.

Three pieces, each knowing nothing about invoices:

* :mod:`ocr.llamacpp.gguf`    -- reads a GGUF file's header, so a wrong, swapped
  or truncated model file is reported at configuration time rather than as a
  cryptic failure inside the server.
* :mod:`ocr.llamacpp.server`  -- starts, health-checks, restarts and stops a
  private ``llama-server`` process bound to loopback.
* :mod:`ocr.llamacpp.client`  -- turns (crop, content kind) into a chat request
  with the image attached, a grammar for constrained kinds, and token
  log-probabilities, and turns the answer into a
  :class:`~core.domain.ocr.Recognition`.

The ``llamacpp`` recognizer (:mod:`ocr.recognizers.llamacpp_recognizer`) wires
them into the pipeline. Standard library plus ``requests`` and Pillow only: the
model runs in the separately installed ``llama-server`` binary, so this
backend adds no Python dependency the service did not already have.
"""
