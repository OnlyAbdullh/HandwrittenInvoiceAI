"""A GGUF vision-language model as a TextRecognizer, run by llama.cpp.

Selected in configuration exactly like the transformers-backed ``qwen``
recognizer, and interchangeable with it::

    ocr:
      recognizer:
        name: llamacpp
        params:
          model_path:  "${INVOICE_AI_NUQTA_MODEL:-OCR_Models/Nuqta-Arabic-OCR-4B-GGUF/nuqta-ocr-4b-Q4_K_M.gguf}"
          mmproj_path: "${INVOICE_AI_NUQTA_MMPROJ:-OCR_Models/Nuqta-Arabic-OCR-4B-GGUF/nuqta-ocr-4b-mmproj-f16.gguf}"

The flow, cropper, blank-cell gate, prompts, output validation, parser and wire
format are all unchanged -- this supplies a different runtime behind the same
:class:`~ocr.recognizers.prompted_recognizer.PromptedRecognizer`. The service
starts ``llama-server`` itself, once, when the pipeline is built (at service
startup, through warmup) and stops it on shutdown; see
:mod:`ocr.llamacpp.server`.

Hardware, for the 4 GB laptop GPU this was written for: llama.cpp's default
(``gpu_layers: auto``) fits as many layers as the free VRAM holds -- keeping
1 GB spare -- and runs the rest on the CPU, so a model that does not fit
degrades to slower, not to an out-of-memory crash. ``gpu_layers: 0`` is CPU
only; ``mmproj_offload: false`` keeps the vision encoder on the CPU to free VRAM
for language-model layers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.domain.roles import ContentKind
from ocr.decoding_policy import MAX_IMAGE_SIDE
from ocr.engines.qwen_vlm_engine import PROMPTS
from ocr.llamacpp.client import LlamaCppVisionBackend
from ocr.llamacpp.gguf import check_vision_model_pair
from ocr.llamacpp.server import (
    PROJECT_ROOT,
    LlamaServer,
    LlamaServerSettings,
    check_extra_args,
    resolve_executable,
)
from ocr.recognizers.prompted_recognizer import (
    PromptedRecognizer,
    check_batch_settings,
    require_prompt_for_every_kind,
)
from ocr.recognizers.registry import recognizer_registry

logger = logging.getLogger(__name__)


def _project_path(value: str, field: str) -> Path:
    text = (value or "").strip()
    if not text:
        raise FileNotFoundError(f"llamacpp recognizer: {field} is not set.")
    path = Path(text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _launcher(server_path: str) -> tuple[str, ...]:
    """The command prefix that starts llama-server.

    One element in production. Its own function so the tests can launch a
    Python stand-in for the server without the configuration having to know
    about interpreters.
    """
    return (resolve_executable(server_path),)


# The Hugging Face Qwen-VL processor's floor (65536 px at 32 px per merged token).
HF_QWEN_VL_MIN_TOKENS = 64


def resolve_image_min_tokens(setting, projector_info) -> int | None:
    """The --image-min-tokens to pass, or None to pass nothing.

    "auto" respects a floor the projector file declares -- the model's own
    authors chose it, and overriding it shrank Nuqta's input to a fifteenth of
    its training size -- and otherwise applies the HF processor's floor.
    """
    if setting == "auto":
        if projector_info.image_min_pixels:
            logger.info(
                "Projector declares a minimum image size of %d px; using it.", projector_info.image_min_pixels
            )
            return None
        return HF_QWEN_VL_MIN_TOKENS
    return setting


def _gpu_layers(value) -> str | None:
    """None for llama.cpp's own automatic fit; otherwise its -ngl argument.

    Accepts what a YAML file or an environment variable produces: an integer,
    or the strings "auto", "all" or a number.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"llamacpp recognizer: gpu_layers must be 'auto', 'all' or a number, got {value!r}.")
    text = str(value).strip().lower()
    if text in ("", "auto"):
        return None
    if text == "all":
        return "all"
    if text.lstrip("-").isdigit() and int(text) >= 0:
        return str(int(text))
    raise ValueError(f"llamacpp recognizer: gpu_layers must be 'auto', 'all' or a number >= 0, got {value!r}.")


@recognizer_registry.register("llamacpp")
class LlamaCppRecognizer(PromptedRecognizer):
    name = "llamacpp"

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        server_path: str = "llama-server",
        default_kind: str = ContentKind.ANY.value,
        # One crop per request: the server has one slot, and a batch that
        # fails is retried crop by crop -- re-reading the ones that succeeded.
        batch_size: int = 1,
        gpu_layers: int | str | None = "auto",
        mmproj_offload: bool = True,
        ctx_size: int = 4096,
        threads: int | None = None,
        # The smallest size a crop is enlarged to, in image tokens. "auto": the
        # floor the projector file declares, when it declares one (Nuqta's asks
        # for ~1 MP -- its training size), else 64, the Hugging Face Qwen-VL
        # processor's floor (65536 px), since llama.cpp's own floor for these
        # projectors is 8 tokens and would hand a small cell over at a fraction
        # of the resolution the model reads at. A number forces it; null leaves
        # it to llama.cpp entirely.
        image_min_tokens: int | str | None = "auto",
        port: int = 0,
        max_new_tokens: int = 64,
        max_image_side: int = MAX_IMAGE_SIDE,
        constrained_decoding: bool = True,
        disable_thinking: bool = True,
        prompts: dict | None = None,
        # None: the per-kind policy (1.1 on free text). A number applies to every
        # kind; 1.0 is off, which is what Nuqta's model card asks for.
        repetition_penalty: float | None = None,
        timeout_s: float = 120.0,
        startup_timeout_s: float = 600.0,
        extra_args: list[str] | None = None,
    ):
        # Everything checkable is checked before the server starts, so a bad
        # setting fails in milliseconds instead of after a model load.
        merged_prompts = {**PROMPTS, **(prompts or {})}
        unknown = sorted(set(prompts or {}) - {k.value for k in ContentKind})
        if unknown:
            raise ValueError(
                f"llamacpp recognizer: prompts given for unknown content kind(s) {unknown}. "
                f"Valid: {[k.value for k in ContentKind]}."
            )
        require_prompt_for_every_kind(merged_prompts, self.name)
        kind = check_batch_settings(self.name, default_kind, batch_size)

        if ctx_size < 1024:
            raise ValueError(f"llamacpp recognizer: ctx_size must be >= 1024, got {ctx_size}.")
        if threads is not None and threads < 1:
            raise ValueError(f"llamacpp recognizer: threads must be >= 1, got {threads}.")
        if not (image_min_tokens is None or image_min_tokens == "auto"
                or (isinstance(image_min_tokens, int) and not isinstance(image_min_tokens, bool) and image_min_tokens >= 1)):
            raise ValueError(
                f"llamacpp recognizer: image_min_tokens must be 'auto', null or a number >= 1, got {image_min_tokens!r}."
            )
        if repetition_penalty is not None and repetition_penalty <= 0:
            raise ValueError(f"llamacpp recognizer: repetition_penalty must be > 0, got {repetition_penalty}.")
        if port != 0 and not 1024 <= port <= 65535:
            raise ValueError(f"llamacpp recognizer: port must be 0 (automatic) or 1024-65535, got {port}.")
        if max_new_tokens < 1 or max_image_side < 64:
            raise ValueError("llamacpp recognizer: max_new_tokens must be >= 1 and max_image_side >= 64.")

        model = _project_path(model_path, "model_path")
        mmproj = _project_path(mmproj_path, "mmproj_path")
        # Missing or wrong files raise FileNotFoundError / ModelFileError --
        # "this machine's model install is broken", reported as not-ready
        # rather than as the request's fault.
        model_info, projector_info = check_vision_model_pair(model, mmproj)
        logger.info(
            "llamacpp model: %s (architecture %s); projector: %s (%s).",
            model_info.path.name,
            model_info.architecture,
            projector_info.path.name,
            projector_info.projector_type or "projector type not stated",
        )
        min_tokens = resolve_image_min_tokens(image_min_tokens, projector_info)

        settings = LlamaServerSettings(
            launcher=_launcher(server_path),
            model_path=model,
            mmproj_path=mmproj,
            ctx_size=ctx_size,
            gpu_layers=_gpu_layers(gpu_layers),
            mmproj_offload=mmproj_offload,
            threads=threads,
            image_min_tokens=min_tokens,
            port=port,
            extra_args=check_extra_args(extra_args or []),
            startup_timeout_s=startup_timeout_s,
        )
        server = LlamaServer(settings, request_timeout_s=timeout_s)
        backend = LlamaCppVisionBackend(
            server,
            prompts=merged_prompts,
            max_new_tokens=max_new_tokens,
            max_image_side=max_image_side,
            constrained_decoding=constrained_decoding,
            disable_thinking=disable_thinking,
            timeout_s=timeout_s,
            repetition_penalty=repetition_penalty,
        )

        # Started now, like the transformers worker, so the model load happens
        # once at service startup rather than inside the first invoice.
        server.start()
        super().__init__(backend, kind, batch_size)

    @property
    def server(self) -> LlamaServer:
        return self._backend.server
