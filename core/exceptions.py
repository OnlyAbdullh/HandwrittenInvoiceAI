from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline-related errors."""


class ConfigurationError(PipelineError):
    """Raised when config is invalid or references an unknown component name."""


class StepExecutionError(PipelineError):
    """Raised when a preprocessing step fails during `apply`."""

    def __init__(self, step_name: str, original: Exception):
        super().__init__(f"Step '{step_name}' failed: {original}")
        self.step_name = step_name
        self.original = original


class RecognitionTimeoutError(PipelineError):
    """A recognizer backend stopped answering within its timeout.

    Raised after the backend has been stopped, so the next request starts a
    fresh one instead of queueing behind the hung one.
    """


class ModelFileError(PipelineError):
    """A model file exists but is not what the configuration says it is.

    Wrong format, truncated, or the two files of a vision model swapped. This
    is the machine's state, not the request's, so it is deliberately *not* a
    ValueError: the HTTP layer reports ValueErrors raised while building a
    pipeline as the client's invalid configuration (422), and a broken model
    install as "engine not ready" (503).
    """


class InferenceServerError(PipelineError):
    """A local inference server could not be started, or answered with an error.

    Carries the tail of the server's own log when there is one, because the
    server's reason ("failed to load model", "out of memory", "unknown model
    architecture") is the only useful part of the failure.
    """

    def __init__(self, message: str, log_tail: str = ""):
        text = message if not log_tail else f"{message}\n--- llama-server log (tail) ---\n{log_tail}"
        super().__init__(text)
        self.log_tail = log_tail


class NotImplementedStrategyError(PipelineError, NotImplementedError):
    """Raised by a strategy variant that was deliberately left unbuilt
    (e.g. an alternative algorithm not yet implemented), as opposed to a
    step that simply hasn't been started yet.

    Inherits from NotImplementedError too so PreprocessingPipeline.run()
    still lets it propagate unwrapped (its existing `except
    NotImplementedError: raise` special case applies here for free),
    while also being identifiable/catchable as "intentionally unbuilt"
    via the PipelineError hierarchy.
    """

    def __init__(self, step_name: str, strategy_name: str, message: str = ""):
        text = f"Step '{step_name}' strategy '{strategy_name}' is not implemented."
        if message:
            text += f" {message}"
        super().__init__(text)
        self.step_name = step_name
        self.strategy_name = strategy_name
