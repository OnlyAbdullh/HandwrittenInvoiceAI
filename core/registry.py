"""Generic name -> class registry used by every pluggable family
(preprocessing steps, OCR engines, table extractors, string matchers,
output formatters).

Each family gets its own Registry instance. Concrete implementations
self-register via the `@registry.register("name")` decorator at import
time; factories then build instances purely from the string name found in
config. Adding a new implementation means adding one file -- nothing that
already exists needs to change (Open/Closed principle).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Generic, Mapping, Type, TypeVar

from core.exceptions import ConfigurationError

T = TypeVar("T")


def accepted_params(target: Callable) -> set[str] | None:
    """The keyword names `target` accepts, or None if it takes ``**kwargs``."""
    parameters = inspect.signature(target).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return None
    return {
        p.name
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def check_params(kind: str, name: str, target: Callable, params: Mapping[str, Any]) -> None:
    """Reject configuration keys `target` does not take, naming the ones it does.

    Components used to accept ``**params`` and silently drop anything unknown,
    so a misspelt key -- or a key a newer config expected a component to honour
    -- did nothing at all, and the run looked perfectly healthy. A configuration
    error should fail when the configuration is loaded, not be discovered weeks
    later as an unexplained accuracy difference.
    """
    accepted = accepted_params(target)
    if accepted is None:
        return

    unknown = sorted(set(params) - accepted)
    if unknown:
        valid = ", ".join(sorted(accepted)) or "(none)"
        raise ConfigurationError(
            f"Unknown parameter(s) {unknown} for {kind} '{name}'. Accepted: {valid}."
        )


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self._kind = kind
        self._registry: Dict[str, Type[T]] = {}
        # Optional implementations whose import failed, and why. Named in the
        # "unknown" error so the fix ("pip install surya-ocr") is in the message.
        self._unavailable: Dict[str, str] = {}

    def mark_unavailable(self, name: str, reason: str) -> None:
        """Record that `name` exists but could not be imported on this machine."""
        if name not in self._registry:
            self._unavailable[name] = reason

    def register(self, name: str):
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(f"{self._kind} '{name}' is already registered.")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        try:
            return self._registry[name]
        except KeyError:
            available = ", ".join(sorted(self._registry)) or "(none registered)"
            if name in self._unavailable:
                raise KeyError(
                    f"{self._kind} '{name}' is not available on this machine: {self._unavailable[name]}. "
                    f"Available: {available}"
                ) from None
            raise KeyError(f"Unknown {self._kind} '{name}'. Available: {available}") from None

    def create(self, name: str, **kwargs) -> T:
        cls = self.get(name)
        check_params(self._kind, name, cls, kwargs)
        return cls(**kwargs)

    def available(self) -> list[str]:
        return sorted(self._registry)
