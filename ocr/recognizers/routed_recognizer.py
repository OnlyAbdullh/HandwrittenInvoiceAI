"""Several recognizers behind one: a model per cell kind, and an optional
second opinion when the first model is unsure.

Two of the combination strategies `tools/analyze_benchmark.py` measures,
made selectable from configuration:

* **Routing by kind** -- numbers to one model, names to another. The cell's
  :class:`~core.domain.roles.ContentKind` picks the model, exactly as it picks
  the prompt inside a single recognizer.
* **Confidence fallback** -- when the routed model's confidence is below a
  threshold (or it has none), the cell is read again by the fallback model and
  *that* reading is used. Confidences from different models are not on one
  scale, so the two are never compared with each other; the threshold is a
  property of the first model only, which is also how the analysis chose it.

Neither is on by default, and neither should be until the analysis shows it
beats the best single model on held-out invoices by more than noise: each
extra model costs its load time and its memory (two VLMs on a 4 GB GPU will not
both fit; on the CPU they fit in 24 GB of RAM but each is slower).

    recognizer:
      name: routed
      params:
        recognizers:
          qwen:  {name: qwen, params: {...}}
          nuqta: {name: llamacpp, params: {...}}
        default: qwen
        routes: {arabic_text: nuqta}          # optional, kind -> alias
        fallback: {to: nuqta, below_confidence: 0.6}   # optional

A Composite: it implements the same :class:`TextRecognizer` protocol it is made
of, so no flow knows it is talking to more than one model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.domain.ocr import Recognition, RegionCrop
from core.domain.roles import ContentKind
from core.exceptions import ConfigurationError
from core.interfaces.text_recognizer import TextRecognizer
from ocr.recognizers.registry import recognizer_registry

_KINDS = {k.value for k in ContentKind}


def _check_spec(alias: str, spec: Any) -> tuple[str, dict]:
    if not isinstance(spec, Mapping) or not isinstance(spec.get("name"), str):
        raise ConfigurationError(f"routed recognizer: '{alias}' must be {{name: <recognizer>, params: {{...}}}}.")
    if spec["name"] == "routed":
        raise ConfigurationError("routed recognizer: a routed recognizer cannot contain another one.")
    unknown = set(spec) - {"name", "params"}
    if unknown:
        raise ConfigurationError(f"routed recognizer: '{alias}' has unknown key(s) {sorted(unknown)}.")
    return spec["name"], dict(spec.get("params") or {})


@recognizer_registry.register("routed")
class RoutedRecognizer:
    name = "routed"

    def __init__(
        self,
        recognizers: Mapping[str, Mapping[str, Any]],
        default: str,
        routes: Mapping[str, str] | None = None,
        fallback: Mapping[str, Any] | None = None,
    ):
        if not recognizers:
            raise ConfigurationError("routed recognizer: `recognizers` names no recognizer.")
        routes = dict(routes or {})
        aliases = set(recognizers)

        bad_kinds = sorted(set(routes) - _KINDS)
        if bad_kinds:
            raise ConfigurationError(
                f"routed recognizer: unknown cell kind(s) {bad_kinds} in `routes`. Kinds: {sorted(_KINDS)}."
            )
        used = {default, *routes.values()}
        self._fallback_alias: str | None = None
        self._threshold = 0.0
        if fallback is not None:
            unknown = set(fallback) - {"to", "below_confidence"}
            if unknown or "to" not in fallback or "below_confidence" not in fallback:
                raise ConfigurationError(
                    "routed recognizer: `fallback` is {to: <alias>, below_confidence: <0..1>}."
                )
            threshold = float(fallback["below_confidence"])
            if not 0.0 < threshold <= 1.0:
                raise ConfigurationError("routed recognizer: `fallback.below_confidence` must be in (0, 1].")
            self._fallback_alias = str(fallback["to"])
            self._threshold = threshold
            used.add(self._fallback_alias)
        missing = sorted(used - aliases)
        if missing:
            raise ConfigurationError(
                f"routed recognizer: {missing} named but not defined under `recognizers` ({sorted(aliases)})."
            )
        unused = sorted(aliases - used)
        if unused:
            # Each one would load a model nothing reads with.
            raise ConfigurationError(f"routed recognizer: {unused} defined but never used.")

        # Built only after the whole configuration has been checked, so a typo
        # never costs a model load.
        self._members: dict[str, TextRecognizer] = {}
        try:
            for alias in sorted(aliases):
                name, params = _check_spec(alias, recognizers[alias])
                self._members[alias] = recognizer_registry.create(name, **params)
        except Exception:
            self.close()
            raise
        self._default = default
        self._routes = routes

    @property
    def description(self) -> str:
        routes = ", ".join(f"{kind}->{alias}" for kind, alias in sorted(self._routes.items()))
        parts = [f"default {self._default}"] + ([routes] if routes else [])
        if self._fallback_alias:
            parts.append(f"below {self._threshold} -> {self._fallback_alias}")
        return "; ".join(parts)

    # -- lifecycle ----------------------------------------------------------

    def is_ready(self) -> bool:
        for member in self._members.values():
            probe = getattr(member, "is_ready", None)
            if callable(probe) and not probe():
                return False
        return True

    def warmup(self) -> None:
        for member in self._members.values():
            warm = getattr(member, "warmup", None)
            if callable(warm):
                warm()

    def close(self) -> None:
        for member in getattr(self, "_members", {}).values():
            close = getattr(member, "close", None)
            if callable(close):
                close()

    # -- reading ------------------------------------------------------------

    def _alias_for(self, crop: RegionCrop) -> str:
        kind = crop.content_kind.value if crop.content_kind is not None else None
        return self._routes.get(kind, self._default)

    def _unsure(self, reading: Recognition) -> bool:
        return reading.confidence is None or reading.confidence < self._threshold

    def read(self, crop: RegionCrop) -> Recognition:
        return self.read_all([crop])[0]

    def read_all(self, crops: Sequence[RegionCrop]) -> list[Recognition]:
        readings: list[Recognition | None] = [None] * len(crops)

        groups: dict[str, list[int]] = {}
        for i, crop in enumerate(crops):
            groups.setdefault(self._alias_for(crop), []).append(i)
        for alias, indices in groups.items():
            for i, reading in zip(indices, self._read_with(alias, [crops[i] for i in indices]), strict=True):
                readings[i] = reading

        if self._fallback_alias is not None:
            # A cell already read by the fallback model is not read by it twice.
            retry = [i for i, r in enumerate(readings)
                     if self._alias_for(crops[i]) != self._fallback_alias and self._unsure(r)]
            if retry:
                for i, reading in zip(retry, self._read_with(self._fallback_alias, [crops[i] for i in retry]), strict=True):
                    readings[i] = reading
        return readings  # type: ignore[return-value]

    def _read_with(self, alias: str, crops: list[RegionCrop]) -> list[Recognition]:
        member = self._members[alias]
        result = list(member.read_all(crops))
        if len(result) != len(crops):
            raise RuntimeError(f"recognizer '{alias}' returned {len(result)} readings for {len(crops)} crops")
        return result
