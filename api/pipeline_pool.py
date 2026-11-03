"""Keeps built pipelines alive between requests.

Constructing a `PipelineOrchestrator` builds preprocessing pipelines, a table
extractor, a layout classifier, a string matcher -- all cheap -- and whichever
OCR components the configured flow needs, which are not: the Surya detector and
the Qwen recognizer load model weights, and the Qwen worker is a separate
process holding several GB. Since every request may carry its own
`configuration`, building an orchestrator per request would pay that cost per
invoice.

Two caches solve it, both least-recently-used and bounded:

* **OCR components**, keyed by the OCR section and the flow name together --
  the two things that decide which models get built. A request that changes a
  threshold block size gets a new orchestrator but the *same* detector and
  recognizer objects, so the weights stay loaded. Bounded by
  `max_loaded_models` (default 1): every entry is a model in memory, and on a
  4 GB GPU a second Qwen worker does not fit. An evicted entry is **closed** --
  its worker process stopped -- rather than left running unreferenced.
* **Orchestrators**, keyed by the whole configuration. In practice the desktop
  app sends one configuration for a whole batch, so this holds a single entry
  and every request after the first is a lookup.

Both caches are guarded by a lock: uvicorn runs the sync endpoint handlers on a
thread pool, so two requests can arrive at the pool genuinely concurrently.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import fields

from config.loader import load_config
from config.schema import AppConfig
from config.settings_contract import PipelineSettings
from orchestration.flows.components import OCRComponents, build_ocr_components
from orchestration.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def _key(payload) -> str:
    """A stable cache key for a pydantic model or plain dict."""
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload
    return json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)


def _components_key(config: AppConfig) -> str:
    # Keyed on the OCR section AND the flow name: the same `ocr` block builds
    # different objects under different flows -- layout_driven deliberately
    # builds no detector at all -- so the flow is part of the identity.
    return _key({"ocr": config.ocr.model_dump(), "flow": config.flow.name})


def close_components(components: OCRComponents) -> None:
    """Release whatever the components hold: worker processes, mostly."""
    for f in fields(components):
        close = getattr(getattr(components, f.name), "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - releasing must not fail a request
                logger.exception("Closing OCR component '%s' failed.", f.name)


class PipelinePool:
    def __init__(self, config_path: str, cache_size: int = 4, max_loaded_models: int = 1):
        self.config_path = config_path
        self.cache_size = max(1, cache_size)
        self.max_loaded_models = max(1, max_loaded_models)

        # The service's own defaults: what a request with no `configuration`
        # runs, what the settings page is served as its starting point, and the
        # sole source of the server-owned sections (OCR, persistence).
        self.default_config: AppConfig = load_config(config_path)

        self._lock = threading.Lock()
        self._components: "OrderedDict[str, OCRComponents]" = OrderedDict()
        self._orchestrators: "OrderedDict[str, tuple[str, PipelineOrchestrator]]" = OrderedDict()

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @property
    def default_settings(self) -> PipelineSettings:
        """The default config in the wire shape, for the settings page."""
        return PipelineSettings.from_app_config(self.default_config)

    @property
    def default_table_photometric_steps(self):
        """The table branch's preprocessing, which the settings contract omits."""
        return self.default_config.preprocessing.table_photometric_steps

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        """What /health reports as the engine.

        The flow answers this, because under the two region-based flows there is
        no single engine -- "layout+qwen" is the honest name and `ocr.engine` is
        not even set.
        """
        try:
            return self.get(self.default_config).engine_name
        except Exception:  # noqa: BLE001 - report the intent when it cannot be built
            return self.default_config.ocr.engine or self.default_config.flow.name

    def is_ready(self, config: AppConfig | None = None) -> bool:
        """Whether what `config` selects can serve a request right now.

        Defaults to the service's own configuration, which is what /health
        reports. An extract request passes the configuration it is actually
        going to run.
        """
        target = config or self.default_config

        try:
            return self.get(target).is_ready()
        except Exception:  # noqa: BLE001 - a missing dependency is "not ready"
            logger.warning(
                "Flow '%s' could not be built for readiness check.", target.flow.name, exc_info=True
            )
            return False

    def warmup(self) -> None:
        """Build and warm the default flow, so readiness is not a guess."""
        self.get(self.default_config).warmup()

    def close(self) -> None:
        """Stop every loaded model; called on service shutdown."""
        with self._lock:
            components = list(self._components.values())
            self._components.clear()
            self._orchestrators.clear()
        for item in components:
            close_components(item)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, config: AppConfig) -> PipelineOrchestrator:
        """The orchestrator for `config`, built once and reused afterwards."""
        cache_key = _key(config)

        with self._lock:
            existing = self._orchestrators.get(cache_key)
            if existing is not None:
                self._orchestrators.move_to_end(cache_key)
                return existing[1]

        # Built outside the lock: constructing an engine can take a minute, and
        # holding the lock through it would serialize every other request
        # behind a cache miss. The worst case is two threads building the same
        # orchestrator concurrently, which wastes work but stays correct.
        components_key = _components_key(config)
        components = self._components_for(config, components_key)
        orchestrator = PipelineOrchestrator(config, ocr_components=components)

        with self._lock:
            self._orchestrators[cache_key] = (components_key, orchestrator)
            self._orchestrators.move_to_end(cache_key)
            while len(self._orchestrators) > self.cache_size:
                evicted, _ = self._orchestrators.popitem(last=False)
                logger.debug("Evicted a cached pipeline configuration: %s", evicted[:120])

        return orchestrator

    def _components_for(self, config: AppConfig, components_key: str) -> OCRComponents:
        """The built models for this configuration, shared across orchestrators."""
        with self._lock:
            components = self._components.get(components_key)
            if components is not None:
                self._components.move_to_end(components_key)
                return components

        logger.info("Building OCR components for flow '%s'.", config.flow.name)
        components = build_ocr_components(config)

        evicted: list[OCRComponents] = []
        with self._lock:
            existing = self._components.get(components_key)
            if existing is not None:
                # Another thread won the race; keep one instance per OCR config.
                evicted.append(components)
                components = existing
            else:
                self._components[components_key] = components
                while len(self._components) > self.max_loaded_models:
                    old_key, old = self._components.popitem(last=False)
                    evicted.append(old)
                    # Orchestrators built on the evicted models must not be
                    # handed out again: their workers are about to be stopped.
                    for key in [k for k, (ck, _) in self._orchestrators.items() if ck == old_key]:
                        del self._orchestrators[key]

        for item in evicted:
            logger.info("Unloading an OCR model configuration to stay within max_loaded_models.")
            close_components(item)
        return components
