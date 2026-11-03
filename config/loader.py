from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from config.schema import AppConfig

# Relative paths in a config are relative to the project, never to the CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ${NAME} or ${NAME:-default}. The default may itself contain anything but "}".
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Substitute ``${NAME:-default}`` in every string inside a parsed YAML tree.

    Machine-specific values -- where the model weights live, which python.exe
    runs the VLM worker -- belong in the environment, not in a committed file.
    Hard-coding them made every shipped config point at a path that existed on
    exactly one developer's machine.

    An unset variable with no default is an error rather than an empty string:
    an empty model path would fail later and far less clearly.
    """
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name)
        if resolved:
            return resolved
        if default is not None:
            return default
        raise ValueError(f"Environment variable '{name}' is not set and has no default.")

    return _ENV_REFERENCE.sub(substitute, value)


def anchor_paths(config: AppConfig, root: Path = PROJECT_ROOT) -> AppConfig:
    """Make the config's relative file paths absolute, against the project root.

    The YAML files say `keywords/ar_invoice_terms.json` and `results/`. Those
    were resolved against the current directory, so the keyword dictionary
    silently vanished whenever the service or a CLI tool was started from
    anywhere else. (OCR worker and model paths are anchored the same way by the
    engine that owns them.)
    """
    dictionary = Path(config.string_matching.dictionary_path)
    if not dictionary.is_absolute():
        config.string_matching.dictionary_path = str(root / dictionary)

    output_dir = config.persistence.store_params.get("output_dir")
    if output_dir and not Path(output_dir).is_absolute():
        config.persistence.store_params["output_dir"] = str(root / output_dir)

    if config.debug_dir and not Path(config.debug_dir).is_absolute():
        config.debug_dir = str(root / config.debug_dir)

    return config


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an AppConfig from a YAML file.

    Read as UTF-8 explicitly: the files carry Arabic comments, and the locale
    default (cp1252 on most Windows installs) cannot decode them. Validation
    failures raise pydantic.ValidationError with a precise, field-level
    message -- config problems fail fast at startup instead of deep inside the
    pipeline.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return anchor_paths(AppConfig(**expand_env(raw)))
