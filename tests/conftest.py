"""Test-run configuration.

Keeps the suite from writing into the project. The service persists a
`result.json` and a display image per run by default -- that is the audit trail
and it belongs in normal operation -- but a test run producing a hundred of them
under `results/` is just litter, and the assertions never look at them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# What the service under test runs. The OCR engine is the machine's choice, not
# the request's (config/settings_contract.py: enforce_server_owned), so the API
# tests choose the deterministic stub the way a deployment would: by the
# service's own configuration file.
STUB_SERVICE_CONFIG = Path(__file__).parent / "fixtures" / "service_config_stub.yaml"


@pytest.fixture(scope="session", autouse=True)
def _isolate_service_output(tmp_path_factory):
    """Point persistence at a temp directory and switch it off for the session.

    Set before anything imports the settings singleton, so the environment is
    what `ServiceSettings()` reads when the app's lifespan first builds it.
    """
    output_dir = tmp_path_factory.mktemp("results")

    previous = {
        key: os.environ.get(key)
        for key in (
            "INVOICE_AI_PERSIST_RESULTS",
            "INVOICE_AI_WRITE_STEP_DEBUG_IMAGES",
            "INVOICE_AI_CONFIG_PATH",
            "INVOICE_AI_API_TOKEN",
        )
    }

    os.environ["INVOICE_AI_PERSIST_RESULTS"] = "false"
    os.environ["INVOICE_AI_WRITE_STEP_DEBUG_IMAGES"] = "false"
    os.environ["INVOICE_AI_CONFIG_PATH"] = str(STUB_SERVICE_CONFIG)
    os.environ.pop("INVOICE_AI_API_TOKEN", None)

    yield output_dir

    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
