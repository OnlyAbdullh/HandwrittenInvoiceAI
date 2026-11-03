"""Who may call the service.

The service binds to loopback, which keeps the LAN out but not the browser: a
web page the user happens to open can POST a multipart form to
``http://127.0.0.1:8765`` without any CORS preflight. Two checks close that:

1. **Origin.** Browsers attach an ``Origin`` header to cross-site POSTs; the
   desktop app's ``HttpClient`` does not send one. A request with a foreign
   origin is refused. The service's own origin (its Swagger page) is allowed.
2. **Token** (optional, ``INVOICE_AI_API_TOKEN``). When set, every endpoint
   except /health requires ``X-Invoice-AI-Token`` to match, compared in
   constant time.

Failures are answered in the contract's error envelope, like every other error.
"""

from __future__ import annotations

import hmac

from fastapi import Request
from fastapi.responses import JSONResponse

from api.schemas import ErrorBody, ErrorCodes, ErrorEnvelope
from api.service_settings import get_settings

TOKEN_HEADER = "X-Invoice-AI-Token"


class RequestRejected(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _own_origins() -> set[str]:
    settings = get_settings()
    hosts = {settings.host, "127.0.0.1", "localhost"}
    return {f"http://{host}:{settings.port}" for host in hosts}


def check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is not None and origin.rstrip("/") not in _own_origins():
        raise RequestRejected(
            403,
            ErrorCodes.FORBIDDEN_ORIGIN,
            f"Requests from origin '{origin}' are not accepted by this local service.",
        )


def check_token(request: Request) -> None:
    expected = get_settings().api_token
    if not expected:
        return
    supplied = request.headers.get(TOKEN_HEADER, "")
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise RequestRejected(
            401, ErrorCodes.UNAUTHORIZED, f"Missing or invalid {TOKEN_HEADER} header."
        )


def trusted_client(request: Request) -> None:
    """Router dependency for every endpoint that does work or reveals config."""
    check_origin(request)
    check_token(request)


async def rejected_handler(request: Request, exc: RequestRejected) -> JSONResponse:
    envelope = ErrorEnvelope(error=ErrorBody(code=exc.code, message=exc.message))
    return JSONResponse(status_code=exc.status, content=envelope.model_dump())
