"""
Bearer-token authentication middleware for the MCP HTTP server.

The expected header format is:
    Authorization: Bearer <base64-encoded-api-key>

The raw API key is read from the ``MCP_API_KEY`` environment variable.
If that variable is unset, auth is disabled (development convenience).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger("mcp_server.auth")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MCP_API_KEY = os.getenv("MCP_API_KEY", "").strip()
AUTH_ENABLED = bool(MCP_API_KEY)

if AUTH_ENABLED:
    # Pre-compute the *expected* Bearer token value so we can do a constant-time
    # comparison later without re-encoding on every request.
    EXPECTED_BEARER_TOKEN = base64.b64encode(MCP_API_KEY.encode("utf-8")).decode("ascii")
    logger.info("Bearer-token auth enabled (MCP_API_KEY is set).")
else:
    EXPECTED_BEARER_TOKEN = ""
    logger.warning(
        "Bearer-token auth is DISABLED — set MCP_API_KEY to secure the server."
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class BearerTokenAuthMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that enforces Bearer-token authentication.

    The client must send::

        Authorization: Bearer <base64(MCP_API_KEY)>

    Any request missing the header, using the wrong scheme, or presenting an
    invalid token receives a 401 response.
    """

    def __init__(self, app: ASGIApp, exempt_paths: list[str] | None = None) -> None:
        super().__init__(app)
        # Paths that skip auth (e.g. health probes).
        self._exempt_paths = set(exempt_paths or ["/health"])

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        # If auth is not configured, let everything through.
        if not AUTH_ENABLED:
            return await call_next(request)

        # Skip auth for exempt paths.
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        # Extract and validate the Authorization header.
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            logger.warning("Auth failure: missing Authorization header (%s)", request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Authorization header."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        parts = auth_header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning("Auth failure: invalid scheme (%s)", request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authentication scheme. Expected Bearer token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        supplied_token = parts[1]

        # Constant-time comparison to mitigate timing attacks.
        if not _secure_compare(supplied_token, EXPECTED_BEARER_TOKEN):
            logger.warning("Auth failure: invalid token (%s)", request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


def _secure_compare(a: str, b: str) -> bool:
    """
    Constant-time string comparison to prevent timing side-channels.

    Falls back to Python's ``hmac.compare_digest`` when available.
    """
    try:
        import hmac

        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        # Fallback (not constant-time, but good enough for basic use).
        return a == b
