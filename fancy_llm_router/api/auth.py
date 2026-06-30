"""Shared API key authentication for public router deployments."""

import logging
import secrets
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Unauthenticated endpoints (health checks, API docs).
_PUBLIC_EXACT_PATHS = frozenset({
    "/",
    "/health",
    "/favicon.ico",
    "/docs",
    "/openapi.json",
    "/redoc",
})

# Routes that require a key when ``api_auth_token`` is configured.
_PROTECTED_PREFIXES = ("/api/v1",)


def extract_api_key(request: Request) -> Optional[str]:
    """Read a shared key from Authorization Bearer or X-API-Key."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token

    header_key = request.headers.get("X-API-Key", "").strip()
    if header_key:
        return header_key

    return None


def path_requires_auth(path: str) -> bool:
    """Return True when the path should be checked for an API key."""
    if path in _PUBLIC_EXACT_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES)


def verify_api_key(provided: Optional[str], expected: str) -> bool:
    """Constant-time comparison of client and server keys."""
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided, expected)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject protected requests when the shared key is missing or wrong."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if not path_requires_auth(request.url.path):
            return await call_next(request)

        config = getattr(request.app.state, "config", None)
        expected = getattr(config, "api_auth_token", None) if config else None
        if not expected:
            return await call_next(request)

        provided = extract_api_key(request)
        if not verify_api_key(provided, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
