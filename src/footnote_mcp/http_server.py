"""Authenticated Streamable HTTP entry point for hosted MCP deployments.

The existing ``footnote-mcp`` command intentionally remains a stdio server. Run
this module with Uvicorn when a remote MCP endpoint is needed instead.
"""

from __future__ import annotations

import contextlib
import asyncio
import json
import os
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from urllib.parse import urlsplit

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from .server import server


@dataclass(frozen=True)
class HostedSettings:
    api_keys: dict[str, "APIKey"]
    public_url: str
    allowed_host: str
    allowed_origin: str


@dataclass(frozen=True)
class APIKey:
    secret: str
    requests_per_minute: int


def _hosted_settings() -> HostedSettings:
    owner_key = os.environ.get("FOOTNOTE_MCP_API_KEY")
    public_url = os.environ.get("FOOTNOTE_MCP_PUBLIC_URL")
    api_keys: dict[str, APIKey] = {}
    if owner_key:
        api_keys["owner"] = APIKey(owner_key, _positive_int(os.environ.get("FOOTNOTE_MCP_DEFAULT_RPM", "30")))
    raw_user_keys = os.environ.get("FOOTNOTE_MCP_API_KEYS", "{}")
    try:
        user_keys = json.loads(raw_user_keys)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FOOTNOTE_MCP_API_KEYS must be valid JSON") from exc
    if not isinstance(user_keys, dict):
        raise RuntimeError("FOOTNOTE_MCP_API_KEYS must be a JSON object keyed by user ID")
    for user_id, config in user_keys.items():
        if not isinstance(user_id, str) or not user_id or user_id == "owner" or not isinstance(config, dict):
            raise RuntimeError("FOOTNOTE_MCP_API_KEYS entries must be non-owner user IDs with object values")
        secret = config.get("key")
        if not isinstance(secret, str) or not secret:
            raise RuntimeError(f"FOOTNOTE_MCP_API_KEYS.{user_id}.key must be a non-empty string")
        api_keys[user_id] = APIKey(secret, _positive_int(config.get("rpm", os.environ.get("FOOTNOTE_MCP_DEFAULT_RPM", "30"))))
    if not api_keys:
        raise RuntimeError("Set FOOTNOTE_MCP_API_KEY or FOOTNOTE_MCP_API_KEYS for the HTTP server")
    if not public_url:
        raise RuntimeError("FOOTNOTE_MCP_PUBLIC_URL must be set, for example https://footnote-mcp.onrender.com")

    parsed = urlsplit(public_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RuntimeError("FOOTNOTE_MCP_PUBLIC_URL must be an origin, for example https://footnote-mcp.onrender.com")
    return HostedSettings(
        api_keys=api_keys,
        public_url=public_url.rstrip("/"),
        allowed_host=parsed.netloc,
        allowed_origin=f"{parsed.scheme}://{parsed.netloc}",
    )


def _positive_int(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("MCP rate limits must be positive integers") from exc
    if result < 1:
        raise RuntimeError("MCP rate limits must be positive integers")
    return result


class PerKeyRateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key_id: str, requests_per_minute: int) -> tuple[bool, int]:
        now = time.monotonic()
        threshold = now - 60
        async with self._lock:
            requests = self._requests[key_id]
            while requests and requests[0] <= threshold:
                requests.popleft()
            if len(requests) >= requests_per_minute:
                return False, 0
            requests.append(now)
            return True, requests_per_minute - len(requests)


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, api_keys: dict[str, APIKey]) -> None:
        super().__init__(app)
        self._api_keys = api_keys
        self._rate_limiter = PerKeyRateLimiter()

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path == "/healthz":
            return await call_next(request)
        actual = request.headers.get("authorization", "")
        key_id = next(
            (candidate_id for candidate_id, key in self._api_keys.items() if secrets.compare_digest(actual, f"Bearer {key.secret}")),
            None,
        )
        if key_id is None:
            return PlainTextResponse(
                "Unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
        allowed, remaining = await self._rate_limiter.allow(key_id, self._api_keys[key_id].requests_per_minute)
        if not allowed:
            return PlainTextResponse("Rate limit exceeded", status_code=429, headers={"Retry-After": "60"})
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._api_keys[key_id].requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


async def healthz(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def create_app(settings: HostedSettings | None = None) -> Starlette:
    """Build the authenticated ``/mcp`` ASGI application."""
    settings = settings or _hosted_settings()
    manager = StreamableHTTPSessionManager(
        app=server,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[settings.allowed_host],
            allowed_origins=[settings.allowed_origin],
        ),
    )

    class MCPApp:
        async def __call__(self, scope, receive, send) -> None:
            await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
        async with manager.run():
            yield

    return Starlette(
        routes=[
            Route("/healthz", endpoint=healthz, methods=["GET"]),
            Route("/mcp", endpoint=MCPApp()),
        ],
        middleware=[Middleware(APIKeyMiddleware, api_keys=settings.api_keys)],
        lifespan=lifespan,
    )


app = create_app()
