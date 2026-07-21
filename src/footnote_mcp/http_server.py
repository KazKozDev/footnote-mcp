"""Authenticated Streamable HTTP entry point for hosted MCP deployments.

The existing ``footnote-mcp`` command intentionally remains a stdio server. Run
this module with Uvicorn when a remote MCP endpoint is needed instead.
"""

from __future__ import annotations

import contextlib
import os
import secrets
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
    api_key: str
    public_url: str
    allowed_host: str
    allowed_origin: str


def _hosted_settings() -> HostedSettings:
    api_key = os.environ.get("FOOTNOTE_MCP_API_KEY")
    public_url = os.environ.get("FOOTNOTE_MCP_PUBLIC_URL")
    if not api_key:
        raise RuntimeError("FOOTNOTE_MCP_API_KEY must be set for the HTTP server")
    if not public_url:
        raise RuntimeError("FOOTNOTE_MCP_PUBLIC_URL must be set, for example https://footnote-mcp.onrender.com")

    parsed = urlsplit(public_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RuntimeError("FOOTNOTE_MCP_PUBLIC_URL must be an origin, for example https://footnote-mcp.onrender.com")
    return HostedSettings(
        api_key=api_key,
        public_url=public_url.rstrip("/"),
        allowed_host=parsed.netloc,
        allowed_origin=f"{parsed.scheme}://{parsed.netloc}",
    )


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, api_key: str) -> None:
        super().__init__(app)
        self._expected = f"Bearer {api_key}"

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path == "/healthz":
            return await call_next(request)
        actual = request.headers.get("authorization", "")
        if not secrets.compare_digest(actual, self._expected):
            return PlainTextResponse(
                "Unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
        return await call_next(request)


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
        middleware=[Middleware(APIKeyMiddleware, api_key=settings.api_key)],
        lifespan=lifespan,
    )


app = create_app()
