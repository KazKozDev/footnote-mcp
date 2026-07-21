import os

from starlette.testclient import TestClient

os.environ.setdefault("FOOTNOTE_MCP_API_KEY", "test-key")
os.environ.setdefault("FOOTNOTE_MCP_PUBLIC_URL", "https://footnote.test")

from footnote_mcp.http_server import HostedSettings, create_app


def _app():
    return create_app(
        HostedSettings(
            api_key="test-key",
            public_url="https://footnote.test",
            allowed_host="footnote.test",
            allowed_origin="https://footnote.test",
        )
    )


def test_health_endpoint_is_public():
    with TestClient(_app(), base_url="https://footnote.test") as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_mcp_endpoint_requires_bearer_key():
    with TestClient(_app(), base_url="https://footnote.test") as client:
        response = client.post("/mcp", json={})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_mcp_endpoint_completes_initialize_handshake():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    with TestClient(_app(), base_url="https://footnote.test") as client:
        response = client.post(
            "/mcp",
            json=payload,
            headers={
                "Authorization": "Bearer test-key",
                "Accept": "application/json, text/event-stream",
            },
        )
    assert response.status_code == 200
    assert response.headers["mcp-session-id"]
    assert '"serverInfo":{"name":"footnote"' in response.text
