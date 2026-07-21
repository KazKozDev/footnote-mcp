import os

from starlette.testclient import TestClient

os.environ.setdefault("FOOTNOTE_MCP_API_KEY", "test-key")
os.environ.setdefault("FOOTNOTE_MCP_PUBLIC_URL", "https://footnote.test")

from footnote_mcp.http_server import APIKey, HostedSettings, create_app


def _app():
    return create_app(
        HostedSettings(
            api_keys={"alice": APIKey("test-key", 2), "bob": APIKey("bob-key", 1)},
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


def test_each_key_has_its_own_rate_limit():
    with TestClient(_app(), base_url="https://footnote.test") as client:
        for _ in range(2):
            assert client.get("/mcp", headers={"Authorization": "Bearer test-key"}).status_code == 406
        limited = client.get("/mcp", headers={"Authorization": "Bearer test-key"})
        bob = client.get("/mcp", headers={"Authorization": "Bearer bob-key"})
    assert limited.status_code == 429
    assert bob.status_code == 406


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
