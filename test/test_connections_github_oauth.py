"""Codex GitHub consent uses no Kiro process or operator credentials in tests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request
from dashboard_owner_helpers import as_owner

from kiro_crew.connections import github_oauth as oauth
from kiro_crew.connections.oauth_clients import ResolvedOAuthClient
from kiro_crew.secrets import SecretVault


@pytest.fixture
def client_config(monkeypatch, tmp_path):
    client = ResolvedOAuthClient(
        slug="github",
        client_id="test-client",
        client_id_source="config",
        client_secret="test-secret",
        client_secret_source="vault",
        redirect_uri="http://127.0.0.1:48101/callback",
    )
    monkeypatch.setattr(oauth, "resolve_client", lambda home: client)
    monkeypatch.setattr(oauth, "resolve_scopes", lambda: ["public_repo"])
    return client


@pytest.fixture
def listener(monkeypatch):
    """Capture the actual callback handler without occupying the operator's port."""
    listener = SimpleNamespace(app=None, cleaned=False, bound=None)

    class Runner:
        def __init__(self, app, **kwargs):
            assert kwargs["access_log"] is None
            listener.app = app

        async def setup(self):
            pass

        async def cleanup(self):
            listener.cleaned = True

    class Site:
        def __init__(self, runner, host, port):
            listener.bound = (host, port)

        async def start(self):
            pass

    monkeypatch.setattr(oauth.web, "AppRunner", Runner)
    monkeypatch.setattr(oauth.web, "TCPSite", Site)
    return listener


async def waiting(flow):
    async def poll():
        while flow.row["state"] == "minting":
            await asyncio.sleep(0)
        assert flow.row["state"] == "waiting", flow.row

    await asyncio.wait_for(poll(), 5)
    return parse_qs(urlsplit(flow.row["oauth_url"]).query)


async def callback(listener, query, host="127.0.0.1:48101"):
    request = make_mocked_request("GET", "/callback?" + urlencode(query), headers={"Host": host})
    route = next(route for route in listener.app.router.routes() if route.method == "GET")
    return await route.handler(request)


@pytest.mark.asyncio
async def test_consent_pkce_callback_and_encrypted_storage(
    monkeypatch, tmp_path, client_config, listener
):
    exchange = AsyncMock(return_value={"access_token": "test-bearer", "token_type": "bearer"})
    monkeypatch.setattr(oauth, "token_request", exchange)
    flow = oauth.GitHubOAuthFlow(tmp_path)
    try:
        started = await flow.start()
        params = await waiting(flow)
        assert listener.bound == ("127.0.0.1", 48101)
        assert params["client_id"] == [client_config.client_id]
        assert params["redirect_uri"] == [client_config.redirect_uri]
        assert params["code_challenge_method"] == ["S256"]
        assert params["scope"] == ["public_repo"]
        assert "test-secret" not in json.dumps(flow.row)
        reply = await callback(listener, {"state": params["state"][0], "code": "fake-code"})
        assert reply.status == 200
        await asyncio.wait_for(flow.task, 5)
        assert flow.row == {"slug": "github", "state": "granted", "token": started["token"]}
        posted = exchange.call_args.args[0]
        assert posted["client_secret"] == "test-secret"
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(posted["code_verifier"].encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert params["code_challenge"] == [expected]
        assert await asyncio.to_thread(oauth.access_token, tmp_path) == "test-bearer"
        assert "test-bearer" not in (tmp_path / ".vault" / "secrets.enc").read_text()
        assert listener.cleaned
    finally:
        await flow.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["wrong_state", "duplicate_state", "wrong_host", "missing_code"])
async def test_invalid_callback_does_not_exchange(
    monkeypatch, tmp_path, client_config, listener, kind
):
    exchange = AsyncMock()
    monkeypatch.setattr(oauth, "token_request", exchange)
    flow = oauth.GitHubOAuthFlow(tmp_path)
    try:
        await flow.start()
        params = await waiting(flow)
        query = {"state": params["state"][0], "code": "fake-code"}
        host = "127.0.0.1:48101"
        if kind == "wrong_state":
            query["state"] = "wrong"
        elif kind == "duplicate_state":
            query = [("state", params["state"][0]), ("state", "other"), ("code", "fake-code")]
        elif kind == "wrong_host":
            host = "attacker.example:48101"
        else:
            del query["code"]
        assert (await callback(listener, query, host)).status == 400
        exchange.assert_not_called()
        assert flow.row["state"] == "waiting"
    finally:
        await flow.cancel()
    assert listener.cleaned


@pytest.mark.asyncio
async def test_denial_and_cancel_do_not_remove_prior_grant(
    monkeypatch, tmp_path, client_config, listener
):
    oauth._save(tmp_path, client_config, {"access_token": "prior-bearer"})
    flow = oauth.GitHubOAuthFlow(tmp_path)
    try:
        await flow.start()
        params = await waiting(flow)
        await callback(listener, {"state": params["state"][0], "error": "access_denied"})
        await asyncio.wait_for(flow.task, 5)
        assert flow.row["reason"] == "github_consent_denied"
        await flow.start()
        await waiting(flow)
        assert not await flow.cancel("another-tabs-token")
        assert await flow.cancel(flow.row["token"])
        assert oauth.grant_presence(tmp_path) is True
    finally:
        await flow.cancel()


@pytest.mark.asyncio
async def test_flow_expires_and_closes_listener(monkeypatch, tmp_path, client_config, listener):
    monkeypatch.setattr(oauth, "FLOW_TTL", 0)
    flow = oauth.GitHubOAuthFlow(tmp_path)
    await flow.start()
    try:
        await asyncio.wait_for(flow.task, 5)
        assert flow.row["state"] == "expired"
        assert "oauth_url" not in flow.row
        assert listener.cleaned
    finally:
        await flow.cancel()


def test_grant_missing_unreadable_changed_client_and_disconnect(
    monkeypatch, tmp_path, client_config
):
    assert oauth.grant_presence(tmp_path) is False
    oauth._save(tmp_path, client_config, {"access_token": "grant"})
    assert oauth.grant_presence(tmp_path) is True
    assert oauth.forget_grant(tmp_path)
    assert not oauth.forget_grant(tmp_path)
    SecretVault(tmp_path).set_sync(oauth.GRANT_NAME, "[]")
    assert oauth.grant_presence(tmp_path) is None
    SecretVault(tmp_path).set_sync(
        oauth.GRANT_NAME, json.dumps({"client_id": "different", "access_token": "grant"})
    )
    assert oauth.grant_presence(tmp_path) is False
    assert oauth.access_token(tmp_path) is None


def test_expired_grant_refresh_is_persisted(monkeypatch, tmp_path, client_config):
    oauth._save(
        tmp_path,
        client_config,
        {"access_token": "old", "refresh_token": "refresh", "expires_in": 0},
    )
    exchange = AsyncMock(
        return_value={"access_token": "new", "refresh_token": "rotated", "expires_in": 3600}
    )
    monkeypatch.setattr(oauth, "token_request", exchange)
    assert oauth.access_token(tmp_path) == "new"
    assert oauth.access_token(tmp_path) == "new"
    assert exchange.await_count == 1
    assert exchange.call_args.args[0]["grant_type"] == "refresh_token"
    assert oauth._load(tmp_path)["refresh_token"] == "rotated"


def test_projection_exact_destination_and_no_mutation(monkeypatch):
    monkeypatch.setattr(oauth, "access_token", lambda home: "native-bearer")
    entry = {"name": "github", "type": "http", "url": oauth.MCP_URL, "headers": []}
    variants = [
        entry,
        {**entry, "name": "another"},
        {**entry, "url": oauth.MCP_URL + "?redirect=evil"},
        {**entry, "url": "https://evil.example/mcp/"},
        {**entry, "type": "sse"},
        {**entry, "headers": [{"name": "authorization", "value": "explicit"}]},
    ]
    original = deepcopy(variants)
    projected = oauth.project_headers(variants)
    assert projected[0]["headers"] == [{"name": "Authorization", "value": "Bearer native-bearer"}]
    assert projected[1:] == original[1:]
    assert variants == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"access_token": "valid", "token_type": "bearer"}, None),
        (
            {"error": "incorrect_client_credentials", "error_description": "secret-from-upstream"},
            "github_incorrect_client_credentials",
        ),
        ({"error": "secret-from-upstream"}, "github_token_exchange_failed"),
        (
            {"access_token": "bad\r\nheader", "token_type": "bearer"},
            "github_invalid_token_response",
        ),
        ({"access_token": "valid", "token_type": "other"}, "github_invalid_token_response"),
    ],
)
async def test_token_http_exchange_and_safe_errors(monkeypatch, payload, expected):
    seen = []

    async def endpoint(request):
        seen.append(dict(await request.post()))
        return web.json_response(payload)

    app = web.Application()
    app.router.add_post("/token", endpoint)
    async with TestServer(app) as server:
        monkeypatch.setattr(oauth, "TOKEN_URL", str(server.make_url("/token")))
        if expected:
            with pytest.raises(oauth.OAuthFailure, match=f"^{expected}$"):
                await oauth.token_request({"client_secret": "test-only"})
        else:
            assert await oauth.token_request({"client_secret": "test-only"}) == payload
    assert seen == [{"client_secret": "test-only"}]


@pytest.mark.asyncio
async def test_routes_dispatch_codex_without_kiro(monkeypatch, tmp_path):
    from kiro_crew.dashboard.handlers import connections

    flow = oauth.GitHubOAuthFlow(tmp_path)
    flow.start = AsyncMock(
        return_value={"ok": True, "slug": "github", "state": "minting", "token": "row-token"}
    )
    monkeypatch.setattr(oauth, "enabled", lambda: True)
    monkeypatch.setattr(connections, "_oauth_client_configured", lambda provider: True)
    monkeypatch.setattr(
        oauth, "test_tools", AsyncMock(return_value={"verdict": "usable", "toolCount": 2})
    )
    app = as_owner(web.Application())
    app[oauth.APP_STATE_KEY] = {"github": flow}
    app.router.add_post("/mint", connections.api_connections_mint)
    app.router.add_get("/mint", connections.api_connections_mint_state)
    app.router.add_post("/premint", connections.api_connections_premint)
    app.router.add_post("/test", connections.api_connections_test)
    async with TestClient(TestServer(app)) as client:
        assert (await (await client.post("/mint", json={"slug": "github"})).json())[
            "token"
        ] == "row-token"
        assert (await (await client.get("/mint?slug=github")).json())["state"] == "idle"
        assert (await (await client.post("/premint")).json())["preminting"] == []
        assert (await (await client.post("/test", json={"slug": "github"})).json())[
            "toolCount"
        ] == 2
        denied = await client.post(
            "/mint", json={"slug": "github"}, headers={"X-Test-User": "stranger"}
        )
        assert denied.status == 403
    flow.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_other_backends_and_providers_do_not_select_native(monkeypatch):
    monkeypatch.setattr(oauth, "enabled", lambda: False)
    request = make_mocked_request("GET", "/")
    assert await oauth.for_request(request) is None
    monkeypatch.setattr(oauth, "enabled", lambda: True)
    assert await oauth.for_request(request, "notion") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("sse", [False, True])
async def test_test_action_uses_projected_headers_and_never_calls_tools(monkeypatch, tmp_path, sse):
    from kiro_crew.providers.mirrors import codex

    calls = []

    async def endpoint(request):
        assert request.headers["Authorization"] == "Bearer explicit-operator-token"
        assert request.headers["X-MCP-Readonly"] == "true"
        body = await request.json()
        calls.append(body["method"])
        if body["method"] == "notifications/initialized":
            return web.Response(status=202)
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26"}
        else:
            assert body["method"] == "tools/list"
            result = {
                "tools": [{"name": "allowed", "description": "x" * 200000}, {"name": "blocked"}]
            }
        reply = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        if sse:
            return web.Response(
                text="event: message\ndata: " + json.dumps(reply) + "\n\n",
                content_type="text/event-stream",
            )
        return web.json_response(reply)

    app = web.Application()
    app.router.add_post("/mcp", endpoint)
    async with TestServer(app) as server:
        url = str(server.make_url("/mcp"))
        monkeypatch.setattr(oauth, "MCP_URL", url)
        projection = SimpleNamespace(
            params={
                "mcpServers": [
                    {
                        "name": "github",
                        "type": "http",
                        "url": url,
                        "headers": [
                            {"name": "Authorization", "value": "Bearer explicit-operator-token"},
                            {"name": "X-MCP-Readonly", "value": "true"},
                        ],
                    }
                ]
            },
            denied_tools=frozenset({("github", "blocked")}),
        )
        monkeypatch.setattr(codex, "codex_projection", lambda agent: projection)
        result = await oauth.test_tools(tmp_path)
    assert result["verdict"] == "usable", result
    assert result["toolCount"] == 1
    assert calls == ["initialize", "notifications/initialized", "tools/list"]


@pytest.mark.asyncio
@pytest.mark.parametrize("shared,unreadable", [(False, False), (True, False), (False, True)])
async def test_native_disconnect_obeys_census_and_never_deletes_kiro(
    monkeypatch, shared, unreadable
):
    from test_connections_disconnect import _wire

    from kiro_crew.connections import ownership

    purged, revoked, removed = [], [], []
    specs = {"kirocrew": {"github": {"url": oauth.MCP_URL}}}
    if shared:
        specs["agent:custom"] = {"github": {"url": oauth.MCP_URL}}
    _wire(
        monkeypatch,
        removed=[],
        surviving=[],
        inventory=[],
        purged=purged,
        revoked=revoked,
        raw_specs=specs,
        unreadable=("agent:unreadable",) if unreadable else (),
    )

    def remove_native():
        removed.append(True)
        return True

    scope = await ownership.remove_provider_entry(
        "github", oauth.MCP_URL, (), revoke_runtime_grant=False, native_grant_remover=remove_native
    )
    assert not revoked
    assert bool(removed) is (not shared and not unreadable)
    assert bool(scope.grant_removed) is (not shared and not unreadable)
    assert bool(scope.grant_shared_with) is shared
    assert scope.census_incomplete is unreadable


@pytest.mark.parametrize(
    "hints,expected",
    [
        ({}, ["read:user", "read:org", "public_repo"]),
        ({"scopes": []}, []),
        ({"scopes": ["public_repo"]}, ["public_repo"]),
        ({"oauthScopes": ["read:user"]}, ["read:user"]),
        ({"oauth": {"oauthScopes": ["read:org"]}}, ["read:org"]),
    ],
)
@pytest.mark.parametrize("store_owned", [False, True])
def test_scopes_follow_configured_entry_including_empty(
    monkeypatch, tmp_path, hints, expected, store_owned
):
    from kiro_crew import agent
    from kiro_crew.agent_files import AGENT_FILENAME
    from kiro_crew.dashboard.handlers import mcp
    from kiro_crew.mcp_utils import kiro_oauth_wire_entry

    monkeypatch.setattr(agent, "kiro_agents_dir_path", lambda: tmp_path)
    monkeypatch.setattr(mcp, "_KIROCREW_MCP_JSON", tmp_path / "mcp.json")
    entry = {"url": oauth.MCP_URL, **hints}
    if store_owned:
        (tmp_path / "mcp.json").write_text(
            json.dumps({"mcpServers": {"github": entry}}), encoding="utf-8"
        )
        entry = kiro_oauth_wire_entry(entry, store_entry=entry, server="github")
    (tmp_path / AGENT_FILENAME).write_text(
        json.dumps(
            {
                "name": "kirocrew",
                "mcpServers": {"github": entry},
            }
        ),
        encoding="utf-8",
    )
    assert oauth.resolve_scopes() == expected


def test_unreadable_scope_source_does_not_request_default_scopes(monkeypatch, tmp_path):
    from kiro_crew.dashboard.handlers import mcp

    path = tmp_path / "mcp.json"
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(mcp, "_KIROCREW_MCP_JSON", path)
    with pytest.raises(oauth.OAuthFailure, match="github_mcp_config_unreadable"):
        oauth.resolve_scopes()
