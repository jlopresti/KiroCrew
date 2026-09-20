"""GitHub Connections OAuth for Codex, independent of the Kiro runtime.

Only the gateway reads the encrypted grant. The Codex mirror supplies the bearer
in memory, after the ordinary spec/allowlist projection, never in an agent file.
The browser flow is owned by one dashboard application and is closed on shutdown.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import aiohttp
from aiohttp import web

from kiro_crew.acp_backends import ACP_BACKEND_CODEX
from kiro_crew.connections.oauth_clients import ResolvedOAuthClient, resolve_oauth_client
from kiro_crew.connections.registry import get_provider
from kiro_crew.secrets import SecretVault

logger = logging.getLogger(__name__)

SLUG = "github"
MCP_URL = "https://api.githubcopilot.com/mcp/"
AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
GRANT_NAME = "CONNECTIONS_GITHUB_CODEX_OAUTH_GRANT"
APP_STATE_KEY = "connections_native_oauth"
FLOW_TTL = 600
HTTP_TIMEOUT = 30
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
REFRESH_MARGIN = 60
_STORE_LOCK = threading.RLock()


class OAuthFailure(Exception):
    """A stable, credential-free error code, never an upstream response body."""


def enabled() -> bool:
    from kiro_crew.config.loader import read_config_for_update

    config = read_config_for_update()
    agent = config.get("agent")
    return isinstance(agent, dict) and agent.get("acp_backend") == ACP_BACKEND_CODEX


def resolve_client(home: Path) -> ResolvedOAuthClient | None:
    from kiro_crew.config.loader import read_config_for_update

    provider = get_provider(SLUG)
    assert provider is not None
    return resolve_oauth_client(provider, config=read_config_for_update(), vault=SecretVault(home))


def resolve_scopes() -> list[str]:
    """Use the configured server's scopes, including an explicit empty list."""
    from kiro_crew import agent
    from kiro_crew.agent_discovery import _read_agent_spec
    from kiro_crew.agent_files import AGENT_FILENAME
    from kiro_crew.dashboard.handlers.mcp import _kirocrew_mcp_json
    from kiro_crew.mcp_utils import (
        INTERNAL_SCOPES_KEY,
        KIRO_OAUTH_KEY,
        KIRO_SCOPES_KEY,
        kiro_entry_scopes,
    )

    # The renderer deliberately erases empty scope hints. Consult its source
    # first so an explicit [] never becomes a request for the default scopes.
    try:
        source = json.loads(_kirocrew_mcp_json().read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError("not an object")
    except FileNotFoundError:
        source = {}
    except (OSError, ValueError):
        raise OAuthFailure("github_mcp_config_unreadable") from None
    servers = source.get("mcpServers")
    entry = servers.get(SLUG) if isinstance(servers, dict) else None
    if not isinstance(entry, dict):
        path = agent.kiro_agents_dir_path() / AGENT_FILENAME
        spec = _read_agent_spec(path, operation="connections_mint", source="dashboard")
        if spec is None and path.exists():
            raise OAuthFailure("github_agent_spec_unreadable")
        servers = (spec or {}).get("mcpServers")
        entry = servers.get(SLUG) if isinstance(servers, dict) else None
    if isinstance(entry, dict) and entry.get("url") == MCP_URL:
        nested = entry.get(KIRO_OAUTH_KEY)
        if (
            INTERNAL_SCOPES_KEY in entry
            or KIRO_SCOPES_KEY in entry
            or isinstance(nested, dict)
            and KIRO_SCOPES_KEY in nested
        ):
            return kiro_entry_scopes(entry)
    provider = get_provider(SLUG)
    return list(provider.get("recommended_scopes", [])) if provider else []


def _load(home: Path) -> dict[str, Any] | None:
    value = SecretVault(home).get(GRANT_NAME)
    if value is None:
        return None
    grant = json.loads(value.reveal())
    if not isinstance(grant, dict):
        raise OAuthFailure("github_grant_unreadable")
    return grant


def grant_presence(home: Path) -> bool | None:
    """No network; distinguish missing, expired, and unreadable grants."""
    try:
        with _STORE_LOCK:
            grant = _load(home)
            client = resolve_client(home)
            if not grant or not client or grant.get("client_id") != client.client_id:
                return False
            expires = grant.get("expires_at", 0)
            return bool(grant.get("access_token")) and (
                not expires or float(expires) > time.time() or bool(grant.get("refresh_token"))
            )
    except Exception:
        return None


def forget_grant(home: Path) -> bool:
    with _STORE_LOCK:
        vault = SecretVault(home)
        present = GRANT_NAME in vault.list_names()
        vault.delete_sync(GRANT_NAME)
        return present


async def _read_json(response: aiohttp.ClientResponse) -> dict[str, Any]:
    body = b""  # do not trust Content-Length
    async for chunk in response.content.iter_chunked(65536):
        body += chunk
        if len(body) > MAX_RESPONSE_BYTES:
            raise OAuthFailure("github_response_too_large")
    try:
        result = json.loads(body)
    except (ValueError, UnicodeError):
        raise OAuthFailure("github_invalid_response") from None
    if not isinstance(result, dict):
        raise OAuthFailure("github_invalid_response")
    return result


async def token_request(data: dict[str, str]) -> dict[str, Any]:
    """Fixed GitHub destination, verified TLS, no redirects or response logging."""
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            TOKEN_URL, data=data, headers={"Accept": "application/json"}, allow_redirects=False
        ) as response:
            if response.status != 200:
                raise OAuthFailure("github_token_exchange_failed")
            result = await _read_json(response)
    if result.get("error"):
        code = result["error"]
        known = {"incorrect_client_credentials", "redirect_uri_mismatch", "bad_verification_code"}
        raise OAuthFailure(f"github_{code}" if code in known else "github_token_exchange_failed")
    token = result.get("access_token")
    if (
        not isinstance(token, str)
        or not token
        or len(token) > 4096
        or any(ord(c) <= 32 or ord(c) >= 127 for c in token)
        or str(result.get("token_type", "")).lower() != "bearer"
    ):
        raise OAuthFailure("github_invalid_token_response")
    return result


def _save(home: Path, client: ResolvedOAuthClient, result: dict[str, Any]) -> None:
    with _STORE_LOCK:
        current = resolve_client(home)
        if current is None or current != client:
            raise OAuthFailure("github_client_changed")
        expires = result.get("expires_in")
        grant = {
            "client_id": client.client_id,
            "access_token": result["access_token"],
            "refresh_token": result.get("refresh_token"),
            "expires_at": time.time() + max(0, float(expires)) if expires is not None else 0,
        }
        SecretVault(home).set_sync(GRANT_NAME, json.dumps(grant))


def access_token(home: Path) -> str | None:
    """Worker-thread only. Refresh expiring grants before session construction."""
    with _STORE_LOCK:
        grant = _load(home)
        client = resolve_client(home)
        if not grant or not client or grant.get("client_id") != client.client_id:
            return None
        expires = grant.get("expires_at", 0)
        if expires and float(expires) <= time.time() + REFRESH_MARGIN:
            refresh = grant.get("refresh_token")
            if not isinstance(refresh, str) or not refresh:
                return None
            result = asyncio.run(
                token_request(
                    {
                        "grant_type": "refresh_token",
                        "refresh_token": refresh,
                        "client_id": client.client_id,
                        "client_secret": client.client_secret or "",
                    }
                )
            )
            result.setdefault("refresh_token", refresh)
            _save(home, client, result)
            return str(result["access_token"])
        token = grant.get("access_token")
        return token if isinstance(token, str) else None


def project_headers(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Authenticate ONLY the exact GitHub entry already admitted by the mirror.

    Do not override an operator's explicit Authorization header. No token is
    looked up for any other name, URL, transport or withheld server.
    """
    from kiro_crew.config import config_dir

    out = []
    for element in elements:
        if (
            element.get("name") == SLUG
            and element.get("type") == "http"
            and element.get("url") == MCP_URL
            and not any(
                str(h.get("name", "")).lower() == "authorization"
                for h in element.get("headers", [])
            )
        ):
            try:
                token = access_token(config_dir())
            except Exception:
                # Never put an exception containing upstream data in diagnostics.
                logger.warning("GitHub OAuth grant unavailable; reconnect in Connections")
                token = None
            if token:
                element = {
                    **element,
                    "headers": [
                        *element.get("headers", []),
                        {"name": "Authorization", "value": f"Bearer {token}"},
                    ],
                }
        out.append(element)
    return out


async def test_tools(home: Path) -> dict[str, Any]:
    """Authenticated initialize + tools/list, with no LLM or tool invocation."""
    from kiro_crew.providers.mirrors.codex import codex_projection

    result: dict[str, Any] = {
        "schema_version": 1,
        "slug": SLUG,
        "verdict": "failed",
        "code": "github_connection_test_failed",
        "toolCount": 0,
    }
    try:
        projection = await asyncio.to_thread(codex_projection, "kirocrew")
        server = next(
            (
                server
                for server in projection.params.get("mcpServers", [])
                if server.get("name") == SLUG
                and server.get("url") == MCP_URL
                and server.get("type") == "http"
            ),
            None,
        )
        if server is None:
            result.update(verdict="no_tools", code="mcp_server_not_loaded")
            return result
        headers = {
            **{h["name"]: h["value"] for h in server.get("headers", [])},
            "Accept": "application/json, text/event-stream",
        }
        if not any(name.lower() == "authorization" for name in headers):
            result["code"] = "github_not_authorized"
            return result
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        ) as session:

            async def rpc(body: dict[str, Any]) -> dict[str, Any]:
                async with session.post(
                    MCP_URL, json=body, headers=headers, allow_redirects=False
                ) as response:
                    if response.status in (401, 403):
                        raise OAuthFailure("github_not_authorized")
                    if response.status not in (200, 202, 204):
                        raise OAuthFailure("github_mcp_unavailable")
                    if response.headers.get("Mcp-Session-Id"):
                        headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
                    if "id" not in body:
                        return {}
                    reply = await _read_rpc(response, body["id"])
                    if reply.get("error") or not isinstance(reply.get("result"), dict):
                        raise OAuthFailure("github_invalid_mcp_response")
                    return reply["result"]

            init = await rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "kirocrew-connections", "version": "1"},
                    },
                }
            )
            headers["MCP-Protocol-Version"] = str(init.get("protocolVersion", "2025-03-26"))
            await rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
            listing = await rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            tools = listing.get("tools")
            if not isinstance(tools, list):
                raise OAuthFailure("github_invalid_mcp_response")
            count = len(
                {
                    tool["name"]
                    for tool in tools
                    if isinstance(tool, dict)
                    and isinstance(tool.get("name"), str)
                    and (SLUG, tool["name"]) not in projection.denied_tools
                }
            )
            result.update(
                verdict="usable" if count else "no_tools",
                code="tools_available" if count else "no_tools_exposed",
                toolCount=count,
            )
    except OAuthFailure as exc:
        result["code"] = str(exc)
    except Exception:
        pass  # The API deliberately never receives raw exception/response text.
    return result


async def _read_rpc(response: aiohttp.ClientResponse, request_id: int) -> dict[str, Any]:
    """Bound both JSON and SSE, and stop at our response even on an open stream."""
    if response.content_type != "text/event-stream":
        reply = await _read_json(response)
        if reply.get("id") != request_id:
            raise OAuthFailure("github_invalid_mcp_response")
        return reply
    size = 0
    data: list[bytes] = []
    buffered = b""
    async for chunk in response.content.iter_chunked(65536):
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise OAuthFailure("github_response_too_large")
        buffered += chunk
        while b"\n" in buffered:
            line, _, buffered = buffered.partition(b"\n")
            if line.startswith(b"data:"):
                data.append(line[5:].strip())
            elif not line.strip() and data:
                reply = json.loads(b"\n".join(data))
                data.clear()
                if isinstance(reply, dict) and reply.get("id") == request_id:
                    return reply
    raise OAuthFailure("github_invalid_mcp_response")


class GitHubOAuthFlow:
    """One app-owned, cancellable browser consent flow; no process-global grants."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.row: dict[str, Any] = {"slug": SLUG, "state": "idle"}
        self.task: asyncio.Task[None] | None = None
        self.lock = asyncio.Lock()

    async def start(self) -> dict[str, Any]:
        async with self.lock:
            await self._cancel()
            self.row = {"slug": SLUG, "state": "minting", "token": secrets.token_urlsafe(24)}
            self.task = asyncio.create_task(self._run())
            return {"ok": True, **self.row}

    async def _cancel(self) -> bool:
        running = self.task is not None and not self.task.done()
        if running:
            assert self.task is not None
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        return running

    async def cancel(self, token: str | None = None) -> bool:
        async with self.lock:
            if token is not None and token != self.row.get("token"):
                return False
            dropped = await self._cancel()
            self.row = {"slug": SLUG, "state": "idle"}
            return dropped

    async def _run(self) -> None:
        runner: web.AppRunner | None = None
        try:
            client = await asyncio.to_thread(resolve_client, self.home)
            if client is None:
                raise OAuthFailure("client_not_configured")
            scopes = await asyncio.to_thread(resolve_scopes)
            state = secrets.token_urlsafe(32)
            verifier = secrets.token_urlsafe(48)
            challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
                .rstrip(b"=")
                .decode()
            )
            callback = urlsplit(client.redirect_uri)
            consent: asyncio.Future[str] = asyncio.get_running_loop().create_future()

            async def receive(request: web.Request) -> web.Response:
                # No access log: the query string carries the authorization code.
                values = request.query.getall("state", [])
                if (
                    request.host != callback.netloc
                    or len(values) != 1
                    or not hmac.compare_digest(values[0], state)
                ):
                    return web.Response(status=400, text="Invalid OAuth callback.")
                codes = request.query.getall("code", [])
                if consent.done():
                    return web.Response(status=409, text="This authorization was already received.")
                if request.query.get("error"):
                    consent.set_exception(OAuthFailure("github_consent_denied"))
                elif len(codes) == 1 and codes[0] and len(codes[0]) <= 4096:
                    consent.set_result(codes[0])
                else:
                    return web.Response(status=400, text="Missing authorization code.")
                return web.Response(
                    text="Authorization received. Return to Kiro Crew.",
                    headers={
                        "Cache-Control": "no-store",
                        "Referrer-Policy": "no-referrer",
                    },
                )

            app = web.Application()
            app.router.add_get(callback.path, receive)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            try:
                await web.TCPSite(runner, "127.0.0.1", callback.port).start()
            except OSError:
                raise OAuthFailure("github_callback_unavailable") from None
            self.row.update(
                state="waiting",
                oauth_url=AUTHORIZE_URL
                + "?"
                + urlencode(
                    {
                        "client_id": client.client_id,
                        "redirect_uri": client.redirect_uri,
                        "scope": " ".join(scopes),
                        "state": state,
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    }
                ),
            )
            code = await asyncio.wait_for(consent, timeout=FLOW_TTL)
            self.row.pop("oauth_url", None)
            self.row["state"] = "minting"
            response = await token_request(
                {
                    "client_id": client.client_id,
                    "client_secret": client.client_secret or "",
                    "code": code,
                    "redirect_uri": client.redirect_uri,
                    "code_verifier": verifier,
                }
            )
            # A cancelled to_thread write still runs. Drain it before cancel /
            # disconnect can delete the grant, preventing post-disconnect resurrection.
            save = asyncio.create_task(asyncio.to_thread(_save, self.home, client, response))
            try:
                await asyncio.shield(save)
            except asyncio.CancelledError:
                await save
                raise
            self.row.update(state="granted")
        except asyncio.TimeoutError:
            self.row.update(state="expired", reason="github_oauth_timeout")
        except OAuthFailure as exc:
            self.row.update(state="failed", reason=str(exc))
        except Exception:
            self.row.update(state="failed", reason="github_oauth_failed")
        finally:
            self.row.pop("oauth_url", None)
            if runner is not None:
                await runner.cleanup()


async def for_request(request: web.Request, slug: str = SLUG) -> GitHubOAuthFlow | None:
    """Select the opt-in Codex adapter, leaving Kiro's engine untouched."""
    if slug != SLUG or not await asyncio.to_thread(enabled):
        return None
    from kiro_crew.config import config_dir

    holder = request.app.get(APP_STATE_KEY)
    if holder is None:
        raise RuntimeError("native OAuth lifecycle not registered")
    if "github" not in holder:
        home = await asyncio.to_thread(config_dir)
        # A concurrent first request may have installed the manager while resolving home.
        holder.setdefault("github", GitHubOAuthFlow(home))
    return holder["github"]
