"""Dormant Copilot preparation: offline contracts, not live safety evidence."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew import agent as agent_mod
from kiro_crew.acp import session_mcp
from kiro_crew.acp.client import AcpClient, AcpError
from kiro_crew.acp.types import JsonRpcMessage
from kiro_crew.agent_sdk import backends, host_auth
from kiro_crew.agent_sdk.backends import ACP_BACKEND_COPILOT
from kiro_crew.providers.mirrors import Concern, mirror_for


def test_dormant_backend_cannot_be_selected_or_registered(monkeypatch):
    monkeypatch.setattr(backends, "_baseline", set(backends._baseline))
    monkeypatch.setattr(backends, "_selectable", set(backends._selectable))
    before = backends.selectable_backends()
    assert ACP_BACKEND_COPILOT in backends.ACP_BACKENDS_KNOWN
    assert ACP_BACKEND_COPILOT not in backends.BASELINE_SELECTABLE_BACKENDS
    assert backends.routing_for(ACP_BACKEND_COPILOT) is backends.Routing.UNVERIFIED
    with pytest.raises(ValueError, match="unverified"):
        backends.register_selectable_backend(ACP_BACKEND_COPILOT)
    assert backends.selectable_backends() == before
    assert backends.resolve_selected_backend(ACP_BACKEND_COPILOT) == backends.ACP_BACKEND_KIRO


def test_launch_is_native_acp_without_permission_bypass_flags():
    launch = backends.launch_for(ACP_BACKEND_COPILOT)
    assert launch.binary == "copilot"
    assert launch.acp_args == ("--acp", "--stdio")
    assert launch.bin_env_var == "COPILOT_BIN"
    assert launch.protocol_version == 1
    assert ACP_BACKEND_COPILOT not in backends.ACP_BACKENDS_ACP_RUNTIME
    assert ACP_BACKEND_COPILOT not in backends.ACP_BACKENDS_INTERNAL_SANDBOX
    assert ACP_BACKEND_COPILOT not in backends.ACP_BACKENDS_MEMBER_DISPATCH


def test_credentials_have_no_unverified_sandbox_carveout():
    declaration = host_auth.declaration_for(ACP_BACKEND_COPILOT)
    assert declaration.credential_leaves == (".copilot/config.json",)
    assert declaration.home_override_env_vars == ("COPILOT_HOME",)
    assert declaration.override_relative_leaves == ("config.json",)
    assert declaration.adapter_own_leaves == ()
    assert not declaration.host_logout_retires_children


@pytest.fixture
def copilot_spec(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    agents.mkdir()
    monkeypatch.setattr(agent_mod, "KIRO_AGENTS_DIR", agents)
    monkeypatch.setattr(agent_mod, "_KIRO_MCP_JSON", tmp_path / "mcp.json")
    monkeypatch.setattr(session_mcp, "ensure_agent_materialized", lambda _a: True)
    monkeypatch.setattr(session_mcp, "_mcp_registry_mode", lambda: False)
    monkeypatch.setattr(session_mcp, "managed_mcp_spec_entry", lambda _name: None)
    spec = {
        "name": "copilot-test",
        "tools": ["@allowed", "@narrowed"],
        "mcpServers": {
            "allowed": {"command": "/synthetic/server", "args": ["serve"]},
            "narrowed": {"command": "/synthetic/server", "disabledTools": ["write"]},
            "unreferenced": {"command": "/synthetic/server"},
        },
    }
    (agents / "copilot-test.json").write_text(json.dumps(spec), encoding="utf-8")
    return "copilot-test"


def test_projection_filters_servers_and_cannot_readd_narrowed_stub(copilot_spec):
    mirror = mirror_for(ACP_BACKEND_COPILOT)
    assert set(mirror.rulings()) == set(Concern)
    assert "unmeasured" in mirror.rulings()[Concern.MCP_SERVERS].reason
    stub = {"name": "narrowed", "command": "/synthetic/stub", "args": [], "env": []}
    projection = mirror.session_projection(
        copilot_spec, stub_server_names=("narrowed",), stub_elements=[stub]
    )
    assert [entry["name"] for entry in projection.params["mcpServers"]] == ["allowed"]
    assert projection.params["mcpServers"][0]["args"] == ["serve"]
    assert projection.denied_tools == frozenset()
    assert mirror.session_params(copilot_spec) == projection.params


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_handshake_sends_cached_mcp_array_on_new_and_load(tmp_path, monkeypatch, resume):
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_COPILOT)
    client._process = MagicMock()
    client._process.returncode = None
    client._process.stdin.drain = AsyncMock()
    client._resume_session_id = "prior" if resume else None
    elements = [{"name": "fixture", "command": "/synthetic/server", "args": [], "env": []}]
    client._session_mcp_cache = elements
    sent = []

    async def send(method, params):
        sent.append((method, params))
        return len(sent)

    async def response(request_id, **kwargs):
        method = sent[request_id - 1][0]
        if method == "initialize":
            return {"protocolVersion": 1, "agentCapabilities": {"loadSession": True}}
        if method in {"session/new", "session/load"}:
            return {
                "sessionId": "fresh",
                "modes": {},
                "models": {
                    "currentModelId": "fixture-model",
                    "availableModels": [{"modelId": "fixture-model", "name": "Fixture"}],
                },
            }
        return {}

    monkeypatch.setattr(client, "_send_request", AsyncMock(side_effect=send))
    monkeypatch.setattr(client, "_wait_for_response", AsyncMock(side_effect=response))
    monkeypatch.setattr(client, "_drain_notifications", AsyncMock())
    monkeypatch.setattr(client, "_persist_advertised_models_if_changed", AsyncMock())
    await asyncio.wait_for(client._initialize_session(), timeout=10)
    assert sent[0][1]["protocolVersion"] == 1
    sessions = [
        (method, params) for method, params in sent if method in {"session/new", "session/load"}
    ]
    assert len(sessions) == 1
    assert sessions[0][0] == ("session/load" if resume else "session/new")
    assert sessions[0][1]["mcpServers"] == elements
    assert "_meta" not in sessions[0][1]
    assert client._session_id == ("prior" if resume else "fresh")
    assert client._resolved_model_id == "fixture-model"
    assert "session/set_model" not in [method for method, _ in sent]
    await asyncio.wait_for(client.set_model("fixture-model"), timeout=10)
    assert sent[-1] == (
        "session/set_model",
        {"sessionId": client._session_id, "modelId": "fixture-model"},
    )
    assert "session/set_mode" not in [method for method, _ in sent]


@pytest.mark.asyncio
@pytest.mark.parametrize("allow", [False, True])
async def test_synthetic_permission_reply_uses_the_advertised_option(tmp_path, monkeypatch, allow):
    """Pins Crew's reply only; does not prove Copilot asks or honors a refusal."""
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_COPILOT)
    message = JsonRpcMessage(
        id=41,
        method="session/request_permission",
        params={
            "toolCall": {"toolCallId": "synthetic-call", "title": "fixture_tool"},
            "options": [
                {"optionId": "fixture-yes", "name": "Yes", "kind": "allow_once"},
                {"optionId": "fixture-no", "name": "No", "kind": "reject_once"},
            ],
        },
    )
    client._build_permission_event(message)
    reply = AsyncMock()
    monkeypatch.setattr(client, "_send_response", reply)
    action = client.approve_tool if allow else client.reject_tool
    await asyncio.wait_for(action(41), timeout=5)
    reply.assert_awaited_once_with(
        41,
        {"outcome": {"outcome": "selected", "optionId": "fixture-yes" if allow else "fixture-no"}},
    )
    assert 41 not in client._permission_options


@pytest.mark.asyncio
async def test_session_error_propagates_without_claiming_a_session(tmp_path, monkeypatch):
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_COPILOT)
    monkeypatch.setattr(client, "_send_request", AsyncMock(return_value=1))
    monkeypatch.setattr(
        client, "_wait_for_response", AsyncMock(side_effect=AcpError("synthetic sign-in failure"))
    )
    with pytest.raises(AcpError, match="synthetic sign-in failure"):
        await asyncio.wait_for(client._new_session_following_substitution(), timeout=10)
    assert client._session_id is None
