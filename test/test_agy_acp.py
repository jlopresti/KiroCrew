"""Offline tests of the internal bridge; not evidence of native AGY tool safety."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kiro_crew import agy_acp
from kiro_crew.acp import client as client_mod
from kiro_crew.agent_sdk import backends, host_auth
from kiro_crew.agent_sdk.backend_install import INSTALLED, MISSING, _probe_agy
from kiro_crew.agent_sdk.backends import ACP_BACKEND_AGY


async def request(bridge, method, **params):
    await asyncio.wait_for(
        bridge.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            }
        ),
        timeout=5,
    )


@pytest_asyncio.fixture
async def harness(tmp_path, monkeypatch):
    frames = []
    bridge = agy_acp.AgyBridge("/synthetic/agy", frames.append)
    proc = MagicMock()
    proc.pid = 12345678
    proc.stdin.drain = AsyncMock()
    proc.stdout = asyncio.StreamReader(limit=agy_acp.MAX_FRAME_BYTES)
    proc.wait = AsyncMock(return_value=0)
    spawn = AsyncMock(return_value=proc)
    kill = MagicMock()
    monkeypatch.setattr(agy_acp.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(agy_acp.platform_compat, "kill_process_tree", kill)
    await request(bridge, "initialize", protocolVersion=1)
    await request(bridge, "session/new", cwd=str(tmp_path), mcpServers=[])
    try:
        yield bridge, frames, proc, spawn, kill
    finally:
        await asyncio.wait_for(bridge.cancel(), timeout=5)
        await asyncio.wait_for(bridge.stop(), timeout=5)


def feed(proc, *events):
    for event in events:
        proc.stdout.feed_data((json.dumps(event) + "\n").encode())


async def prompt(bridge, text="hello"):
    await request(
        bridge,
        "session/prompt",
        sessionId=bridge.session_id,
        prompt=[{"type": "text", "text": text}],
    )
    assert bridge.prompt_task is not None
    await asyncio.wait_for(bridge.prompt_task, timeout=5)


def test_dormant_and_no_unverified_capabilities(monkeypatch):
    monkeypatch.setattr(backends, "_baseline", set(backends._baseline))
    monkeypatch.setattr(backends, "_selectable", set(backends._selectable))
    assert ACP_BACKEND_AGY in backends.ACP_BACKENDS_KNOWN
    assert ACP_BACKEND_AGY not in backends.BASELINE_SELECTABLE_BACKENDS
    assert backends.routing_for(ACP_BACKEND_AGY) is backends.Routing.UNVERIFIED
    with pytest.raises(ValueError, match="unverified"):
        backends.register_selectable_backend(ACP_BACKEND_AGY)
    assert backends.resolve_selected_backend(ACP_BACKEND_AGY) == backends.ACP_BACKEND_KIRO
    for members in (
        backends.ACP_BACKENDS_SELF_SERVED_ACP,
        backends.ACP_BACKENDS_INTERNAL_SANDBOX,
        backends.ACP_BACKENDS_ACP_RUNTIME,
        backends.ACP_BACKENDS_MEMBER_DISPATCH,
    ):
        assert ACP_BACKEND_AGY not in members
    assert client_mod._PROTOCOL_VERSION_BY_BACKEND[ACP_BACKEND_AGY] == 1


def test_auth_has_no_invented_file_or_sandbox_exemption():
    auth = host_auth.declaration_for(ACP_BACKEND_AGY)
    assert auth.entitlement_source == host_auth.ENTITLEMENT_OWN_CREDENTIAL_STORE
    assert auth.credential_leaves == auth.adapter_own_leaves == ()
    assert not auth.host_logout_retires_children


@pytest.mark.parametrize("available", [False, True])
def test_resolver_and_install_probe_are_binary_only(monkeypatch, available):
    monkeypatch.delenv("AGY_BIN", raising=False)
    monkeypatch.setattr(client_mod, "augmented_path", lambda _p: "/synthetic/path")
    which = MagicMock(return_value="/synthetic/agy" if available else None)
    monkeypatch.setattr(client_mod.shutil, "which", which)
    assert client_mod._resolve_agy_bin() == (which.return_value, "/synthetic/path")
    assert _probe_agy().installed == (INSTALLED if available else MISSING)
    which.assert_called_with("agy", path="/synthetic/path")


@pytest.mark.parametrize("executable", [False, True])
def test_explicit_binary_never_falls_back(monkeypatch, executable):
    monkeypatch.setenv("AGY_BIN", "/synthetic/custom")
    monkeypatch.setattr(client_mod, "augmented_path", lambda _p: "/synthetic/path")
    monkeypatch.setattr(client_mod.platform_compat, "is_executable_file", lambda _p: executable)
    which = MagicMock(side_effect=AssertionError("must not search PATH"))
    monkeypatch.setattr(client_mod.shutil, "which", which)
    assert client_mod._resolve_agy_bin()[0] == ("/synthetic/custom" if executable else None)


@pytest.mark.asyncio
async def test_initialize_is_honest_and_lazy(harness):
    bridge, frames, proc, spawn, kill = harness
    caps = frames[0]["result"]["agentCapabilities"]
    assert caps["loadSession"] is False
    assert not any(caps["promptCapabilities"].values())
    assert not any(caps["mcpCapabilities"].values())
    spawn.assert_not_called()
    await request(bridge, "session/load", sessionId="old")
    assert frames[-1]["error"]["code"] == -32601


@pytest.mark.asyncio
@pytest.mark.parametrize("servers", [[{"name": "must-not-disappear"}], None, {}])
async def test_mcp_is_refused_not_silently_dropped(tmp_path, servers):
    frames = []
    bridge = agy_acp.AgyBridge("unused", frames.append)
    await request(bridge, "initialize", protocolVersion=1)
    await request(bridge, "session/new", cwd=str(tmp_path), mcpServers=servers)
    assert "error" in frames[-1]
    assert bridge.session_id == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocks", [[], [{"type": "image", "data": "x"}], [{"type": "text", "text": " "}], [None]]
)
async def test_invalid_prompt_never_starts_native_cli(harness, blocks):
    bridge, frames, proc, spawn, kill = harness
    await request(bridge, "session/prompt", sessionId=bridge.session_id, prompt=blocks)
    assert "error" in frames[-1]
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_stream_deltas_and_two_turns_share_one_process(harness):
    bridge, frames, proc, spawn, kill = harness
    feed(
        proc,
        {"event": "init", "conversation_id": "native"},
        {
            "event": "step_update",
            "step_update": {
                "step_type": "agent_response",
                "state": "ACTIVE",
                "text_delta": "Hello ",
            },
        },
        {
            "event": "step_update",
            "step_update": {"step_type": "agent_response", "state": "DONE", "text_delta": "world"},
        },
        {"event": "result", "result": {"status": "SUCCESS", "response": "Hello world"}},
    )
    await prompt(bridge)
    chunks = [f["params"]["update"]["content"]["text"] for f in frames if "method" in f]
    assert chunks == ["Hello ", "world"]
    feed(proc, {"event": "result", "result": {"status": "SUCCESS", "response": "Fallback"}})
    await prompt(bridge, "second turn")
    assert frames[-2]["params"]["update"]["content"]["text"] == "Fallback"
    assert frames[-1]["result"] == {"stopReason": "end_turn"}
    spawn.assert_awaited_once()
    assert spawn.call_args.args == (
        "/synthetic/agy",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
    )
    assert spawn.call_args.kwargs["cwd"] == bridge.cwd
    assert json.loads(proc.stdin.write.call_args.args[0]) == {
        "event": "user",
        "message": {"content": "second turn"},
    }
    kill.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_model_only_before_first_prompt(harness):
    bridge, frames, proc, spawn, kill = harness
    await request(bridge, "session/set_model", sessionId=bridge.session_id, modelId="fixture-model")
    feed(proc, {"event": "result", "result": {"status": "SUCCESS"}})
    await prompt(bridge)
    assert spawn.call_args.args[-2:] == ("--model", "fixture-model")
    await request(bridge, "session/set_model", sessionId=bridge.session_id, modelId="other")
    assert "error" in frames[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_tools_are_observations_not_permission_requests(harness, failed):
    bridge, frames, proc, spawn, kill = harness
    feed(
        proc,
        {
            "event": "step_update",
            "step_update": {
                "step_type": "tool",
                "step_index": 2,
                "state": "DONE",
                "tool_name": "fixture",
                "tool_info": {"parameters": {"x": 1}, "output": {"ok": True}, "error": failed},
            },
        },
        {"event": "result", "result": {"status": "SUCCESS"}},
    )
    await prompt(bridge)
    updates = [f["params"]["update"] for f in frames if "method" in f]
    assert [u["sessionUpdate"] for u in updates] == ["tool_call", "tool_call_update"]
    assert updates[0]["rawInput"] == {"x": 1}
    assert updates[1]["status"] == ("failed" if failed else "completed")
    assert {f["method"] for f in frames if "method" in f} == {"session/update"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        b"not-json\n",
        b"[]\n",
        b"",
        b'{"event":"result","result":{"status":"ERROR","error":"SECRET"}}\n',
    ],
)
async def test_broken_stream_fails_and_reaps_without_echoing_native_errors(harness, data):
    bridge, frames, proc, spawn, kill = harness
    proc.stdout.feed_data(data)
    proc.stdout.feed_eof()
    await prompt(bridge)
    assert frames[-1]["error"]["code"] == -32000
    assert "SECRET" not in json.dumps(frames)
    assert bridge.closed
    kill.assert_called_once()
    proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_before_task_runs_still_replies(harness):
    bridge, frames, proc, spawn, kill = harness
    await bridge.handle(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "session/prompt",
            "params": {
                "sessionId": bridge.session_id,
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }
    )
    await asyncio.wait_for(bridge.cancel(), timeout=5)
    replies = [f for f in frames if f.get("id") == 99]
    assert replies == [{"jsonrpc": "2.0", "id": 99, "result": {"stopReason": "cancelled"}}]
    assert bridge.closed


@pytest.mark.asyncio
async def test_cancel_during_process_creation_reaps_late_child(harness):
    bridge, frames, proc, spawn, kill = harness
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_spawn(*args, **kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=3)
        return proc

    spawn.side_effect = slow_spawn
    await request(
        bridge,
        "session/prompt",
        sessionId=bridge.session_id,
        prompt=[{"type": "text", "text": "hello"}],
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        bridge.prompt_task.cancel()
        release.set()
        await asyncio.wait_for(bridge.prompt_task, timeout=5)
        assert frames[-1]["result"] == {"stopReason": "cancelled"}
        kill.assert_called_once()
        proc.wait.assert_awaited_once()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_timeout_reaps_child(harness, monkeypatch):
    bridge, frames, proc, spawn, kill = harness
    # Start outside the zero-timeout turn so this tests cleanup of a live handle.
    await asyncio.wait_for(bridge.start(), timeout=5)
    monkeypatch.setattr(agy_acp, "TURN_TIMEOUT", 0)
    await prompt(bridge)
    assert "error" in frames[-1]
    kill.assert_called_once()
    proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_prompt_notification_is_ignored(harness):
    bridge, frames, proc, spawn, kill = harness
    count = len(frames)
    await bridge.handle(
        {
            "jsonrpc": "2.0",
            "method": "session/prompt",
            "params": {
                "sessionId": bridge.session_id,
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }
    )
    assert len(frames) == count
    assert bridge.prompt_task is None
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_failure_still_answers_and_retains_handle_for_retry(harness):
    bridge, frames, proc, spawn, kill = harness
    proc.wait.side_effect = [TimeoutError(), 0]
    proc.stdout.feed_eof()
    await prompt(bridge)
    assert "cleanup failed" in frames[-1]["error"]["message"]
    assert bridge.process is proc
    await asyncio.wait_for(bridge.stop(), timeout=5)
    assert bridge.process is None
    assert proc.wait.await_count == 2


@pytest.mark.asyncio
async def test_busy_prompt_is_refused_and_active_turn_is_cancelled(harness):
    bridge, frames, proc, spawn, kill = harness
    await request(
        bridge,
        "session/prompt",
        sessionId=bridge.session_id,
        prompt=[{"type": "text", "text": "first"}],
    )
    await request(
        bridge,
        "session/prompt",
        sessionId=bridge.session_id,
        prompt=[{"type": "text", "text": "second"}],
    )
    assert "already in flight" in frames[-1]["error"]["message"]
    await asyncio.wait_for(bridge.cancel(), timeout=5)
    assert frames[-1]["result"] == {"stopReason": "cancelled"}
    await request(
        bridge,
        "session/prompt",
        sessionId=bridge.session_id,
        prompt=[{"type": "text", "text": "third"}],
    )
    assert "session ended" in frames[-1]["error"]["message"]


@pytest.mark.asyncio
async def test_already_exited_process_is_still_reaped(harness):
    bridge, frames, proc, spawn, kill = harness
    kill.side_effect = ProcessLookupError()
    proc.stdout.feed_eof()
    await prompt(bridge)
    assert frames[-1]["error"]["code"] == -32000
    proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_stdio_protocol_and_eof_never_launch_agy(tmp_path, monkeypatch):
    wire = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {
                "cwd": str(tmp_path),
                "mcpServers": [],
            },
        },
    ]
    stdin = SimpleNamespace(
        buffer=io.BytesIO(("invalid\n" + "\n".join(json.dumps(f) for f in wire) + "\n").encode())
    )
    stdout = io.StringIO()
    spawn = AsyncMock(side_effect=AssertionError("handshake must not launch native CLI"))
    # Restore the capture streams before making assertions so failure reports
    # cannot accidentally be written to this protocol fixture.
    with monkeypatch.context() as patch:
        patch.setattr(agy_acp.sys, "stdin", stdin)
        patch.setattr(agy_acp.sys, "stdout", stdout)
        patch.setattr(agy_acp.asyncio, "create_subprocess_exec", spawn)
        await asyncio.wait_for(agy_acp.serve("unused"), timeout=5)
    frames = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert frames[0]["error"]["code"] == -32700
    assert frames[1]["result"]["protocolVersion"] == 1
    assert frames[2]["result"]["sessionId"]
    spawn.assert_not_called()
