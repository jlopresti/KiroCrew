"""Opt-in, account-free handshake probe; never a permission-routing proof."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from kiro_crew import platform_compat


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_explicit_copilot_binary_speaks_acp(tmp_path, record_property):
    # An explicit path opts in. Never probe an installed binary during collection
    # or use ambient auth tokens; this sends initialize only, not a model prompt.
    configured = os.environ.get("KIROCREW_TEST_COPILOT_BIN")
    if not configured:
        pytest.skip("set KIROCREW_TEST_COPILOT_BIN to an absolute Copilot executable")
    binary = Path(configured)
    assert binary.is_absolute() and binary.is_file(), "explicit Copilot executable required"
    binary = binary.resolve()
    assert "shims" not in binary.parts, "use a real executable, not a version-manager shim"
    node = shutil.which("node")
    node_dir = Path(node).resolve().parent if node else binary.parent
    assert "shims" not in node_dir.parts, "resolve Node before changing HOME"
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update(
        {
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
            "COPILOT_HOME": str(tmp_path / "copilot"),
            "COPILOT_CACHE_HOME": str(tmp_path / "cache"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "TMPDIR": str(tmp_path),
            "TEMP": str(tmp_path),
            "TMP": str(tmp_path),
            "PATH": os.pathsep.join([str(binary.parent), str(node_dir), os.defpath]),
        }
    )
    proc = await asyncio.wait_for(
        asyncio.create_subprocess_exec(
            str(binary),
            "--acp",
            "--stdio",
            cwd=tmp_path,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=platform_compat.IS_POSIX,
            creationflags=platform_compat.CREATE_NEW_PROCESS_GROUP,
        ),
        timeout=10,
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "kirocrew-handshake-test", "version": "0"},
            },
        }
        proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        await asyncio.wait_for(proc.stdin.drain(), timeout=5)
        async with asyncio.timeout(30):
            while True:
                line = await proc.stdout.readline()
                assert line, "Copilot exited before answering initialize"
                frame = json.loads(line)
                if frame.get("id") == 1:
                    assert "error" not in frame, "Copilot rejected initialize"
                    result = frame["result"]
                    assert result["protocolVersion"] == 1
                    assert result.get("agentInfo", {}).get("version")
                    record_property("copilot_version", result["agentInfo"]["version"])
                    break
    finally:
        platform_compat.kill_process_tree(proc.pid, platform_compat.SIGKILL)
        await asyncio.wait_for(proc.wait(), timeout=5)
