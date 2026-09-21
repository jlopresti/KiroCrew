"""Dormant ACP bridge to Antigravity's documented headless JSON stream.

This is not a permission bridge: AGY decides its tools itself. Never advertise
this transport as a selectable Crew backend until that boundary is resolved.
No terminal scraping, SQLite access, settings writes or automatic installation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiro_crew import platform_compat

MAX_FRAME_BYTES = 4 * 1024 * 1024
TURN_TIMEOUT = 300


class BridgeError(Exception):
    """An actionable protocol refusal, without native stderr or credential data."""

    def __init__(self, message: str, code: int = -32602):
        super().__init__(message)
        self.code = code


class AgyBridge:
    """One ACP session, one lazily started AGY process, no shared host state."""

    def __init__(self, binary: str, emit: Callable[[dict[str, Any]], None]):
        self.binary = binary
        self.emit = emit
        self.session_id = ""
        self.cwd = ""
        self.model = "auto"
        self.initialized = False
        self.process: asyncio.subprocess.Process | None = None
        self.prompt_task: asyncio.Task[None] | None = None
        self.prompt_id: Any = None
        self.closed = False

    def result(self, request_id: Any, result: dict[str, Any]) -> None:
        self.emit({"jsonrpc": "2.0", "id": request_id, "result": result})

    def error(self, request_id: Any, message: str, code: int = -32602) -> None:
        self.emit(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": code,
                    "message": message,
                },
            }
        )

    def update(self, update: dict[str, Any]) -> None:
        self.emit(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": self.session_id,
                    "update": update,
                },
            }
        )

    def require_session(self, params: dict[str, Any]) -> None:
        if not self.session_id or params.get("sessionId") != self.session_id:
            raise BridgeError("Unknown AGY bridge session")
        if self.closed:
            raise BridgeError("AGY session ended; start a new Crew session", -32000)

    async def handle(self, message: Any) -> None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            self.error(None, "Invalid JSON-RPC request", -32600)
            return
        request_id = message.get("id")
        method = message.get("method")
        # Only cancellation is a notification with side effects. Never execute
        # a prompt sent without a request id (there would be no terminal reply).
        if "id" not in message and method != "session/cancel":
            return
        params = message.get("params", {})
        try:
            if not isinstance(params, dict):
                raise BridgeError("Parameters must be an object")
            if method == "initialize":
                if self.initialized or params.get("protocolVersion") != 1:
                    raise BridgeError("AGY bridge requires one ACP version 1 initialization")
                self.initialized = True
                self.result(
                    request_id,
                    {
                        "protocolVersion": 1,
                        "agentInfo": {"name": "kirocrew-agy-bridge", "version": "0.1.0"},
                        "agentCapabilities": {
                            "loadSession": False,
                            "promptCapabilities": {
                                "image": False,
                                "audio": False,
                                "embeddedContext": False,
                            },
                            "mcpCapabilities": {"http": False, "sse": False},
                        },
                        "authMethods": [],
                    },
                )
                return
            if not self.initialized:
                raise BridgeError("Initialize the AGY bridge first")
            if method == "session/new":
                if self.session_id:
                    raise BridgeError("AGY bridge serves one session per process")
                if params.get("mcpServers", []) != []:
                    raise BridgeError(
                        "AGY bridge cannot forward MCP servers; refusing to drop them"
                    )
                cwd = params.get("cwd")
                if (
                    not isinstance(cwd, str)
                    or not Path(cwd).is_absolute()
                    or not Path(cwd).is_dir()
                ):
                    raise BridgeError("AGY requires an existing absolute working directory")
                self.cwd = cwd
                self.session_id = str(uuid.uuid4())
                self.result(request_id, {"sessionId": self.session_id})
                return
            if method in {"session/load", "session/resume", "authenticate", "session/set_mode"}:
                raise BridgeError("This AGY bridge method is not supported", -32601)
            self.require_session(params)
            if method == "session/set_model":
                model = params.get("modelId")
                if self.process is not None or (self.prompt_task and not self.prompt_task.done()):
                    raise BridgeError("AGY model changes require a new session")
                if not isinstance(model, str) or not model.strip() or model.startswith("-"):
                    raise BridgeError("AGY requires a non-empty model id")
                self.model = model
                self.result(request_id, {})
            elif method == "session/prompt":
                if self.prompt_task and not self.prompt_task.done():
                    raise BridgeError("An AGY prompt is already in flight", -32000)
                blocks = params.get("prompt")
                if not isinstance(blocks, list) or not blocks:
                    raise BridgeError("AGY requires text prompt blocks")
                if any(
                    not isinstance(b, dict)
                    or b.get("type") != "text"
                    or not isinstance(b.get("text"), str)
                    for b in blocks
                ):
                    raise BridgeError(
                        "AGY bridge supports text only; unsupported content was not sent"
                    )
                text = "\n".join(b["text"] for b in blocks)
                if not text.strip():
                    raise BridgeError("AGY requires a non-empty prompt")
                self.prompt_id = request_id
                self.prompt_task = asyncio.create_task(self.prompt(request_id, text))
            elif method == "session/cancel":
                await self.cancel()
                if "id" in message:
                    self.result(request_id, {})
            else:
                raise BridgeError("Unknown AGY bridge method", -32601)
        except BridgeError as exc:
            if "id" in message:
                self.error(request_id, str(exc), exc.code)

    async def start(self) -> None:
        argv = [self.binary, "--input-format", "stream-json", "--output-format", "stream-json"]
        if self.model != "auto":
            argv.extend(["--model", self.model])
        # Never add skip-permissions. Headless policies are NOT a Crew gate.
        self.process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=MAX_FRAME_BYTES,
            start_new_session=platform_compat.IS_POSIX,
            creationflags=platform_compat.CREATE_NEW_PROCESS_GROUP,
        )

    async def prompt(self, request_id: Any, text: str) -> None:
        seen_text = False
        tools: set[str] = set()
        try:
            async with asyncio.timeout(TURN_TIMEOUT):
                if self.process is None:
                    # Cancellation can race subprocess creation. Retain and reap
                    # the child even if cancellation arrives before its handle.
                    starting = asyncio.create_task(self.start())
                    try:
                        await asyncio.shield(starting)
                    except asyncio.CancelledError:
                        await starting
                        raise
                proc = self.process
                assert proc is not None and proc.stdin is not None and proc.stdout is not None
                payload = {"event": "user", "message": {"content": text}}
                proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                await proc.stdin.drain()
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        raise BridgeError(
                            "AGY exited before a result; check installation and sign-in with agy",
                            -32000,
                        )
                    frame = json.loads(line)
                    if not isinstance(frame, dict):
                        raise BridgeError("Invalid AGY stream frame", -32000)
                    event = frame.get("event")
                    if event == "step_update":
                        step = frame.get("step_update")
                        if not isinstance(step, dict):
                            raise BridgeError("Invalid AGY step", -32000)
                        if step.get("step_type") == "agent_response":
                            delta = step.get("text_delta")
                            if isinstance(delta, str) and delta:
                                seen_text = True
                                self.update(
                                    {
                                        "sessionUpdate": "agent_message_chunk",
                                        "content": {
                                            "type": "text",
                                            "text": delta,
                                        },
                                    }
                                )
                        elif step.get("step_type") == "tool":
                            self.tool_update(step, tools)
                    elif event == "result":
                        result = frame.get("result")
                        if not isinstance(result, dict):
                            raise BridgeError("Invalid AGY result", -32000)
                        status = result.get("status")
                        if status in {"CANCELED", "INTERRUPTED"}:
                            await self.finish_stopped(request_id, cancelled=True)
                            return
                        if status != "SUCCESS":
                            # Native errors can contain prompt/config secrets. Do not
                            # reflect them into JSON-RPC. SUCCESS itself can still
                            # include native soft denials; it is not gate evidence.
                            raise BridgeError(
                                "AGY did not complete the turn; check sign-in, model and native permissions",
                                -32000,
                            )
                        response = result.get("response")
                        if not seen_text and isinstance(response, str) and response:
                            self.update(
                                {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": {
                                        "type": "text",
                                        "text": response,
                                    },
                                }
                            )
                        self.result(request_id, {"stopReason": "end_turn"})
                        return
        except asyncio.CancelledError:
            await self.finish_stopped(request_id, cancelled=True)
        except (BridgeError, OSError, ValueError, TimeoutError):
            await self.finish_stopped(request_id, cancelled=False)

    async def finish_stopped(self, request_id: Any, *, cancelled: bool) -> None:
        """A failed cleanup must still give the waiting ACP caller a terminal error."""
        try:
            await self.stop()
        except (OSError, TimeoutError):
            self.error(
                request_id, "AGY process cleanup failed; close this bridge before retrying", -32000
            )
            return
        if cancelled:
            self.result(request_id, {"stopReason": "cancelled"})
        else:
            self.error(
                request_id,
                "AGY turn failed; verify the CLI, sign-in, model and stream format",
                -32000,
            )

    def tool_update(self, step: dict[str, Any], tools: set[str]) -> None:
        index = step.get("step_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise BridgeError("AGY tool event has no step index", -32000)
        tool_id = f"agy-step-{index}"
        info = step.get("tool_info") or {}
        if not isinstance(info, dict):
            raise BridgeError("Invalid AGY tool details", -32000)
        if tool_id not in tools:
            tools.add(tool_id)
            self.update(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tool_id,
                    "title": str(step.get("tool_name") or "AGY tool"),
                    "kind": "other",
                    "status": "in_progress",
                    "rawInput": info.get("parameters", {}),
                }
            )
        if step.get("state") == "DONE":
            output = info.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            self.update(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tool_id,
                    "status": "failed" if info.get("error") else "completed",
                    "content": [{"type": "content", "content": {"type": "text", "text": output}}],
                }
            )

    async def stop(self) -> None:
        self.closed = True
        proc = self.process
        if proc is not None:
            try:
                await asyncio.to_thread(
                    platform_compat.kill_process_tree, proc.pid, platform_compat.SIGKILL
                )
            except ProcessLookupError:
                pass  # The child already exited; still reap it below.
            finally:
                await asyncio.wait_for(proc.wait(), timeout=5)
            self.process = None

    async def cancel(self) -> None:
        task = self.prompt_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # A task cancelled before its first instruction never enters
                # prompt's handler, but still owes the ACP caller a response.
                await self.finish_stopped(self.prompt_id, cancelled=True)


async def serve(binary: str) -> None:
    def emit(frame: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(frame, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    bridge = AgyBridge(binary, emit)
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.buffer.readline, MAX_FRAME_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_FRAME_BYTES:
                bridge.error(None, "ACP frame exceeds the AGY bridge limit", -32600)
                break
            try:
                message = json.loads(line)
            except (ValueError, UnicodeError):
                bridge.error(None, "Invalid JSON", -32700)
                continue
            await bridge.handle(message)
    finally:
        await bridge.cancel()
        await bridge.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agy-bin", required=True)
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.agy_bin))
    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
