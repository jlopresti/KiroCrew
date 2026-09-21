"""Offline preparation for Copilot's documented ACP session MCP array.

No authenticated Copilot exchange has validated this projection. The backend
remains unselectable while its permission routing is UNVERIFIED.
"""

from __future__ import annotations

from typing import Any, Collection, Mapping

from kiro_crew.agent_sdk.backends import ACP_BACKEND_COPILOT
from kiro_crew.providers.mirrors.base import (
    AgentConfigMirror,
    Concern,
    Disposition,
    Ruling,
    SessionProjection,
)
from kiro_crew.providers.mirrors.opencode import opencode_projection


class CopilotMirror(AgentConfigMirror):
    """Prepare session parameters without touching Copilot's own settings."""

    backend = ACP_BACKEND_COPILOT

    def rulings(self) -> Mapping[Concern, Ruling]:
        return {
            Concern.MCP_SERVERS: Ruling(
                Disposition.TRANSLATED,
                "prepared as session/new and session/load mcpServers elements using "
                "the existing translator; documented by GitHub, but the authenticated "
                "round trip is unmeasured and the backend is not selectable",
            ),
            Concern.TOOL_ALLOWLIST: Ruling(
                Disposition.TRANSLATED,
                "filters the projected servers and pooled stubs before sending the array",
            ),
            Concern.DENIED_TOOLS: Ruling(
                Disposition.TRANSLATED,
                "withholds a narrowed server whole, including the control plane; no "
                "Copilot per-call tool identity or permission route has been verified",
            ),
            Concern.MODEL: Ruling(
                Disposition.WITHHELD,
                "not carried by the MCP array; the client resolves its model from the "
                "session's advertised catalog without a hardcoded fallback",
            ),
            Concern.MODEL_ALLOWLIST: Ruling(
                Disposition.WITHHELD,
                "the session's advertised catalog owns the available model vocabulary",
            ),
            Concern.AUTO_APPROVE: Ruling(
                Disposition.WITHHELD,
                "no automatic approval is projected; saved approvals and allow-all "
                "settings are unresolved routing risks, not a Crew safety guarantee",
            ),
            Concern.PERMISSION_MODE: Ruling(
                Disposition.WITHHELD,
                "no permission posture is asserted without a verified read-back and "
                "a denied-action sentinel; routing stays UNVERIFIED",
            ),
            Concern.PROMPT: Ruling(
                Disposition.WITHHELD,
                "the MCP array carries no agent definition; turn text is separate",
            ),
            Concern.RESOURCES: Ruling(
                Disposition.WITHHELD,
                "resource projection is outside this dormant transport preparation",
            ),
            Concern.HOOKS: Ruling(
                Disposition.NO_CHANNEL,
                "Crew does not translate agent-spec hooks into Copilot's hook format",
                channel="Copilot hook configuration; its mapping and permission "
                "semantics require separate authenticated validation",
            ),
        }

    def session_params(self, agent: str | None, **kwargs: Any) -> dict[str, Any]:
        return self.session_projection(agent, **kwargs).params

    def session_projection(
        self,
        agent: str | None,
        *,
        stub_server_names: Collection[str] = (),
        stub_elements: Collection[Mapping[str, Any]] = (),
        work_dir: object = None,
        session_key: str = "",
        channel_id: str = "",
        session_token: str = "",
        **kwargs: Any,
    ) -> SessionProjection:
        # Reuse host-side filtering, not the sibling's measured wire guarantees.
        return opencode_projection(
            agent,
            stub_server_names=stub_server_names,
            stub_elements=stub_elements,
            work_dir=work_dir,
            session_key=session_key,
            channel_id=channel_id,
            session_token=session_token,
        )
