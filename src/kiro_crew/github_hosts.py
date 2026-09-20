"""Exact, operator-configured GitHub hosts for credentialed monitor reads."""

from __future__ import annotations

import re

PUBLIC_GITHUB_HOST = "github.com"
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def normalize_github_hosts(raw: object) -> list[str]:
    """Drop malformed entries; never turn URLs, ports or wildcards into hosts."""
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        host = entry.strip().lower()
        if not host or len(host) > 253:
            continue
        if not all(_DNS_LABEL.fullmatch(label) for label in host.split(".")):
            continue
        if host not in result:
            result.append(host)
    return result


def github_host_allowed(host: str) -> bool:
    """Read current configuration so removing a host also revokes existing watches."""
    from kiro_crew.config import KiroCrewConfig

    return host in normalize_github_hosts(KiroCrewConfig.load().monitoring.github_hosts)
