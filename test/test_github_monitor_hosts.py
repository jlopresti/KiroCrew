"""Configurable GitHub Enterprise PR monitoring, without contacting a server."""

import json
from types import SimpleNamespace

import pytest

from kiro_crew.config import KiroCrewConfig
from kiro_crew.config.schema import SCHEMA_REGISTRY
from kiro_crew.config.sections import MonitoringConfig
from kiro_crew.github_hosts import normalize_github_hosts
from kiro_crew.monitoring.github_pull_request import parse_github_pull_request_target
from kiro_crew.monitoring.targets import infer_pull_request_kind, normalize_pull_request_target
from kiro_crew.probes.gh_pr import PrWatchProbe
from kiro_crew.probes.targets import infer

HOST = "github.corp.example"
URL = f"https://{HOST}/owner/repo/pull/42"


@pytest.fixture
def configured(monkeypatch):
    config = KiroCrewConfig(monitoring=MonitoringConfig(github_hosts=["github.com", HOST]))
    monkeypatch.setattr(KiroCrewConfig, "load", lambda: config)
    return config


@pytest.mark.parametrize("raw", [None, "github.corp.example", {}, 42])
def test_malformed_config_fails_closed(raw):
    assert normalize_github_hosts(raw) == []


@pytest.mark.parametrize("hosts", [[], None, "github.com", ["https://github.corp.example"]])
def test_loader_does_not_authorize_malformed_lists(tmp_path, monkeypatch, hosts):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"monitoring": {"github_hosts": hosts}}), encoding="utf-8")
    monkeypatch.setattr("kiro_crew.config.loader.config_path", lambda: path)
    expected = [] if isinstance(hosts, list) else ["github.com"]
    assert KiroCrewConfig.load().monitoring.github_hosts == expected
    with pytest.raises(ValueError):
        parse_github_pull_request_target(URL)


def test_config_round_trip_and_schema(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr("kiro_crew.config.loader.config_path", lambda: path)
    config = KiroCrewConfig(monitoring=MonitoringConfig(github_hosts=["github.com", HOST]))
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert KiroCrewConfig.load().monitoring.github_hosts == ["github.com", HOST]
    entry = next(e for e in SCHEMA_REGISTRY if e.path == "monitoring.github_hosts")
    assert entry.type == "array"
    assert entry.default_value == ["github.com"]
    assert not entry.requires_restart


def test_host_normalization_is_not_url_sanitization():
    assert normalize_github_hosts(
        [
            " GITHUB.CORP.EXAMPLE ",
            HOST,
            "https://evil.example",
            "*.example",
            "user@evil.example",
            "evil.example:443",
            "evil.example/path",
            "bad..example",
            "evil.example?x",
            "evil.example#x",
            "evil.example\n.other",
            "-bad.example",
            42,
        ]
    ) == [HOST]


def test_enterprise_target_and_arming(configured):
    parsed = parse_github_pull_request_target(URL)
    assert parsed.host == HOST
    assert parsed.identity == f"{HOST}/owner/repo#42"
    assert parsed.url == URL
    assert infer_pull_request_kind(URL, gitlab_hosts=[]) == "github_pull_request"
    assert normalize_pull_request_target("github_pull_request", URL, gitlab_hosts=[]) == URL
    target = infer(f"Watch {URL} until green")
    assert target is not None
    assert target.host_key == HOST
    assert json.loads(target.message)["host"] == HOST


@pytest.mark.parametrize(
    "url",
    [
        "https://other.example/owner/repo/pull/42",
        f"http://{HOST}/owner/repo/pull/42",
        f"https://user@{HOST}/owner/repo/pull/42",
        f"https://{HOST}:443/owner/repo/pull/42",
        f"https://{HOST}.evil.example/owner/repo/pull/42",
        URL + "?q=1",
        URL + "#fragment",
    ],
)
def test_enterprise_does_not_relax_url_validation(configured, url):
    with pytest.raises(ValueError):
        parse_github_pull_request_target(url)


def test_inference_refuses_ambiguous_hosts(configured):
    assert infer(f"Watch {URL} and https://github.com/owner/repo/pull/42") is None
    assert infer(f"Watch {URL} and https://unknown.example/owner/repo/pull/42") is None


def test_legacy_probe_host_identity_and_revocation(configured, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "kiro_crew.probes.gh_pr._run_gh",
        lambda args, pin_host="": calls.append((args, pin_host)) or (0, '{"state": "MERGED"}'),
    )
    probe = PrWatchProbe()
    ctx = SimpleNamespace(message=json.dumps({"repo": "owner/repo", "pr": 42, "host": HOST}))
    assert probe.identity(ctx) == ("gh-pr", f"{HOST}/owner/repo#42")
    probe.observe(ctx)
    assert calls[0][1] == HOST
    configured.monitoring.github_hosts = ["github.com"]
    with pytest.raises(ValueError, match="monitoring.github_hosts"):
        probe.identity(ctx)
