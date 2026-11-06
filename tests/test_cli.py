import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur import cli
from mcp_for_cascadeur.settings import Config, load_config


def test_default_command_runs_stdio_server_with_saved_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    captured = []
    monkeypatch.setitem(sys.modules, "mcp_for_cascadeur.server", SimpleNamespace(main=captured.append))
    assert cli.main([]) == 0
    assert captured == [Config()]


def test_setup_uses_stable_runtime_and_configures_requested_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path / "settings"))
    home = tmp_path / "Cascadeur"
    calls = {}
    monkeypatch.setattr(cli.installation, "detect_cascadeur_home", lambda _: home)
    monkeypatch.setattr(cli.installation, "install_runtime", lambda config: tmp_path / "start_bridge.py")
    monkeypatch.setattr(cli.installation, "install_startup", lambda config, bridge: {"installed": True})
    monkeypatch.setattr(
        cli.installation, "configure_codex", lambda **kwargs: calls.update(kwargs) or {"changed": True}
    )
    workspace = tmp_path / "My scenes"
    assert (
        cli.main(
            [
                "setup",
                "--workspace",
                str(workspace),
                "--install-startup",
                "--allow-python",
                "--source",
                "git+https://example.test/repo@tag",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["startup"]["installed"]
    assert load_config().workspace == workspace
    assert load_config().allow_python
    assert calls == {"source": "git+https://example.test/repo@tag"}


def test_doctor_reads_health_and_native_status_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    calls = []

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def health(self):
            calls.append("health")
            return {"ok": True}

        def call(self, name, arguments):
            calls.append((name, arguments))
            return {"version": "test"}

    monkeypatch.setitem(sys.modules, "mcp_for_cascadeur.runner", SimpleNamespace(RunnerClient=FakeRunner))
    assert cli.main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert calls == ["health", ("app_status", {})]


def test_doctor_returns_actionable_json_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))

    class FailingRunner:
        def __init__(self, *args, **kwargs):
            pass

        def health(self):
            raise ConnectionError("Start Cascadeur's script runner.")

    monkeypatch.setitem(sys.modules, "mcp_for_cascadeur.runner", SimpleNamespace(RunnerClient=FailingRunner))
    assert cli.main(["doctor", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"]
    assert "Start Cascadeur" in result["error"]


def test_tools_json_serializes_real_contracts_without_a_running_application(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    assert cli.main(["tools", "--json"]) == 0
    tools = json.loads(capsys.readouterr().out)
    assert len(tools) >= 2
    status = next(tool for tool in tools if tool["name"] == "app_status")
    assert status["read_only"]
    assert status["input_schema"]["additionalProperties"] is False


def test_module_entrypoint_preserves_failed_command_exit_status(tmp_path):
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    environment = dict(os.environ, MCP_FOR_CASCADEUR_HOME=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "mcp_for_cascadeur", "doctor", "--json"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["ok"] is False


@pytest.mark.parametrize(
    ("flags", "expected_source"),
    [
        ([], "git+https://example.test/repo@pinned"),
        (["--source", "explicit-source"], "explicit-source"),
        (["--local-tool"], None),
    ],
)
def test_setup_source_precedence_uses_current_remote_install_by_default(
    flags, expected_source, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path / "settings"))
    monkeypatch.setattr(cli.installation, "detect_cascadeur_home", lambda _: tmp_path / "Cascadeur")
    monkeypatch.setattr(cli.installation, "install_runtime", lambda _: tmp_path / "start_bridge.py")
    monkeypatch.setattr(
        cli.installation, "default_package_source", lambda: "git+https://example.test/repo@pinned"
    )
    captured = []
    monkeypatch.setattr(
        cli.installation, "configure_codex", lambda **options: captured.append(options) or {"changed": True}
    )
    assert cli.main(["setup", "--workspace", str(tmp_path / "workspace"), "--json", *flags]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert captured == [{"source": expected_source}]
