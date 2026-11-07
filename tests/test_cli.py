import json
import os
import subprocess
import sys
import warnings
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
    cloud = next(tool for tool in tools if tool["name"] == "uthana_text_start")
    assert cloud["open_world"] is True
    assert cloud["read_only"] is False
    assert {"max_cost_usd", "request_id"} <= set(cloud["input_schema"]["required"])
    assert status["open_world"] is False


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


def test_credential_prompt_hides_value_and_does_not_change_config(tmp_path, monkeypatch, capsys):
    from mcp_for_cascadeur import credentials

    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(cli.getpass, "getpass", lambda _: "test-fixture-hidden")
    monkeypatch.setattr(
        credentials,
        "store_secret",
        lambda provider, value: calls.append((provider, value)) or {"provider": provider, "stored": True},
    )
    assert cli.main(["credentials", "set", "uthana", "--json"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["stored"] is True
    assert "test-fixture-hidden" not in output.out + output.err
    assert calls == [("uthana", "test-fixture-hidden")]
    assert not (tmp_path / "config.json").exists()


def test_credentials_from_env_does_not_prompt_or_echo(tmp_path, monkeypatch, capsys):
    from mcp_for_cascadeur import credentials

    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-fixture-inherited")
    monkeypatch.setattr(cli.getpass, "getpass", lambda _: pytest.fail("must not prompt"))
    values = []
    monkeypatch.setattr(
        credentials,
        "store_secret",
        lambda provider, value: values.append((provider, value)) or {"provider": provider, "stored": True},
    )
    assert cli.main(["credentials", "set", "gemini", "--from-env", "--json"]) == 0
    output = capsys.readouterr()
    assert values == [("gemini", "test-fixture-inherited")]
    assert "test-fixture-inherited" not in output.out + output.err


def test_credentials_refuse_echoing_getpass_fallback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))

    def insecure_prompt(_):
        warnings.warn("Cannot control echo on the terminal.", cli.getpass.GetPassWarning)
        pytest.fail("must not reach fallback input")

    monkeypatch.setattr(cli.getpass, "getpass", insecure_prompt)
    assert cli.main(["credentials", "set", "uthana", "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["code"] == "SECURE_PROMPT_UNAVAILABLE"


def test_credentials_unexpected_failure_redacts_exception(tmp_path, monkeypatch, capsys):
    from mcp_for_cascadeur import credentials

    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    monkeypatch.setattr(
        credentials,
        "credential_status",
        lambda _: (_ for _ in ()).throw(RuntimeError("test-fixture-sensitive-error")),
    )
    assert cli.main(["credentials", "status", "uthana", "--json"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["code"] == "CREDENTIAL_OPERATION_FAILED"
    assert "test-fixture-sensitive-error" not in output.out + output.err


def test_credentials_status_and_delete_do_not_connect_to_native(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    monkeypatch.delenv("UTHANA_API_KEY", raising=False)
    assert cli.main(["credentials", "status", "uthana", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["configured"] is False
    assert cli.main(["credentials", "delete", "uthana", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] is False
