import json

import pytest

from mcp_for_cascadeur.settings import Config, data_home, load_config, save_config


def test_persistent_configuration_uses_user_data_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "Сцены with spaces"
    expected = Config(workspace=workspace, allow_python=True, cascadeur_home=tmp_path / "Cascadeur")
    target = save_config(expected)
    assert target == data_home() / "config.json"
    assert load_config() == expected
    assert json.loads(target.read_text(encoding="utf-8"))["workspace"] == str(workspace)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8765",
        "http://example.com:8765",
        "http://127.0.0.1",
        "http://user:secret@localhost:8765",
        "http://localhost:8765/run",
        "http://localhost:99999",
    ],
)
def test_endpoint_rejects_remote_or_ambiguous_addresses(endpoint):
    with pytest.raises(ValueError):
        Config(endpoint=endpoint)


def test_absent_settings_do_not_write_any_files(tmp_path):
    target = tmp_path / "absent.json"
    assert load_config(target).endpoint == "http://127.0.0.1:8765"
    assert not target.exists()


def test_non_boolean_python_permission_is_rejected():
    with pytest.raises(ValueError):
        Config(allow_python="false")


def test_persisted_settings_never_include_inherited_cloud_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("UTHANA_API_KEY", "test-fixture-uthana")
    monkeypatch.setenv("GEMINI_API_KEY", "test-fixture-gemini")
    target = save_config(Config(workspace=tmp_path), tmp_path / "settings.json")
    document = target.read_text(encoding="utf-8")
    assert "test-fixture" not in document
    assert not {"api_key", "uthana_api_key", "gemini_api_key"} & json.loads(document).keys()
