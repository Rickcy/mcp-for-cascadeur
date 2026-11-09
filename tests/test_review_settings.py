import json
import os
from pathlib import Path

import pytest

from mcp_for_cascadeur import cli
from mcp_for_cascadeur import review_settings as settings


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path / "private"))
    for name in (*settings.ENVIRONMENT.values(), "VISUAL_REVIEW_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cipher(monkeypatch):
    def transform(data, *, decrypt=False):
        return bytes(value ^ 0x5A for value in data)
    monkeypatch.setattr(settings, "_dpapi", transform)


def test_default_status_and_configuration_never_return_a_key():
    result = settings.configure_review(provider="gemini", model="gemini-flash-latest")
    assert result["provider"] == "gemini"
    assert result["key_configured"] is False
    assert "api_key" not in result
    assert settings.load_review_config() == {"provider": "gemini", "model": "gemini-flash-latest"}


def test_key_is_encrypted_and_resolved_without_plaintext_configuration(cipher):
    key = "development-fixture-credential"
    settings.set_review_key("gemini", key)
    assert settings.resolve_review_config()["api_key"] == key
    for path in settings.data_home().rglob("*"):
        if path.is_file():
            assert key.encode() not in path.read_bytes()
    assert key not in json.dumps(settings.public_review_status())


def test_environment_has_priority_and_provider_keys_do_not_cross(cipher, monkeypatch):
    settings.set_review_key("gemini", "stored-cloud-key")
    settings.configure_review(provider="gemini", model="gemini-flash-latest")
    monkeypatch.setenv("GEMINI_API_KEY", "environment-cloud-key")
    assert settings.resolve_review_config()["api_key"] == "environment-cloud-key"
    local = settings.resolve_review_config(provider="openai-compatible")
    assert local["api_key"] is None and local.get("model") is None
    monkeypatch.setenv("VISUAL_REVIEW_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    monkeypatch.setenv("VISUAL_REVIEW_API_KEY", "explicit-local-server-key")
    local = settings.resolve_review_config(provider="openai-compatible")
    assert local["model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert local["api_key"] == "explicit-local-server-key"
    assert local["api_key_provider"] == "openai-compatible"


def test_provider_switch_discards_incompatible_saved_model():
    settings.configure_review(provider="gemini", model="gemini-flash-latest")
    with pytest.raises(ValueError, match="exact locally served model"):
        settings.configure_review(provider="openai-compatible")
    settings.configure_review(provider="openai-compatible", model="Qwen3-VL", media_transport="frames")
    assert settings.resolve_review_config()["model"] == "Qwen3-VL"


def test_credentials_cannot_be_saved_in_git_checkout(cipher, tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path / "private"))
    with pytest.raises(ValueError, match="outside every Git checkout"):
        settings.set_review_key("gemini", "fixture-key")
    assert not settings.data_home().exists()


@pytest.mark.parametrize("values", [
    {"api_key": "never-a-config-field"}, {"base_url": "https://remote.test/v1"},
    {"base_url": "http://user:secret@localhost:8000/v1"}, {"max_total_frames": True},
    {"sampling_fps": float("nan")}, {"frame_width": 1}, {"media_transport": "still-only"},
])
def test_invalid_configuration_fails_without_persisting(values):
    with pytest.raises(ValueError):
        settings.configure_review(**values)
    assert not settings.review_config_path().exists()


def test_delete_only_selected_provider_key(cipher):
    settings.set_review_key("gemini", "cloud-fixture")
    settings.set_review_key("openai-compatible", "local-fixture")
    assert settings.delete_review_key("gemini")["removed"]
    assert settings.resolve_review_config()["api_key"] is None
    assert settings.resolve_review_config(provider="openai-compatible")["api_key"] == "local-fixture"


def test_cli_hidden_input_and_status_do_not_print_secret(cipher, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "hidden-cli-fixture")
    assert cli.main(["review", "key", "set", "gemini", "--json"]) == 0
    assert "hidden-cli-fixture" not in capsys.readouterr().out
    assert cli.main(["review", "status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["key_configured"]


def test_cli_reads_key_from_named_environment_variable(cipher, monkeypatch, capsys):
    monkeypatch.setenv("REVIEW_TEST_KEY", "noninteractive-fixture")
    assert cli.main(["review", "key", "set", "gemini", "--from-env", "REVIEW_TEST_KEY", "--json"]) == 0
    assert "noninteractive-fixture" not in capsys.readouterr().out
    assert settings.resolve_review_config()["api_key"] == "noninteractive-fixture"


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI")
def test_windows_current_user_dpapi_roundtrip():
    secret = b"dpapi-isolated-fixture"
    encrypted = settings._dpapi(secret)
    assert secret not in encrypted
    assert settings._dpapi(encrypted, decrypt=True) == secret


def test_local_secret_files_are_not_provider_responses(cipher):
    settings.set_review_key("gemini", "private-fixture")
    path = Path(settings.public_review_status()["config_path"])
    assert path.name == "visual-review.json"
    assert "private-fixture" not in json.dumps(settings.public_review_status())
