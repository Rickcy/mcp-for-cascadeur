import json
import sys

import pytest

from mcp_for_cascadeur import credentials


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    for variable in credentials.ENVIRONMENT.values():
        monkeypatch.delenv(variable, raising=False)


def test_encrypted_store_roundtrip_contains_no_plaintext(tmp_path, monkeypatch):
    operations = []

    def fake_dpapi(data, provider, *, decrypt):
        operations.append((provider, decrypt))
        return data.removeprefix(b"encrypted:")[::-1] if decrypt else b"encrypted:" + data[::-1]

    monkeypatch.setattr(credentials, "_dpapi", fake_dpapi)
    result = credentials.store_secret("uthana", "test-fixture-api-key")
    assert result == {"provider": "uthana", "stored": True, "protection": "windows-dpapi-current-user"}
    path = tmp_path / "credentials" / "uthana.dpapi.json"
    assert "test-fixture-api-key" not in path.read_text()
    assert credentials.load_secret("uthana") == "test-fixture-api-key"
    assert operations == [("uthana", False), ("uthana", True)]
    assert list(path.parent.iterdir()) == [path]


def test_environment_takes_precedence_without_decrypting(monkeypatch):
    monkeypatch.setenv("UTHANA_API_KEY", "test-fixture-from-env")
    monkeypatch.setattr(credentials, "_dpapi", lambda *a, **kw: pytest.fail("must not decrypt"))
    assert credentials.load_secret() == "test-fixture-from-env"
    status = credentials.credential_status("uthana")
    assert status["source"] == "environment" and status["configured"]
    assert "test-fixture-from-env" not in json.dumps(status)


def test_missing_credential_has_actionable_secret_free_error():
    with pytest.raises(credentials.CredentialError, match="credentials set uthana") as error:
        credentials.load_secret("uthana")
    assert error.value.code == "CREDENTIAL_REQUIRED"
    assert error.value.execution_state == "not_started"


def test_non_windows_store_does_not_fall_back_to_plaintext(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "linux")
    with pytest.raises(credentials.CredentialError) as error:
        credentials.store_secret("gemini", "test-fixture-api-key")
    assert error.value.code == "SECURE_STORE_UNAVAILABLE"
    assert not (tmp_path / "credentials").exists()


@pytest.mark.parametrize("value", ["", " ", "key\nvalue", "padding ", "\x00key", "x" * 17000])
def test_invalid_credential_is_rejected_before_storage(value, monkeypatch):
    monkeypatch.setattr(credentials, "_dpapi", lambda *a, **kw: pytest.fail("must not encrypt"))
    with pytest.raises(credentials.CredentialError):
        credentials.store_secret("uthana", value)


def test_corrupt_record_is_rejected_without_echoing_contents(tmp_path):
    directory = tmp_path / "credentials"
    directory.mkdir()
    (directory / "uthana.dpapi.json").write_text('"fixture-hidden-corruption"')
    with pytest.raises(credentials.CredentialError) as error:
        credentials.load_secret("uthana")
    assert error.value.code == "INVALID_CREDENTIAL_STORE"
    assert "fixture-hidden-corruption" not in str(error.value)


def test_status_and_delete_do_not_decrypt_and_preserve_environment(tmp_path, monkeypatch):
    directory = tmp_path / "credentials"
    directory.mkdir()
    target = directory / "gemini.dpapi.json"
    target.write_text("opaque encrypted record")
    monkeypatch.setattr(credentials, "_dpapi", lambda *a, **kw: pytest.fail("must not decrypt"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-fixture-env")
    assert credentials.credential_status("gemini")["stored"]
    assert credentials.delete_secret("gemini")["deleted"]
    assert not target.exists()
    assert credentials.load_secret("gemini") == "test-fixture-env"


def test_unknown_provider_cannot_escape_store(tmp_path):
    with pytest.raises(credentials.CredentialError) as error:
        credentials.delete_secret("../config")
    assert error.value.code == "UNKNOWN_CREDENTIAL"
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI API")
def test_windows_dpapi_current_user_roundtrip(tmp_path):
    credentials.store_secret("uthana", "test-fixture-only-never-a-live-key")
    assert credentials.load_secret("uthana") == "test-fixture-only-never-a-live-key"
    assert "test-fixture-only" not in (tmp_path / "credentials" / "uthana.dpapi.json").read_text()
