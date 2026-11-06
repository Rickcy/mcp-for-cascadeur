import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomlkit

from mcp_for_cascadeur import installation
from mcp_for_cascadeur.settings import Config


@pytest.fixture
def isolated_installation(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path / "persistent"))
    home = tmp_path / "Cascadeur"
    scripts = home / "resources/scripts/python"
    (scripts / "scripts/mcp/script_server").mkdir(parents=True)
    (scripts / "scripts/mcp/script_server/server.py").write_text("# vendor sentinel", encoding="utf-8")
    (scripts / "events/application_started").mkdir(parents=True)
    return home, scripts


def test_runtime_is_standalone_after_source_location_disappears(isolated_installation, tmp_path, monkeypatch):
    home, scripts = isolated_installation
    package = tmp_path / "temporary-tool" / "mcp_for_cascadeur"
    (package / "host").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "contracts.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "host/__init__.py").write_text(
        "import json\ndef start_bridge(**options):\n    print(json.dumps(options))\n", encoding="utf-8"
    )
    (package / "installation.py").touch()
    monkeypatch.setattr(installation, "__file__", str(package / "installation.py"))
    config = Config(cascadeur_home=home, workspace=tmp_path / "workspace")
    bridge = installation.install_runtime(config)
    runtime = bridge.parent / "runtime/python/mcp_for_cascadeur"
    assert (runtime / "host/__init__.py").is_file()
    assert (runtime / "contracts.py").is_file()
    assert not (runtime / "installation.py").exists()
    assert str(package.parent) not in bridge.read_text(encoding="utf-8")
    assert not list(runtime.rglob("server.py"))
    assert (scripts / "scripts/mcp/script_server/server.py").read_text() == "# vendor sentinel"
    compile(bridge.read_text(encoding="utf-8"), str(bridge), "exec")
    moved_source = tmp_path / "discarded-tool-location"
    assert package.parent.resolve().is_relative_to(tmp_path.resolve())
    assert moved_source.resolve().is_relative_to(tmp_path.resolve())
    package.parent.rename(moved_source)
    result = subprocess.run([sys.executable, str(bridge)], capture_output=True, text=True, check=True)
    assert str(config.workspace).replace("\\", "\\\\") in result.stdout


def test_startup_preserves_unrelated_and_modified_hooks(isolated_installation, tmp_path):
    home, scripts = isolated_installation
    config = Config(cascadeur_home=home)
    target = scripts / "events/application_started" / installation.HOOK_NAME
    target.write_text("unrelated()\n", encoding="utf-8")
    with pytest.raises(installation.InstallationError, match="unrelated"):
        installation.install_startup(config, tmp_path / "bridge.py")
    assert target.read_text() == "unrelated()\n"
    target.unlink()
    installation.install_startup(config, tmp_path / "bridge.py")
    original = target.read_text()
    installation.install_startup(config, tmp_path / "bridge.py")
    assert target.read_text() == original
    target.write_text(original + "# changed\n", encoding="utf-8")
    with pytest.raises(installation.InstallationError, match="modified"):
        installation.uninstall_startup(home)
    assert target.exists()


def test_startup_hook_uses_deferred_native_main_thread(isolated_installation, tmp_path):
    home, scripts = isolated_installation
    bridge = tmp_path / "bridge with spaces.py"
    installation.install_startup(Config(cascadeur_home=home), bridge)
    target = scripts / "events/application_started" / installation.HOOK_NAME
    text = target.read_text()
    assert "scene_idle_manager.execute_when_idle(_start_bridge)" in text
    compile(text, str(target), "exec")
    assert installation.uninstall_startup(home)["removed"]
    assert not target.exists()


def test_codex_edits_preserve_other_servers_comments_and_backup(tmp_path):
    target = tmp_path / "config.toml"
    original = '# preserve this comment\nmodel = "example"\n[mcp_servers.blender]\ncommand = "uvx"\nargs = ["mcp-for-blender"]\n'
    target.write_text(original, encoding="utf-8")
    result = installation.configure_codex(target)
    assert Path(result["backup"]).read_text(encoding="utf-8") == original
    content = target.read_text(encoding="utf-8")
    parsed = tomlkit.parse(content)
    assert parsed["model"] == "example"
    assert "# preserve this comment" in content
    assert parsed["mcp_servers"]["blender"]["args"] == ["mcp-for-blender"]
    assert parsed["mcp_servers"][installation.CODEX_SERVER_NAME]["args"] == [
        "--from",
        installation.DEFAULT_SOURCE,
        "mcp-for-cascadeur",
    ]
    assert installation.configure_codex(target)["changed"] is False
    installation.configure_codex(target, remove=True)
    assert "blender" in tomlkit.parse(target.read_text())["mcp_servers"]


def test_codex_conflicting_command_is_not_replaced(tmp_path):
    target = tmp_path / "config.toml"
    original = '[mcp_servers.mcp_for_cascadeur]\ncommand = "something-else"\n'
    target.write_text(original, encoding="utf-8")
    with pytest.raises(installation.InstallationError, match="already used"):
        installation.configure_codex(target)
    assert target.read_text() == original
    with pytest.raises(installation.InstallationError, match="preserved"):
        installation.configure_codex(target, remove=True)
    assert target.read_text() == original


def test_codex_records_custom_settings_home_without_removing_other_environment(tmp_path, monkeypatch):
    selected_home = tmp_path / "settings"
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(selected_home))
    target = tmp_path / "codex.toml"
    target.write_text(
        '[mcp_servers.mcp_for_cascadeur]\ncommand = "uvx"\nargs = ["mcp-for-cascadeur"]\n[mcp_servers.mcp_for_cascadeur.env]\nEXISTING = "keep"\n',
        encoding="utf-8",
    )
    installation.configure_codex(target, source=None)
    environment = tomlkit.parse(target.read_text())["mcp_servers"][installation.CODEX_SERVER_NAME]["env"]
    assert environment["EXISTING"] == "keep"
    assert environment["MCP_FOR_CASCADEUR_HOME"] == str(selected_home)


def test_explicit_detection_rejects_installation_without_vendor_runner(tmp_path):
    with pytest.raises(installation.InstallationError, match="not found"):
        installation.detect_cascadeur_home(tmp_path)


def test_codex_uses_absolute_uvx_and_recognizes_it_on_repeat(tmp_path, monkeypatch):
    executable = str(tmp_path / "uvx.exe")
    monkeypatch.setattr(installation.shutil, "which", lambda _: executable)
    target = tmp_path / "codex.toml"
    installation.configure_codex(target, source=None)
    entry = tomlkit.parse(target.read_text())["mcp_servers"][installation.CODEX_SERVER_NAME]
    assert entry["command"] == executable
    assert not installation.configure_codex(target, source=None)["changed"]
    assert installation.configure_codex(target, remove=True)["changed"]


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {
                "url": "https://github.com/Rickcy/mcp-for-cascadeur",
                "vcs_info": {
                    "vcs": "git",
                    "requested_revision": "feature-branch",
                    "commit_id": "a" * 40,
                },
            },
            "git+https://github.com/Rickcy/mcp-for-cascadeur@" + "a" * 40,
        ),
        (
            {"url": "https://example.test/mcp_for_cascadeur-0.1.0-py3-none-any.whl", "archive_info": {}},
            "https://example.test/mcp_for_cascadeur-0.1.0-py3-none-any.whl",
        ),
        ({"url": "file:///C:/uv/cache/package.whl", "archive_info": {}}, installation.DEFAULT_SOURCE),
        ({"url": "file:///C:/checkout", "dir_info": {"editable": True}}, installation.DEFAULT_SOURCE),
        (
            {"url": "https://example.test/repo", "vcs_info": {"vcs": "git", "commit_id": "main"}},
            installation.DEFAULT_SOURCE,
        ),
        ([], installation.DEFAULT_SOURCE),
    ],
)
def test_remote_package_source_is_reproducible_and_never_uses_local_cache(record, expected, monkeypatch):
    import json

    def distribution(name):
        assert name == "mcp-for-cascadeur"
        return SimpleNamespace(
            read_text=lambda filename: json.dumps(record) if filename == "direct_url.json" else None
        )

    monkeypatch.setattr(installation.metadata, "distribution", distribution)
    assert installation.default_package_source() == expected


@pytest.mark.parametrize("record", [None, "{not json"])
def test_absent_or_invalid_direct_url_falls_back_to_public_source(record, monkeypatch):
    monkeypatch.setattr(
        installation.metadata, "distribution", lambda _: SimpleNamespace(read_text=lambda _: record)
    )
    assert installation.default_package_source() == installation.DEFAULT_SOURCE


def test_uninstalled_distribution_falls_back_to_public_source(monkeypatch):
    def missing(_):
        raise installation.metadata.PackageNotFoundError("mcp-for-cascadeur")

    monkeypatch.setattr(installation.metadata, "distribution", missing)
    assert installation.default_package_source() == installation.DEFAULT_SOURCE
