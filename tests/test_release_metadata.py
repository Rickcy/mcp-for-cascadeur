"""Distribution metadata must describe the runtime and packaged review client."""

import tomllib
from pathlib import Path

from mcp_for_cascadeur import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_lock_and_runtime_versions_agree():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    locked = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    own_package = next(item for item in locked if item["name"] == project["name"])
    assert project["version"] == own_package["version"] == __version__


def test_review_client_ships_without_the_checkout_generator_or_model_assets():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = metadata["tool"]["hatch"]["build"]["targets"]["wheel"]
    bundled = wheel["force-include"]
    assert {
        "mcp_for_cascadeur/review_backend/motion_review.py",
        "mcp_for_cascadeur/review_backend/visual_provider.py",
    } <= set(bundled.values())
    assert all(Path(source).is_file() for source in (ROOT / item for item in bundled))
    assert not any("pipeline.py" in item or ".local" in item for item in bundled)


def test_animation_dependencies_remain_optional():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    animation = project["optional-dependencies"]["animation"]
    assert any(item.startswith("numpy") for item in animation)
    assert any(item.startswith("imageio-ffmpeg") for item in animation)
    assert not any(item.startswith(("numpy", "imageio-ffmpeg", "torch")) for item in project["dependencies"])
