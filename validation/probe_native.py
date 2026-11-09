"""Read installed API facts needed by the native acceptance scenarios."""

from __future__ import annotations

import json

from mcp_for_cascadeur.runner import RunnerClient
from mcp_for_cascadeur.settings import load_config


def main() -> None:
    config = load_config()
    client = RunnerClient(config.endpoint, config.workspace, allow_python=config.allow_python)
    code = """
manager = scene.assets_manager()
assets = []
for asset_id in manager.all_ids():
    asset = manager.at(asset_id)
    assets.append({'id': str(asset_id), 'class': type(asset).__name__,
                   'members': [member for member in dir(asset) if not member.startswith('_')]})
result = {'scene': view.get_path_name(), 'frames': scene.data_viewer().get_animation_size(),
          'objects': len(scene.model_viewer().get_objects()), 'assets': assets,
          'additive_stacks': csc.additive_layers.get_manager(scene).get_stacked_layers_count()}
"""
    print(json.dumps(client.call("python_execute", {"code": code})["result"], indent=2))


if __name__ == "__main__":
    main()
