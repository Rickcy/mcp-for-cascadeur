"""Measure and inspect a full model checkpoint without modifying the scene."""

import json

from mcp_for_cascadeur.installation import install_runtime
from mcp_for_cascadeur.runner import RunnerClient
from mcp_for_cascadeur.settings import load_config


def main():
    config = load_config()
    install_runtime(config)
    client = RunnerClient(config.endpoint, config.workspace, allow_python=config.allow_python)
    code = """
import time
from mcp_for_cascadeur.host import state
started = time.monotonic()
checkpoint = state.capture(scene)
result = {'frames': checkpoint['count'], 'objects': len(checkpoint['objects']),
          'channels': len(checkpoint['channels']), 'tracks': len(checkpoint['tracks']),
          'bytes': len(state.canonical_bytes(checkpoint)), 'sha256': state.digest(checkpoint),
          'seconds': time.monotonic() - started,
          'self_compare': state.compare(checkpoint, state.capture(scene)),
          'scene_selectors': [name for name in dir(scene) if 'select' in name],
          'external': checkpoint['external']}
"""
    print(json.dumps(client.call("python_execute", {"code": code})["result"], indent=2))


if __name__ == "__main__":
    main()
