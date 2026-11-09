"""Read the exact native track-selection interval and item identities."""

import json

from native_advanced import probe

print(
    json.dumps(
        probe("""
selection = scene.get_layers_selector().selection()
interval = selection.frames_interval()
result = {'first': interval.first, 'last': interval.last,
          'selected': [str(item) for item in selection.item_ids()],
          'layers': {str(item): scene.layers_viewer().header(item).name for item in scene.layers_viewer().all_layer_ids()}}
"""),
        indent=2,
    )
)
