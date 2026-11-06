"""Summarize mismatching native channels without printing full checkpoint data."""

import json

from native_advanced import probe

print(
    json.dumps(
        probe("""
from mcp_for_cascadeur.host import recovery, state
record = recovery._pending()
actual = state.capture(scene)
expected = record['checkpoint']
comparison = state.compare(expected, actual)
def deviation(first, second):
    if isinstance(first, dict) and isinstance(second, dict):
        return max((deviation(first[k], second[k]) for k in first.keys() & second.keys()), default=0)
    if isinstance(first, list) and isinstance(second, list):
        return max((deviation(a, b) for a, b in zip(first, second)), default=0)
    if type(first) in (int, float) and type(second) in (int, float):
        return abs(first - second)
    return 0 if first == second else float('inf')
channels = []
for key, item in expected['channels'].items():
    observed = actual['channels'][key]
    if state.channel_equal(item, observed):
        continue
    channels.append({'key': key, 'object': expected['objects'][item['owner']]['name'],
                     'name': item['name'], 'kind': item['kind'], 'mode': item['mode'],
                     'max_component_delta': deviation(item, observed),
                     'frames': [frame for frame, (a,b) in enumerate(zip(item.get('values', []), observed.get('values', [])))
                                if not state.equivalent_values(a,b,exact=item['kind']=='setting')]})
result = {'request_id': record['request_id'], 'difference_count': comparison['difference_count'],
          'channels': channels[:8], 'view': recovery._view_comparison(recovery._view_state(view, scene), record['view_state'])}
"""),
        indent=2,
    )
)
