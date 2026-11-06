"""Read the recorded and actual UI state while diagnosing a recovery fence."""

import json

from native_advanced import probe

print(
    json.dumps(
        probe("""
from mcp_for_cascadeur.host import recovery
record = recovery._pending()
result = {'saved': record['view_state'], 'actual': recovery._view_state(view, scene),
          'rollback': record.get('rollback')}
"""),
        indent=2,
    )
)
