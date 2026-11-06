"""Check native Euler winding preservation through checkpoint encode/decode."""

import json

from native_advanced import CONFIG, probe

result = probe("""
import math
from mcp_for_cascadeur.host import state, animation
original = csc.math.Rotation.from_euler(2 * math.pi + 0.25, 0.1, -0.2)
encoded = state.encode(original)
restored = state.encode(state.decode(encoded))
assert abs(encoded['euler_radians'][0] - (2 * math.pi + 0.25)) < 1e-6
assert state.equivalent_values(encoded, restored)
copied = state.encode(animation._copy_native(csc, original))
assert state.equivalent_values(encoded, copied)
result = {'original': encoded, 'restored': restored, 'copied': copied, 'full_turn_preserved': True}
""")
(CONFIG.workspace / "validation" / "native-euler-fidelity.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result), flush=True)
