"""Read native FK switches and input values while checking a Joint fixture."""

import json

from native_advanced import CONFIG, probe

result = probe("""
from mcp_for_cascadeur.host import animation, common, state
obj = common.resolve_object(scene, 'MCP_FK_Root')
data = scene.data_viewer()
result = {'settings': [{'id':str(i),'name':data.get_setting(i).name,
                        'mode': data.get_setting(i).mode.name,
                        'values':[data.get_setting_value(i,f) if data.get_setting(i).mode.name == 'Animation'
                                  else data.get_setting_value(i) for f in range(6)]}
                       for i in data.get_all_settings_id(obj)],
          'data': [{'id':str(i),'name':data.get_data(i).name,'mode':data.get_data(i).mode.name,
                    'first':state.encode(data.get_data_value(i,0) if data.get_data(i).mode.name == 'Animation' else data.get_data_value(i)),
                    'last':state.encode(data.get_data_value(i,5) if data.get_data(i).mode.name == 'Animation' else data.get_data_value(i))}
                   for i in data.get_all_data_id(obj)]}
""")
path = CONFIG.workspace / "validation" / "fk-fixture-probe.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(path, flush=True)
