# Native Cascadeur API research

Research date: 2026-11-06. The inspected installation is Cascadeur
`2026.2.3.0.16863` on Windows. These notes describe installed vendor interfaces;
an API declaration is not evidence that an artistic operation succeeded.
No vendor implementation or character assets are included in this repository.

## Built-in script server

Cascadeur already includes a local script server. The supported user entry point
is **Scripts > MCP > Start script server**; **Stop script server** stops it.
The vendor documents these commands in the
[Scripts Menu](https://cascadeur.com/help/category/320).

Installed implementation reference, relative to the Cascadeur installation:

`resources/scripts/python/scripts/mcp/script_server/server.py`

Observed interface in this installation:

| Interface | Behaviour |
| --- | --- |
| Bind address | `127.0.0.1`, default port `8765` |
| Server version | `0.3.1` |
| `GET /health` | Returns `ok`, `version`, idle/message subscription IDs and queued request count. |
| `POST /run` | JSON object with a `code` string; returns the script result described below. |
| `POST /mcp` | JSON-RPC initialize, tools/list, tools/call, resources/list and resources/read. |
| Native MCP tool | `run_script`, taking a `code` string. |
| Native resources | `cascadeur://docs/python-api`, `cascadeur://modules`. |

The declared MCP protocol version is `2024-11-05`. The HTTP interface has no
authentication in the inspected implementation and starts only when requested.
Keep the default loopback address. Our stdio process can use this installed
interface without copying or replacing the server implementation.

The server queues work from its HTTP thread and executes it from
`events.scene_idle_manager` inside Cascadeur. Its execution namespace supplies
`csc`, `scene` (a domain scene) and `app`. This avoids calling native scene
interfaces from an external process or an HTTP worker thread.

An expression produces `{"ok": true, "value": "<Python repr>", "messages": []}`.
A sequence of statements produces an empty `value` string. Exceptions produce
`ok: false` and an `error` traceback. Messages contain `level` and `text` fields.
The server does not capture normal stdout in this response.

An independently authored adapter should serialize a structured result as JSON
inside an expression, unwrap the outer Python string representation with
`ast.literal_eval`, then decode the JSON with `json.loads`. Never use `eval` on
the returned value. Validate the result shape and propagate both errors and native
error-level messages.

The native queue waits up to 30 seconds. A request not yet started is marked
cancelled on timeout. This does not interrupt a request that is already executing:
a transport timeout can therefore have an unknown write outcome. Inspect scene
state before retrying a write. Keep individual native batches bounded.

The installed menu module is `scripts.mcp.start_server`, whose `run(scene)`
function starts `scripts.mcp.script_server.server`. The corresponding stop module
is `scripts.mcp.stop_server`. For normal use, use the menu rather than editing the
application installation.

## Native access points for structured tools

Installed declarations live under `resources/scripts/stubs/csc/`.
The [Python API overview](https://cascadeur.com/help/category/215) distinguishes
the application and domain-scene levels. Prefer declarations shipped with the
tested application over assumptions from an older web example.

| Area | Installed declarations and relevant interfaces |
| --- | --- |
| Application and scenes | `app/__init__.pyi`: `get_application()`, `Application.get_scene_manager()`, `SceneManager.scenes()`, `current_scene()`, `create_application_scene()`, `set_current_scene()`. |
| Loading | `app/__init__.pyi`: `DataSourceManager.load_scene(path)` loads a full application scene; `ProjectLoader.load_from(path, domain_scene)` is documented as a minimal load that omits auxiliary UI data. |
| Scene save | `view/__init__.pyi`: `Scene.save(path)`, `get_path_name()`, `domain_scene()`. Verify an actual output and reopening, not only the return from a void API. |
| Object discovery | `model.pyi`: `ModelViewer.get_objects()`, `get_object_name()`, `BehaviourViewer.get_behaviour_by_name()`, `get_behaviour_data()`. Resolve actual IDs and channels before writes. |
| Frame reads | `model.pyi`: `DataViewer.get_data_value(data_id, frame)`; `domain/__init__.pyi`: `Scene.get_current_frame()`, `set_current_frame(frame)`. |
| Undoable edits | `domain/__init__.pyi`: `Scene.modify_update(label, callback) -> bool`; callback receives model editor, update editor and scene updater. Check the boolean and independently read back the result. |
| Data writes | `model.pyi`: `DataEditor.set_data_value(data_id, frame, value)` and overloads; `domain/__init__.pyi`: `SceneUpdater.run_update(data_ids, frame)` and `generate_update()`. Rig updates may change coupled controls. |
| Timeline allocation | `model.pyi`: `ModelEditor.fit_animation_size_by_layers()` is present in this installation. Native animation allocation and key-section creation are distinct operations. |
| Track keys | `layers/__init__.pyi`: `Editor.set_section()`, `set_sections()`, `Layer.sections()`, `key_frame_indices()`, `Viewer.frames_count()`, layer-object mapping. Track layers are distinct from additive layers. |
| Interpolation | `layers/layer.pyi`: `Section.key`, `Section.interval`; interpolation includes BEZIER, LOW_AMPLITUDE_BEZIER, LINEAR, STEP, FIXED, NONE, CLAMPED_BEZIER and AI. |
| Easing and tangents | `layers/layer.pyi`: key left/right velocity weights, tangent mode, key and interval common IK/FK/fixation state, AI interpolation description. Preserve untouched metadata rather than replacing a whole section with defaults. |
| Playback boundaries | `view/__init__.pyi`: `AnimationBoundary.first_frame`, `last_frame`, `first_visible_frame`, `last_visible_frame`; `Scene.animation_boundary()`. Playback and visible bounds are separate. |
| Images | `tools/__init__.pyi`: `RenderToFile.take_image(scene_view, render_parameters, filename)`. Discover actual parameter fields and verify a fresh PNG on the tested build. |
| FBX | `fbx.pyi`: `FbxSceneLoader.get_fbx_loader(scene_view)` and `FbxLoader` import/export methods; `Application.is_export_available()` and `is_pro_features_available()` expose license status. |
| Menu actions | `Application.get_action_manager().call_action(id)` dispatches an action; dispatch alone does not verify completion. See the official [action ID list](https://cascadeur.com/help/category/301). |

The installed `tools/auto_posing.pyi` and `tools/auto_interpolation.pyi` do not
declare useful members. This is a discovery gap, not proof the live tools have no
API. Query the running tool/editor and record actual signatures before exposing
an operation. Do not advertise physics, AI interpolation, retargeting or mocap
merely because a menu item or enum exists.

## Events and startup

Installed event infrastructure:

- `resources/scripts/python/events_rule.py` discovers event handler modules with
  a `run` function.
- `events/application_started/application_started_manager.py` dispatches an
  application-start event without scene arguments.
- `events/scene_idle/scene_idle_manager.py` dispatches scene-idle events.
- `events/idle/idle_manager.py` supplies
  `scene_idle_manager.execute_when_idle(callback)` for a one-time callback.
- `events/event_message/event_message_manager.py` supplies native message events.

These are useful for an explicitly managed test bootstrap. A bootstrap can defer
initialization until the first idle event, start the installed server and report
success to a temporary test directory. It should be removable and must not
overwrite an existing user handler. Normal runtime can use the built-in menu.

No supported headless or command-line script execution route was established by
this inspection. Do not describe a normal GUI process launched with a hidden
window as headless. Do not guess CLI switches.

## Focused native acceptance plan

1. Start Cascadeur normally and activate its built-in script server. Verify
   `/health` and a read-only expression returning application/scene information.
2. Create a private test workspace. Use a disposable copy of the installed
   `samples/Cascy.casc` and any needed texture directory. Do not modify the
   installed sample. Capture source hashes before and after the test.
3. Record actual app/API/Python versions, scene path, object types, frame count,
   track keys/interpolation and relevant license flags.
4. Resolve one writable Point or Box control using native behaviour/data IDs.
   Read its exact values at existing frames before constructing a small edit.
5. Create a few explicit keys, allocate animation data, make a bounded native
   transaction, then read the edited frame and inbetween frames. Check finite
   values and resulting rig state; successful dispatch is insufficient.
6. Capture a fresh PNG for visual review, save to a new `.casc`, reopen it and
   compare poses and timeline metadata. Exercise FBX only when entitlement allows.
7. Exercise native Undo for the specific edited operation and verify restoration.
   Record uncertain recovery rather than declaring rollback successful.
8. Close only test-owned scenes/processes normally and observe process exit.
   Remove only the temporary bootstrap created by the test, if one was used.

Keep the measured result separate from API discovery. Expand tests only for the
feature being implemented and its immediate integrations.
