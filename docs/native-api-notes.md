# Native Cascadeur API research

Initial inspection was on 2026-11-06, with live workflow checks on 2026-11-07.
The inspected installation is Cascadeur
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

## Live native workflow checks

The following checks ran on Windows Cascadeur 2026.2.3, build 44356, embedded
Python 3.11 and Qt 6.5.3. Export entitlement was available and the Pro feature
flag was false. Tests used copies of the installed Cascy sample under
`validation/native-tools-20261107-105323` in the configured MCP workspace.

| Operation | Exercised route | Observed result |
| --- | --- | --- |
| Motion Generation | `Application.get_action_manager().call_action("View.Inbetweening_RunMotionGeneration")` | Two full Point poses at frames 0 and 30, translated 100 scene units along Z, produced a walking motion. Intermediate foot and pelvis positions changed and body intervals became FIXED. Saved as `motion-generated.casc`. |
| Reference video | `View.Bind video`, the native file picker, then `LoadVideoDialog.viewModel.onDecompressClicked()` through QML | Imported a generated H.264 MP4 at 30 FPS. Explicit `from=0` and `to=30` produced 31 reference images and a new native plane. |
| Mocap | `View.MocapTool`, queued confirmation and native processing | All 31 selected frames were processed. Native log timestamps give 18.378 seconds between recognition start and finish. Body tracks received keys at every processed frame. Saved as `mocap-result.casc`. A separately reopened copy had zero coordinate difference for pelvis and both feet at frames 0, 15 and 30. |
| Animation Unbaking | `View.Animation unbaking` | Converted the FIXED clip to classic intervals, including CLAMPED_BEZIER. Body tracks contained 11 keys including the terminal frame. Maximum position deviation at four sampled controls on seven frames was 0.907891 scene units. Saved as `unbaked.casc`. |
| AutoPhysics | `AutoPhysicsTool.Switch Auto Physics`, followed by `AutoPhysicsTool.Snap to Auto Physics` and the native **All Frames** choice | Solver convergence was logged. After confirming the apply dialog, sampled controls changed across frames 0–30, by up to 53.810725 scene units. Maximum native connection-anchor error was 0.013754 across 32 stored frames. Saved as `physics-applied.casc`. |
| Review video | Separate `timeline_set_frame` and `viewport_capture` requests, followed by ffmpeg encoding | Captured 31 fresh PNG frames and produced `generated-walk.mp4`, 960 × 720, 30 FPS. Visual checks at different frames confirmed the pose changed with the frame. |

These are technical workflow checks on a synthetic Cascy reference. Artistic
quality, preservation of the requested path after physics, and performance on
real human footage need their own acceptance cases. In particular, the measured
AutoPhysics changes require comparison with the requested motion before choosing
that candidate for delivery.

### Structured jobs

The adapter exposes `motion_generation_start`, `mocap_start`, `unbaking_start`
and `autophysics_start`. Their native driver is restricted to Windows build
44356 (2026.2.3), validated Cascy ownership/semantics and at most 512 stored
frames, including a following terminal guard. The routes above informed this
implementation; current structured-job acceptance is recorded separately in
[validation.md](validation.md). Generic rig support is not established.

Motion Generation consumes keyed poses and a trajectory through the installed
tool. It is not a free-text generation API. Mocap uses a workspace MP4 or an
existing reference plane; model availability is checked before dispatch, and
the optional native download requires explicit `allow_model_download=true`.

`host/jobs.py` persists a request receipt and a hashed native source copy before
starting work on a separate working scene. A Qt timer advances the JSON-only
plan through bounded stages; native callbacks are queued on the application
thread. `native_job_status` exposes phases, observed progress, native events and
the result. Unmeasured progress remains indeterminate. `native_jobs_list` and
status have a checksummed journal fallback in the external MCP server when the
native process or transport cannot answer. This fallback observes saved state
without executing or replaying a native action.

The job retains its write fence through native completion and verification.
Protected-state checks compare the full checkpoint: existing object/channel
identity, static values, unrelated animation, out-of-interval samples and
auxiliary/asset metadata. Mocap may add only its identified video reference
plane, that plane's channels/track and its Mesh/MeshDependency assets. A fresh
output `.casc` and its hash are published after the scoped result is checked.

Cancellation is requested through `native_job_cancel`. Dialog closure alone
does not establish worker quiescence. If a safe stop cannot be verified, the
scene remains fenced; restart Cascadeur and use `native_job_recover` to reopen
the hashed source and verify the complete model before allowing new writes.
Timeouts do not trigger automatic replay.

CLAMPED_BEZIER is supported as native `Interpolation(6)`. The installed `Interval`
has no extra serialized clamping field: its enum, common IK/FK/fixation state,
description and the key's label/tangents/easing weights are retained exactly.
Both retimers and manual checkpoint restoration resolve saved enum names before
changing tracks. Existing format-1 checkpoints remain readable; an unavailable
enum is rejected rather than substituted. Custom spatial tangents are still
outside checked restoration. Character pose/curve edits remain bounded to 121
stored frames, and character retiming to 2–8 synchronized keys; the native-job
frame budget does not expand these separate contracts.

Unbaking may add a predeclared key at the existing terminal guard when that
track has no key there. Its descriptor copies the preceding key's metadata with
a STEP interval and empty AI description. Only this exact metadata addition is
allowed; every terminal-frame data and setting sample remains protected.

`examples/native_job_smoke.py` opens a fresh copy of a prepared scene, starts one
operation and polls its durable status into a JSON acceptance report. Technical
completion does not prove realistic weight, good contacts or the desired style.

### Qt and QML access

The live Mocap editor exposed an empty `SceneTool` through the Python binding.
The Inbetweening editor exposed `get_update_parameter` and
`set_update_parameter`. The actual UI routes above reached additional behavior.

Installed Qt resources can be read with `QFile`. The relevant resources include
`:/MocapTool/MocapTool.qml`, `:/MocapTool/MocapDialog.qml`,
`:/InbetweeningTool/InbetweeningTool.qml` and `:/Main/LoadVideoDialog.qml`.
The Motion Generation button in this build uses
`View.Inbetweening_RunMotionGeneration`. The public action reference lists
`View.MotionGeneration_Run`, so route discovery must record the installed build.

Visual items were reachable through a `QQuickWindow`'s `contentItem()` and
`childItems()`. `QQmlEngine.contextForObject(item)` and `QQmlExpression` allowed
reading Mocap `viewModel.state`, `received` and `total` as a JSON string. Direct
Python conversion of the custom `MocapViewModel*` property raised a converter
error. The same QML route exposed video import progress, timing and its
`onDecompressClicked` method. The video model's `sourceFile` property is
read-only, and the tested file-selection step used the native file picker.

Confirmed model data came from `:/MocapTool/models/info.json`. The package was
installed at `%LOCALAPPDATA%/Nekki Limited/Cascadeur/resources/mocap.pt`, with
825,323,915 bytes and SHA-256
`a151cd3a69f550efc28015ab8b7e62dd1c3b2b581dd0e1edb26c4b31e0690888`.
Its official source is
[the Cascadeur CDN](https://cdn.cascadeur.com/animations/files/17/mocap_010725.pt).
An interrupted download was resumed and checked against the embedded size and
hash before the successful Mocap run.

### Lifetime, asynchronous execution and recovery

Two crashes occurred during these probes. A direct Qt download-confirmation
probe terminated PID 34956 in `Qt6Core.dll` with access violation `0xc0000005`.
That probe also accessed button metadata after invoking the callback. A later
`RenderToFile.play_to_video_file` probe terminated PID 68556 in
`presenter_lib.dll` with the same exception code. Windows application events
confirmed both failures. Their exact native causes require stack analysis.
Subsequent confirmation probes collected metadata first and used
`QMetaObject.invokeMethod(..., Qt.QueuedConnection)` successfully.

Both crashes left pending durable receipts. Cascadeur was restarted with the
saved source scene. Recovery compared the complete native model with its
recorded checkpoint, reported zero differences and verified restoration before
clearing the write fence. The successful generation request also exceeded the
vendor runner's 30-second deadline. Its receipt and later scene readback resolved
the outcome as committed.

The first physics dispatches opened **Apply AutoPhysics** dialogs while the
native model stayed unchanged. Application happened after the **All Frames**
choice. This establishes a required job stage between dispatch and verification.
Long jobs need ownership of the scene, selected interval, pending dialog and
native completion state across requests. Reacquire Qt objects after restarting
the editor and resolve pending operations before submitting another apply.

Image capture is asynchronous as well. Setting many frames and requesting
images in one tight Python loop produced stale mesh poses even though frame
labels changed. Separate frame-setting and capture requests let the scene update
between captures. Use an event-driven capture job with fresh-file checks when
optimizing this route.

The initial Mocap and Motion Generation probes used the optional
`python_execute` tool. Their dedicated structured jobs now provide parameter
validation, progress observation and persistent recovery. Provider integrations
and the wider review/editing workflow remain in the [workflow roadmap](roadmap.md).

## Native workflow details

The job implementation uses pre-write receipts, an immutable
native source, owned queued callbacks and operation-specific completion evidence.
Native testing exposed additional requirements: import QtQuick before wrapping
native windows, traverse large visual trees breadth-first, recognize aliases of
the same Windows filename field, and confirm Mocap processing separately from
model download. A native message Error remains sticky even after progress events
leave the bounded event ring.

AutoPhysics switching uses the effectively visible toolbar button's `released`
signal. The installed ToolbarButton forwards this to the frontend action manager;
the lower-level Csc call does not synchronize that UI state in the same way.
An already enabled preview is switched off, verified off, then recalculated.
Both the new core-finished marker and the Snap marker are required for completion.

Unbaking can wrap an unbound derived Point `Angle Rotation` output across ±pi on
the storage guard. The job verifier accepts only equivalent modulo-2pi Euler
components and unchanged quaternion tolerance for that output; authored rotation
channels retain their strict branch checks.

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
