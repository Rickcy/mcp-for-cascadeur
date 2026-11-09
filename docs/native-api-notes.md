# Native Cascadeur API reference

This reference maps the supported workflow to installed application interfaces.
Use `app_capabilities`, `api_inspect`, `native_tools_list` and
`native_tool_inspect` for the running build; a method or menu item alone does not
establish a completed animation operation. No vendor implementation or assets
are distributed with this package.

## Script runner and application thread

The user entry point is **Scripts > MCP > Start script server**; the corresponding
stop command stops it. The installed runner's loopback HTTP interface provides
`GET /health` and `POST /run`. It queues scripts for scene-idle execution on the
native application thread. An open scene is required.

The execution namespace includes `csc`, `app` and the callback's domain `scene`.
Structured adapter results are JSON inside the expression's outer Python string
representation: decode with `ast.literal_eval`, then `json.loads`, never `eval`.
Propagate native Error messages as well as exceptions.

The runner's request deadline does not interrupt an already executing native
operation. A lost response can therefore have an unknown write outcome. Keep
native batches bounded and inspect transaction/job receipts before another write.
The HTTP runner has no network authentication; retain its loopback address.

## Native interface map

| Task | Interface and operating rule |
| --- | --- |
| Application scenes | `csc.app.get_application()`, scene-manager `scenes/current_scene/set_current_scene`; application views and domain scenes have distinct lifetimes. |
| Load/save | Data-source manager loads full scenes; `view.save(path)` writes a scene. Verify the output and reopen it. |
| Object/channel discovery | Model/behavior/data viewers; resolve exact native IDs and authored versus derived channels before writing. |
| Frame reads | `DataViewer.get_data_value(data_id, frame)` reads stored samples without moving the playhead. |
| Checked edits | `Scene.modify_update` plus native data/layer editors and updater; check its result and independent readback. |
| Solver updates | `SceneUpdater.run_update/generate_update`; coupled controls can move after an authored edit. |
| Keys and allocation | Layer sections/keys and `fit_animation_size_by_layers`; key insertion and stored-frame allocation are separate. |
| Interpolation | Native section/key/interval metadata, including CLAMPED_BEZIER. Do not replace unaffected metadata with defaults. |
| Character bindings | Real RigInfo/RigAdditionalInfo relationships and Point controls; body Joint transforms are solved outputs. |
| Finger edits | Validated authored local rotations; moving a palm Point does not curl finger chains. |
| Camera/viewport | Active viewport camera structure, display mode and `RenderToFile.take_image`. Allow scene updates before asynchronous capture. |
| Exchange | Native FBX loader/exporter; inspect entitlement and verify target-app axes, scale and range. |
| Actions | Exact installed action IDs; dispatch needs operation-specific completion evidence. |

Installed declarations under `resources/scripts/stubs/csc/` are the primary
reference for that installation. Some tools expose additional live editor/QML
state beyond the stub. Discover actual signatures before exposing another route.
See the vendor [Python API overview](https://cascadeur.com/help/category/215),
[script menu](https://cascadeur.com/help/category/320) and
[action reference](https://cascadeur.com/help/category/301).

## Native animation jobs

The supported job driver targets Cascadeur 2026.2.3 build 44356 with validated
Cascy semantics. The build-specific routes include Motion Generation, Mocap,
Animation Unbaking and AutoPhysics. They are accessed through structured jobs,
not by replaying isolated menu calls after a timeout.

Each job owns its source copy, working scene, interval, expected UI/worker stages
and recovery fence. Queued callbacks execute on the application thread. Native
completion markers, geometry and protected-state verification precede saving a
new result. Native progress is indeterminate where no reliable fraction exists.
The external server can read a checksummed durable journal when the application
is unavailable; that observation does not restart native work.

Mocap needs a workspace MP4/reference plane and its official model. Download is
explicitly opt-in. Motion Generation operates on existing endpoint poses and a
trajectory, not text. Unbaking can introduce approximation displacement and
classic CLAMPED_BEZIER intervals. AutoPhysics requires calculation and explicit
application (`all_frames` or `only_keys`); opening an apply dialog is not completion.

## Qt/QML and scene lifetimes

Resolve current native editors/QObjects again after a scene change, tool closure
or application restart. Some custom QML pointer properties cannot be converted
directly to Python; inspect supported editor properties and queued callbacks
instead of retaining invalid wrappers.

Collect required metadata before invoking an action that may close its object.
Queue UI callbacks on the native thread and retain only the necessary owner
until completion. A scene-idle callback must keep its own domain scene alive:
switching the active tab inside that callback does not permit deleting the
callback's original scene. Close inactive owned views only from a distinct,
verified anchor callback.

Render frames through scene-update/capture stages and verify fresh complete PNGs.
A tight loop that changes frame labels and immediately captures can leave stale
mesh poses. The integrated renderer uses a bounded Qt task, restores camera,
display mode and frame, releases its timer/references, then encodes video on CPU.
Direct native video export is not used by the review pipeline.

## Preservation and recovery

CLAMPED_BEZIER is preserved by enum name with supported key/interval metadata.
Custom spatial tangents and unsupported additive/AI states remain outside the
checked contracts. Preserving section metadata does not preserve the trajectory
after a pose changes.

Model checkpoints protect bounded native channels/settings, timeline and
selection state. Opaque asset payloads and arbitrary Python external side effects
are outside that rollback scope. A recovery failure persists across restart.
Read diagnostics and use verified restore/recovery rather than forcibly
clearing a fence.

See [capabilities](capabilities.md), [workflow](workflow.md) and
[verification](verification.md) for exact feature boundaries and acceptance steps.
