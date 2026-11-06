# Validation

Local environment: Windows, Cascadeur 2026.2.3 (build 44356), embedded Python
3.11.0, external Python 3.12.14. Native checks use disposable copies of the
application's installed samples. No sample assets are distributed here.

## Version 0.2 acceptance

The current implementation adds persistent recovery, solver-aware character
retiming, curve-preserving edits, semantic/finger tools and scoped FK recovery.
[Capability contracts](capabilities.md) describe the structured tools and their
supported scope. This record describes actual execution and observed results.

The final 0.2.0 checkout passed 432 tests in the 22 explicitly selected feature
and integration modules listed in CI. Ruff lint and formatting checks passed
for the package, tests, examples and native acceptance scripts. Fresh-wheel
stdio/SDK acceptance is documented in [package-validation.md](package-validation.md).

| Native case | Observed result |
| --- | --- |
| Complete model capture | 289 objects, 26 frames, 8,745 channels, 13 tracks; consecutive captures compare with zero differences. Checkpoint approximately 9.6 MB. |
| Existing-key curve edit | Three left-hand Points moved 0.1 unit. Maximum target error `0.00000367`; exact section metadata and 5,810 protected channels retained. Connection error below `0.00560` across the clip. |
| Nondefault curve metadata | Native Bezier weights `0.31/0.67`, key label and AI interval text retained exactly; the full track-metadata SHA-256 stayed unchanged. All 1,095 channels on an unselected FIXED finger track were protected. CLAMPED_BEZIER rejection left the complete model digest unchanged. |
| Character retiming | Eight keys moved on 13 tracks. Point error `0.001830`; Joint position error `0.001817`, angular error `0.0000881` radians. All 30 finger channels, 51 animated settings and 5,729 protected channels verified. Execution about 11.7 seconds. |
| Checked checkpoint restore | One verified Undo restored the retimed model with zero full-checkpoint differences, including an independent status comparison. Execution about 15.3 seconds. |
| Process-interruption recovery | A real crash left a pending journal. After restart, Python, timeline changes, actions and export were blocked while reads remained available. Manual recovery restored the full checkpoint with zero differences in about 16.3 seconds. |
| Deliberate post-commit failure | An object rename followed by an injected exception returned `TRANSACTION_ROLLED_BACK`, `rollback_verified=true`; one Undo and a full comparison proved restoration. |
| Deliberate recovery failure | Injected failures in both Undo and manual restoration retained the fence with a real changed object. Removing the injections and running recovery verified restoration before clearing the fence. |
| Finger authoring | A left index finger rotated three degrees and was restored. Both edits verified all 26 frames, body Points, non-finger Joint poses, static data and settings. |
| Semantic body edit | A partial left-hand offset passed readiness and native checks; solver adjustments were reported. Durable pose restoration reproduced all recorded Joint poses. |
| Sampled semantic motion | Four complete semantic poses accepted with `sampled_motion=true`; actual Point hashes and all-frame native connection checks returned. |
| Stationary contact retiming | Eight support targets pinned and two foot tracks baked. Maximum contact error `0.002811`; connection error below `0.00560`. Native canonical STEP on an adjacent-key interval is authored and verified explicitly. Subsequent Undo and independent checkpoint status both report zero differences. |
| Euler fidelity | A real native rotation of `2π + 0.25` retains its complete Euler representation through checkpoint encode/decode and native-value cloning. New snapshots compare Euler winding as well as quaternion orientation. |
| Durable body/finger pose v2 | After changing a finger, restoration recovered all 30 authored finger channels, body pose and existing curve metadata. Finger Euler error below `0.00000114`; Joint position error below `0.00190`, angular error below `0.000107` radians. |
| Independent FK subtree v2 | A two-Joint, six-frame clip with a full Euler turn was changed after snapshot. Scoped restore recovered every local/global pose and exact Euler turns, while retaining a later independent Joint edit. Euler restoration error zero. |
| Generic authoring and channels | Eleven Cube Points animated at three keys; maximum position error `0.0000145`. Native leaf-track selection and ancestor groups verified. Animation data read/write, Static bool read/write/restore and endpoint-key deletion down to frame zero passed. |
| Native playback | One play and one stop command; actual frames 0–24 observed continuously, then verified stationary. One initial frame assignment; no scripted per-frame animation. |
| Verified FBX delivery | Fresh 11,051,389-byte ASCII FBX, Y-up, bake/Euler filtering, range 0–25 and fresh native source copy. Model unchanged. Exclusive publication and replacement with matching SHA-256 passed; missing/wrong expected hashes rejected before source-copy creation. |
| FBX import | The verified FBX imported into a new disposable scene: 66 Joints, one Mesh, 26 frames. Fresh `.casc` output saved and hashed. Receiving-application interoperability outside Cascadeur remains untested. |

The first retiming implementation exposed a native access violation during bulk
section replacement. Retiming and manual track restoration now use native key
helpers and `change_section` with JSON metadata. The successful retiming and
recovery runs above used that correction.

The vendor runner has a 30-second deadline. An early checkpoint restore committed
just after that deadline; its durable request record and later full comparison
correctly resolved the outcome. Redundant captures were removed while retaining
post-view-state verification. Unknown writes must be resolved by request ID and
readback, never automatically replayed.

Native testing also exposed float32 camera/quaternion roundoff, native selection
of parent folders and Static getter overloads. Their fixes retain strict model
checks: camera tolerance is separate; angular comparisons normalize quaternion
norm before comparing orientation; unexpected selected leaf tracks still fail;
Static values use the overload without a frame argument.

An early quaternion-only checkpoint could not reconstruct the original Euler
branch at two inbetween frames. Recovery remained fenced. An explicitly selected,
older managed checkpoint was then verified in full before clearing the fence;
the record correctly reports `recovered_to_checkpoint`, not successful rollback
to the failed operation's prestate. Original failure evidence was preserved.
Current model checkpoints and v2 body/FK pose files retain native Euler values.

Custom runtime update-node interpolation flags are outside the serialized model
checkpoint. The FK full-turn fixture deliberately disables the vendor's default
Euler filter for its authored test channel. Complete native scene files remain
the appropriate backup for custom application/runtime state and opaque assets.

## Version 0.1 baseline: Python and distribution

The 132 feature tests cover contracts, HTTP transport, official MCP SDK stdio,
settings, installation, scene tools, animation, motion math and rig profiles.
They include invalid nested inputs, unknown write outcomes, swallowed native
callback errors, metadata preservation and timeline allocation order.

The wheel was installed into a fresh environment outside the checkout. Help,
tool discovery, SDK initialize/resources/argument validation and
`uvx --isolated --from <wheel>` succeeded without access to repository imports.
The registry contains 50 tools; the server exposes 49 when Python is disabled.
`uv pip check`, Ruff and wheel/sdist builds pass.

CI runs Python checks on Windows and Linux with Python 3.11 and 3.12. It does not
install Cascadeur and is not evidence of native support on Linux.

The initial branch's four hosted matrix jobs completed successfully. Motion
planning, reflection and the documented action catalog also work without a
connection to the native application.

## Version 0.1 baseline: native checks

| Check | Observed result |
| --- | --- |
| Startup and diagnostics | Managed hook starts the installed runner after a scene opens; health and live process/scene status respond. |
| Capability discovery | Native build, license flags, tool names, interpolation modes and viewport modes returned by the installed application. |
| Cube scene via MCP SDK | Read controls, snapshot, sample seven motion poses, extend animation, write/read back, set playback bounds, frame the camera, capture PNG, save and reopen `.casc`. |
| Cube pose readback | Maximum position error approximately `0.000011` scene units. |
| Cascy controllers | All 43 native Point controllers read and written on the original frame; maximum position error approximately `0.000933`. |
| Cascy timeline extension | Three poses at frames 0, 12 and 24, with a rigid group translation of 15 scene units; native readback matches within `0.001`. |
| Cascy roundtrip | Seven sampled poses, selected-pose restoration, PNG, native save/reopen and all 43 Point controls checked again after loading. Maximum reopened error `0.001039`, against an explicit `0.002` scene-unit tolerance. |
| Rig profiles | Body group bound to 43 native IDs; profile and grouped pose read both work after reopening the saved scene. |
| Finger controller | Right thumb Box controller rotated 15 degrees in local space, read back, then restored to its original quaternion. |
| FBX export | Native animation export produced a fresh 677,664-byte FBX under the tested installation's export entitlement. Import into another application is not covered by this check. |
| Forced disk reload | A uniquely named copy of the saved Cascy file loaded through the native parser; frames 0, 12 and 24 read back with maximum position error `0.000933`. |

Sampling that Cascy motion at seven frames produced a maximum solver residual of
approximately `0.001039` scene units, just above the default `0.001` readback
tolerance. The native smoke example explicitly uses `0.002` for Cascy and records
the chosen tolerance. Changing this parameter affects the reported match flag,
not the written targets or solver behavior.

The first development test exposed an invalid native allocation order: creating
fixed keys outside stored frames could abort the application. Extension now
allocates a terminal section, resizes native data and generates the update before
using key helpers. A separate native Cascy test established that the requested
last written frame needs a following terminal frame for the rig solver. The
allocation includes that terminal frame; playback can still end at the requested
last frame. Regression tests preserve this ordering and boundary.

Native readback verifies requested channels, not every solver-coupled control,
contact, physical plausibility or artistic quality. Arbitrary rigs and every
interpolation mode have not been accepted. Whole-rig retiming was outside the
initial release's scope; version 0.2 has a separate checked character retimer.

## Repeating native acceptance

With Cascadeur open on a scene and setup complete:

```powershell
uv run python examples/native_smoke.py --sample Cube
uv run python examples/native_smoke.py --sample Cascy
```

Each run creates a new validation directory inside the configured workspace,
opens a copy of the selected installed sample, and records a JSON report beside
the `.casc`, selected-pose JSON and PNG outputs. It never edits the vendor sample.
The saved scene is copied to a new path before loading, so an already-open tab
cannot substitute for reading the saved file.
