# Validation

Local environment: Windows, Cascadeur 2026.2.3 (build 44356), embedded Python
3.11.0, external Python 3.12.14. Native checks use disposable copies of the
application's installed samples. No sample assets are distributed here.

## Uthana text and video transfer

Live acceptance on 2026-11-07 used Uthana SDK 0.8.0 and installed Windows
Cascadeur 2026.2.3 build 44356. Cascy's original neutral model was exported and
uploaded as an existing rig with auto-rigging disabled. Both downloads requested
that target character and 30 FPS. Source FBX files and native scene copies were
kept outside the repository in the validation workspace.

| Check | Observed result |
| --- | --- |
| Text-to-motion 3.0 | A four-second walking request completed; native source metadata contained 97 samples at 24 FPS. Retargeted FBX imported as 120 stored frames at 30 FPS. |
| Video-to-motion 2.1 | A rendered walking reference, 61 frames at 30 FPS with 2.034-second container duration, completed and imported as 61 stored frames. This test does not establish camera-captured human-video accuracy. |
| Target rig | Both imported clips retained 289 objects, 66 Joint objects, all 43 Point controls, original native IDs and owned rig topology. Raw source and target skeleton preflight passed. |
| Native persistence | Both `.casc` files were saved, reopened and compared against complete bounded model checkpoints: zero differences. A reopened application view has a different view identity; model equivalence was checked separately. |
| Editable controls | Both clips passed character edit readiness. A 0.1-unit left-hand correction at video frame 30 passed the checked semantic editor on a separate copy, including protected animation and full-clip connection checks. |
| Rig geometry | Text motion inspection covered all 120 stored frames and 26 body connections; maximum native connection-anchor error was 0.005429 scene units. |
| Playback | Text playback advanced through the full requested range, using one native play command and one stop command. Stop was observed and verified. The single scripted frame set positioned the playhead before playback. |
| Cloud restart/replay | Submission and polling used separate Python processes. Replaying the completed text request through a provider object that rejects every network call returned its original job and result; no generation was repeated. |
| Installed MCP | The final wheel was exercised through the official MCP client: live account read, durable status, completed text-request replay and native scene information all succeeded. Its user-configured server exposed 111 tools including optional Python. |

Import fidelity is an approximation. Across five sampled frames in each clip,
raw FBX versus native Cascy showed maximum Joint position differences of
0.540495 scene units for text and 0.730800 for video. The corresponding maximum
local/global orientation differences were 0.197311 and 0.357904 radians,
including twist joints. Video foot-position differences stayed below 0.026
scene units at these samples; fingertip world-space displacement inherited
wrist changes while finger-local rotations were retained. These measurements
did not pass an exact-pose comparison; no tolerance was relaxed to label them
lossless. The native save/reopen comparisons above concern the imported result.

The account balance changed by $0.5059 during the two generation tests. Native
TTM metadata reported 4.041667 seconds for the four-second request, which
motivated reserving an additional output second in new pre-submit cost
estimates. Balance change is recorded separately from a per-job billing ledger;
the integration does not claim an upstream hard spending cap.

Windows DPAPI stored credentials outside the repository. Neither native
acceptance files nor credential records are distribution contents. Visual
quality scoring, contact repair, arbitrary-rig mapping and checked editing of
longer clips remain separate milestones. See [Uthana setup](uthana.md).

## Native workflow probes on 2026-11-07

Additional installed-build checks exercised Motion Generation, reference-video
import, Mocap, Unbaking and AutoPhysics through native commands and Qt/QML.
Motion Generation produced a walking clip. Mocap processed 31 synthetic-reference
frames in 18.378 seconds of recognition time, saved the result and retained exact
sampled pelvis/foot positions after reopening. Unbaking produced CLAMPED_BEZIER
intervals, and AutoPhysics changed the clip after its apply dialog was confirmed.

These probes also exposed two native crashes, followed by verified restart
recovery with zero model differences, and an asynchronous image-capture issue.
[Native API notes](native-api-notes.md#live-native-workflow-checks) record routes,
measurements, limitations and the output directory. These experiments use the
existing optional Python tool and native actions. Dedicated Mocap and Motion
Generation workflows are planned in [roadmap.md](roadmap.md).

## Version 0.2 acceptance

The current implementation adds persistent recovery, solver-aware character
retiming, curve-preserving edits, semantic/finger tools and scoped FK recovery.
[Capability contracts](capabilities.md) describe the structured tools and their
supported scope. This record describes actual execution and observed results.

The final 0.2.0 checkout passed 432 tests in the 22 explicitly selected feature
and integration modules listed in CI. Ruff lint and formatting checks passed
for the package, tests, examples and native acceptance scripts. Fresh-wheel
stdio/SDK acceptance is documented in [package-validation.md](package-validation.md).

All four [hosted matrix jobs for the implementation commit](https://github.com/Rickcy/mcp-for-cascadeur/actions/runs/42259028598)
passed on Windows/Linux with Python 3.11/3.12, including isolated wheel discovery
and official MCP SDK acceptance. These jobs do not run the native application.

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

## Native animation jobs (2026-11-07)

The structured job routes are exercised on Windows with Cascadeur 2026.2.3,
build 44356, embedded Python 3.11 and Qt 6.5.3. Each run uses a scene copy in the
configured workspace. Local acceptance evidence is stored beside each result;
vendor samples, large scenes, videos and the downloaded model are not committed.

| Check | Observed result |
| --- | --- |
| Motion Generation | `motion_generation_start(0,30)` generated 1,333 changed Point samples, produced FIXED intervals, committed a fresh 4,103,799-byte `.casc`, and passed complete protection checks across 289 objects, 8,745 channels and 13 tracks. |
| Mocap | Starting with a one-frame Cascy scene, automatically chose an MP4, imported 31 reference frames, confirmed processing, observed completion and committed a fresh 4,116,477-byte `.casc`. Full protection passed, including the one owned video plane, 87 new channels and two mesh assets. No manual UI step was needed. |
| Unbaking | Converted generated FIXED intervals to BEZIER/CLAMPED_BEZIER/STEP, held both finger guard keys and committed a 4,095,718-byte native result. The native approximation moved Point samples by up to `4.826861` scene units within the clip; this is measured simplification, not exact motion preservation. |
| AutoPhysics | Set and read back SelectionEditMode and playback/visible bounds 0–30, observed calculation, confirmed All Frames, measured 1,333 changed Point samples, protected guard frame 31 and committed a 4,116,357-byte native result. |
| Enabled AutoPhysics | Verified the real toolbar route in both directions, then started a job with the preview already enabled. The job observed OFF, performed a fresh calculation, applied it and committed with all protected checks passing. |
| Saved curves and poses | Unbaking curves matched the next job's forced disk-loaded input exactly. After AutoPhysics, forced another disk reload: track/curve metadata matched exactly and 129 Point samples differed by at most `0.00000935` scene units. |
| CLAMPED edit | Authored CLAMPED_BEZIER on 11 native body tracks, then offset all 43 Points by one scene unit at frame 0. Target residual `0.000902`, exact track/curve metadata and static/settings preservation, 32 frames checked; saved natively. |
| False native postcondition | Unbaking initially changed unkeyed finger values on guard frame 31. The protected-state check rejected it and reopened the immutable source; the complete recorded model matched with zero differences. This motivated explicit guard anchors rather than excluding terminal values from checks. |
| Process interruption | Terminated the test Cascadeur process after a Mocap job had extended a one-frame scene to 32 frames. With the application stopped, the MCP SDK returned the checksummed durable journal with `connected=false` and unknown execution state. After restarting, `native_job_recover` restored the one-frame source with zero checkpoint differences. |
| Cancellation | Cancelled a Mocap job during preparation. The importer finished without dispatching inference, the native source reopened, full comparison reported zero differences and the write fence was released. A later process restart confirmed the recorded terminal outcome. |

Native Unbaking output can include FK key metadata. CLAMPED support does not
relax the existing IK requirement for Point-preserving character edits; an
ineligible Unbaked scene was rejected and its unchanged checkpoint verified.

The unit and integration checks cover save failure without a fresh result,
committed receipts surviving final journal/marker failures, cancellation without
worker-stop proof, damaged checksums, offline MCP fallback and strict protection
of foreign channels, static data and assets. These checks do not measure visual
animation quality.

The final local check ran 460 tests covering the changed host modules, dispatch,
MCP resources/stdio, contracts and installation. Ruff, wheel/sdist builds and an
isolated `uvx --from <wheel>` tool-discovery check passed. The packaged registry
contains 103 structured tools plus optional Python; eight tools manage native jobs.

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
