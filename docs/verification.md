# Verification and supported scope

Use separate checks for package behavior, native execution and animation quality.
A passing Python test cannot establish native solver behavior; a successful
native operation cannot establish natural movement.

## Supported native target

Native job drivers target Windows Cascadeur 2026.2.3 build 44356. Kimodo transfer
uses the selected native humanoid's mapped Joint/Point rig and works without Pro
Retargeting Copy/Paste. Cascy and the installed UE5 Manny sample have completed
the native acceptance described below. The UE5 and UEFN samples both passed
native mapping and calibration inspection: 21 body roles and 43 actual Point
bindings each. Custom rigs, other builds and operating systems require their
own native acceptance.

The body-keypose route has been exercised with five captured poses and three
120-frame candidates at 30 FPS: generation, all-frame Point transfer, native
geometry checks, save/reopen, anchor comparison, three-view video review and
selection. Source scene integrity was checked. This is a technical acceptance
scope, not a guarantee for every prompt or a benchmark of naturalness.

Earlier Smart Motion acceptance in the installed Qt 6.5.3 `QGuiApplication` used
the former multi-candidate/review interface:
native-menu and MCP opening, two 120-frame body-keypose candidates, all-frame
three-view rendering, Gemini comparison, explicit candidate opening and owned
playback/stop. Safe cancellation during capture and subsequent continuation
reused retained generation/transfer stages. The panel also generated a raw
180-frame NPZ/FBX candidate. Both native results passed geometry, required-anchor
and save/reopen checks; the source file hash remained unchanged. Observed native
timeline progression was about 30 FPS; this does not measure display frame times
or certify movement quality. QtQuick is required; the host does not create a
Qt Widgets application inside Cascadeur.

The current panel creates one animation for the selected character and timeline
range. It submits no-render/no-review jobs and exposes neither VLM settings nor
candidate count or visual scores. Optional MCP/CLI review remains a separate
route. Long-duration generation uses native Kimodo continuation in bounded
240-frame inference segments, with no fixed total-clip cap. Package tests cover
long duration, exact segmentation, retained constraints and per-segment progress;
native acceptance must also check the complete output frame range and segment
boundaries on the intended target.

The current single-result panel generated **540 frames** on UE5 Manny at 30 FPS
in a real installed-app run. Kimodo v1.2 used three native continuation segments
of 180 frames; generation took approximately 10.1 seconds after model loading.
All 540 frames transferred in 27 checked batches. Maximum Point solver drift
was 0.861 cm, native body-connection error 0.00239 cm and auxiliary IK relative
position error 0.000124 cm. All-frame save/reopen drift was below 0.000046 cm.
Protected orientations were compared on SO(3); 290 equivalent Euler branch
changes were reported, so this route does not promise preservation of Euler
winding. The source snapshot checksum stayed unchanged.

The 540-frame result was cold-opened alone and native playback covered the
complete interval and stopped with one native play/stop pair. This workstation's
observed timeline progression was about 9–10 FPS on that scene; the file's 30 FPS
timebase does not establish real-time viewport performance. Preview deadlines
therefore allow slower native playback to finish instead of cutting it at the
nominal clip duration. No VLM request or review video was created for this run.

The same panel completed a constrained UEFN run from the actual timeline
selection **40–219**, producing **180 generated frames** numbered 0–179. Text was
left empty in anchor mode. The two range endpoints were captured automatically.
All 180 frames passed native geometry and auxiliary-IK validation and the saved
scene reopened successfully. Required anchors differed by at most 0.000018 cm
in Point position and 0.05 degrees in mapped Joint orientation. Maximum
save/reopen coordinate drift was 0.000016 cm. This is an in-place keypose-routing
test with identical endpoint poses, not a forward-walk quality claim.

Native panel playback covered UEFN frames 0–179 and stopped automatically.
Observed hot-session progression was about 3.8 FPS; this acceptance proves frame
coverage and native playback ownership, not real-time display performance.
After this complete transfer/reopen/cleanup/playback sequence, Cascadeur exited
normally with Smart Motion open: PID 30704, process exit code **0**. Intermediate
owned scene closure uses DataSourceManager.close_scene, refuses saved linked
scenes and checks that other tabs and the live callback anchor survive.

UEFN exposed a real mapping difference: the pelvis Joint pivot lies about
2.569 cm above its main Point controller. Reverse retargeting now anchors the
root to that actual Point, matching the forward adapter. The measured native
conversion roundtrip error fell below 0.000012 cm. Installed sample originals
were not modified; preparation used private workspace copies.

Earlier lifecycle tests: two fresh native processes exited with code 0 while Smart Motion remained open,
with the generation and variants tabs exercised separately. A cold restart
reopened the saved scene and restored the former review-enabled job history.
These earlier exits do not substitute for shutdown acceptance after a native
transfer in the current implementation.

Application shutdown is a separate lifecycle acceptance check. After opening and
closing/reopening Smart Motion, quit Cascadeur normally and observe the actual
process exit code **0**. Record this result before claiming clean shutdown, then
cold-start the application and reopen the panel and retained scene. A hidden
panel or a completed worker alone does not establish this result.

## Package and protocol checks

Run affected unit/contract tests and immediate integrations. For guide/resource
changes, exercise official MCP SDK initialization, resource listing/reads and
the character prompt. `cascadeur://tools` and `cascadeur://capabilities` must
describe the same enabled tools, including optional Python configuration.

Build a wheel/source distribution and inspect their contents before release.
They must contain the operating instructions and package modules, and exclude
keys, private settings, model weights, scenes, videos and disposable reports.
An isolated installed-wheel check must import from the installed package rather
than an accidental checkout on PYTHONPATH.

## Native editing checks

Use an isolated copy of a saved scene in the configured workspace:

1. Record build/license, scene identity, topology, units/FPS, recovery and unlocked
   track readiness.
2. Snapshot the declared edit scope and relevant protected channels.
3. Apply one bounded operation; read authored values and the solved result.
4. Check connection geometry, target residuals, untouched curves/fingers/settings
   and animation outside the interval.
5. Save a new result and reopen it. Compare stored samples and metadata rather
   than accepting a nonempty file as proof of correctness.
6. Observe native playback and its stop. Display frame-time stability requires
   an additional performance check.

Test valid edits and rejected input. After a timeout, inspect `transaction_status`
or the stage receipt. Verify recovery after native restart without replaying an
uncertain operation or forcibly clearing a fence.

For a fresh-copy retry after a synchronous Point error, verify live idle state
and process identity, the returned error envelope, source/plan hashes and absent
locks before creating the new attempt. Confirm that the old scene, receipt and
unknown transport marker remain byte-identical. The new result must pass the
complete normal save/reopen checks. This establishes a new successful transfer
from source; it does not establish rollback of the failed working copy.

## Native job checks

For Mocap, Motion Generation, Unbaking and AutoPhysics, verify the native source
copy, owned working scene, selected interval, completion markers and saved output.
Check cancellation, unavailable transport and interrupted-process recovery.
Progress stays indeterminate when there is no reliable native fraction.

Unbaking is an approximation: record actual displacement. AutoPhysics can change
motion substantially: compare it with the accepted source. A dispatched action
or closed dialog alone is not successful application.

## Visual reviewer acceptance

This section applies to optional MCP/CLI visual review. The Smart Motion panel
does not submit a visual review or produce review clips.

Use the same clips, duration, camera framing, temporal sampling, prompt and
contact diagnostics for provider comparisons. Include ordinary motion and known
defects: foot sliding, abrupt body changes, poor weight transfer, missing ground
contact, wrong action, occluded limbs and an unassessable video.

Check schema validity, candidate identity, timestamped findings, abstention,
confidence, ranking stability and agreement with an animator's labels. A local
Qwen3-VL endpoint returning the common schema proves integration compatibility;
it does not establish quality parity with Gemini. That claim needs a measured
benchmark on these cases. Do not silently lower sampling/resolution to fit memory.

Visual selection requires native geometry/persistence, required anchors, current media
hashes, adequate evidence and a suitable visual review. Contact labels and Joint
speed are approximate diagnostics. The result is the preferred acceptable
candidate in this comparison, not a physical or game-readiness certificate.

Panel result activation instead requires native geometry/persistence and any
required anchors. Its receipt records `native_validation_only`; this is not a
visual ranking and must not be presented as one.

## Output and resource checks

On a 12 GB GPU, observe actual VRAM during sequential generation. Confirm the
generator is idle before releasing it and loading a local visual model. Renderer
completion must restore camera/frame/display mode and stop its Qt timer. Remove
owned scratch PNGs only after their writer has stopped. Preserve the source,
accepted variants, useful review media and recovery records.

For game delivery, validate the target engine's skeleton/retargeting, scale,
axes, clip range, root extraction, loop seam and contacts after export. These
checks are separate from verifying the native `.casc` file.
