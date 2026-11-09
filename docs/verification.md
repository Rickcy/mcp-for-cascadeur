# Verification and supported scope

Use separate checks for package behavior, native execution and animation quality.
A passing Python test cannot establish native solver behavior; a successful
native operation cannot establish natural movement.

## Supported native target

Native job drivers target Windows Cascadeur 2026.2.3 build 44356 and a validated
Cascy rig. The current Kimodo transfer targets that rig through public Point APIs
and works without Pro Retargeting Copy/Paste. Other builds, custom rigs and
operating systems require their own native acceptance.

The body-keypose route has been exercised with five captured poses and three
120-frame candidates at 30 FPS: generation, all-frame Point transfer, native
geometry checks, save/reopen, anchor comparison, three-view video review and
selection. Source scene integrity was checked. This is a technical acceptance
scope, not a guarantee for every prompt or a benchmark of naturalness.

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

## Native job checks

For Mocap, Motion Generation, Unbaking and AutoPhysics, verify the native source
copy, owned working scene, selected interval, completion markers and saved output.
Check cancellation, unavailable transport and interrupted-process recovery.
Progress stays indeterminate when there is no reliable native fraction.

Unbaking is an approximation: record actual displacement. AutoPhysics can change
motion substantially: compare it with the accepted source. A dispatched action
or closed dialog alone is not successful application.

## Visual reviewer acceptance

Use the same clips, duration, camera framing, temporal sampling, prompt and
contact diagnostics for provider comparisons. Include ordinary motion and known
defects: foot sliding, abrupt body changes, poor weight transfer, missing ground
contact, wrong action, occluded limbs and an unassessable video.

Check schema validity, candidate identity, timestamped findings, abstention,
confidence, ranking stability and agreement with an animator's labels. A local
Qwen3-VL endpoint returning the common schema proves integration compatibility;
it does not establish quality parity with Gemini. That claim needs a measured
benchmark on these cases. Do not silently lower sampling/resolution to fit memory.

Selection requires native geometry/persistence, required anchors, current media
hashes, adequate evidence and a suitable visual review. Contact labels and Joint
speed are approximate diagnostics. The result is the preferred acceptable
candidate in this comparison, not a physical or game-readiness certificate.

## Output and resource checks

On a 12 GB GPU, observe actual VRAM during sequential generation. Confirm the
generator is idle before releasing it and loading a local visual model. Renderer
completion must restore camera/frame/display mode and stop its Qt timer. Remove
owned scratch PNGs only after their writer has stopped. Preserve the source,
accepted variants, useful review media and recovery records.

For game delivery, validate the target engine's skeleton/retargeting, scale,
axes, clip range, root extraction, loop seam and contacts after export. These
checks are separate from verifying the native `.casc` file.
