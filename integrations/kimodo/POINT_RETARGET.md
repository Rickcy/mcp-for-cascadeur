# Point retargeting without a Pro action

`point_retarget.py` implements its own skeletal pose transfer through the public
Python API. It transfers the generator's SOMA motion to native Point targets and
works on Indie without Pro-only Retargeting Copy/Paste.

The source is `kimodo-retarget-motion-v1` JSON from `prepare_retarget_motion.py`:
global positions/rotations, neutral skeleton and a source NPZ checksum. The
target is a real native Cascy calibration from `calibrate(view)`: every body
Joint, every Point and the actual `RigAdditionalInfo` links. Calibration only
reads frame zero and does not move the playhead.

Planning keeps the generated bone directions and twist, adapts the target bone
frames, and uses the target's native rest offsets for FK, including hip and
clavicle branches. Root displacement is scaled by the leg-length ratio. A
single constant vertical shift places the lowest heel/toe on the floor for
unconstrained text generation. The keypose return route instead reverses its
recorded canonical/world transform and preserves the captured placement without
applying this grounding shift again.

Elbow, knee and toe Additional Points are shared anchors of adjacent rigid
bodies. Their ownership must not be inferred from their object names. The plan
solves the two native rigid triangles sharing each anchor, then checks **every
native link and pair of its Points across every frame**. A cluster exceeding
the existing native preflight tolerance is rejected before a scene write.

## Prepare the plan and isolated scene

```powershell
python -B integrations/kimodo/point_retarget.py plan `
  --calibration 'D:\Animation\kimodo\calibration.json' `
  --motion 'D:\Animation\kimodo\walk.retarget.json' `
  --output 'D:\Animation\kimodo\point-plan.json'

python -B integrations/kimodo/point_retarget.py prepare `
  --workspace 'D:\Animation\kimodo' `
  --plan 'D:\Animation\kimodo\point-plan.json' `
  --template 'D:\Cascadeur\samples\Cascy.casc'
```

The second command returns a new receipt and working path. It copies only the
target Cascy scene into a fresh isolated session.

## Native stages

Each stage uses a separate native request:

```powershell
python -B integrations/kimodo/native_import.py step `
  --receipt '<absolute session.json path>' --stage point_attach
python -B integrations/kimodo/native_import.py step `
  --receipt '<absolute session.json path>' --stage point_apply
python -B integrations/kimodo/native_import.py status `
  --receipt '<absolute session.json path>' --native
```

`point_attach` opens only the isolated working path and allocates the requested
frames plus one terminal guard. `point_apply` writes at most twenty complete
Point poses through the existing checked character writer. It preserves native
curve metadata and runs its rigid-target preflight, solver readback and
connection-anchor geometry checks. Each subsequent `point_apply` advances the
durable cursor; an uncertain or failed stage cannot be replayed automatically.

After all frames, the next stages are `point_save`, `point_reopen_prepare`,
`point_reopen` and `point_cleanup`. Reopen preparation activates the retained
`before.casc` in its own native request without closing the result. The following
request may close the result only while both the active tab and the actual
scene-idle callback belong to that retained anchor. Cleanup closes the inactive
anchor in a later request. Closing the active callback's domain scene is refused
even if another tab has already been activated within the same callback.
Reopening compares every Point at every animation frame.
Successful cleanup retains only `before.casc`, `result.casc` and `session.json`.
Input calibration, source motion and plan remain caller-owned inputs.

`native_import.py complete --receipt '<path>'` can run these separate requests
sequentially. Recovery, native
jobs, playback, process identity and transport-uncertainty checks apply to Point
sessions just as to the FBX bridge.

This is a geometry and persistence contract. Foot-contact timing, motion style,
loop continuity and visual quality require review of the actual resulting motion.
The current writer supports up to 120 animation frames at 30 FPS, plus its guard.

## Targeted checks

```powershell
python -B -m unittest discover -s tests -p test_kimodo_point_retarget.py -v
python -B -m unittest discover -s tests -p test_kimodo_native_import.py -v
```

These checks cover rest-pose differences, root translation, world rotation,
wrist roll, hip branching, shared native rigid anchors, invalid rotations and
CLI/Point-module interoperability. Native acceptance remains a separate step.
