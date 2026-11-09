# Native Point retargeting

`point_retarget.py` transfers Kimodo's SOMA motion to the chosen Cascadeur
character through public Joint/Point APIs. It works on Indie without the
Pro-only Retargeting Copy/Paste action. The target is selected from the native
scene rather than replaced by a fixed sample character.

## Target mapping and calibration

`calibrate(view, character=..., frame=0, mapping=...)` reads the selected
character's native anatomical reference, Point controllers and RigAdditionalInfo
links. It does not rename objects, edit the source or move the playhead.
Calibration records exact native identifiers and a topology fingerprint.

The humanoid profile maps 21 body roles. Native names can be recognized from
common Cascadeur, UE5 and UEFN hierarchies; ambiguous or custom names require an
explicit role-to-Joint mapping. Actual Point membership and native links remain
authoritative. A plain FBX bone hierarchy must first be rigged in Cascadeur.
Extra bones remain part of the target; the solver controls their existing
connections. This is body motion transfer, not generated finger/facial animation
or support for every nonhumanoid topology.

The source is `kimodo-retarget-motion-v1` JSON from
`prepare_retarget_motion.py`: global transforms, neutral skeleton and an NPZ
checksum. Plans preserve source bone direction/twist and target native rest
offsets, including hip and clavicle branches. Root displacement scales with leg
length. Unconstrained generation applies one constant floor-placement shift.
The keypose route reverses its captured canonical/world transform instead.

Shared elbow, knee and toe anchors are resolved from native links. Planning
checks every link and Point pair on every frame before writing the target.
There is no fixed clip-length cap. Inference segmentation belongs to the
generator; native transfer receives the complete resulting interval.

## Prepare an isolated result

Smart Motion prepares these files automatically from the saved source copy.
The lower-level commands remain available:

```powershell
python -B integrations/kimodo/point_retarget.py plan `
  --calibration 'D:\Animation\kimodo\target-calibration.json' `
  --motion 'D:\Animation\kimodo\walk.retarget.json' `
  --output 'D:\Animation\kimodo\point-plan.json'

python -B integrations/kimodo/point_retarget.py prepare `
  --workspace 'D:\Animation\kimodo' `
  --plan 'D:\Animation\kimodo\point-plan.json' `
  --template 'D:\Animation\MyCharacter.casc'
```

The second command copies that target scene into a fresh owned session and
returns its receipt and working path. The result contains the requested number
of frames beginning at zero, plus a stored terminal guard outside the playback
range. The source scene is retained separately.

## Native stages and verification

Each stage uses a separate native request. `point_attach` opens the isolated
working path and allocates its frames. Subsequent `point_apply` stages advance
the durable cursor in bounded batches of complete Point poses. The writer
checks target membership, rigid geometry, solver readback and connection anchors.
An uncertain stage is retained for inspection and is not replayed automatically.

`point_save`, `point_reopen_prepare`, `point_reopen` and `point_cleanup` save and
verify the result. Reopen preparation activates the retained `before.casc` in
its own request before closing the result in a later safe callback. Reopening
compares every written Point at every animation frame. Source/calibration/plan
hashes remain pinned throughout the session.

```powershell
python -B integrations/kimodo/native_import.py status --receipt '<session.json>' --native
python -B integrations/kimodo/native_import.py step --receipt '<session.json>' --stage point_apply
```

Successful cleanup retains the useful source/result scenes and session evidence.
Failures preserve the owned files needed for recovery. The integration does not
discard a user's source animation or treat a nonempty `.casc` file as proof of
successful transfer.

Explicit pipeline continuation may create a new owned attempt when a synchronous
Point callback returned a terminal error. It first verifies the original source
and plan hashes and inspects the same live native process in an idle state. The
old working scene, session and uncertain transport marker are retained unchanged.
The new attempt starts from the immutable source and has its own directory and
receipt; the candidate's `native-attempt.json` identifies the current attempt.
This is not a verified rollback of the old copy. A running operation, retained
CLI lock, lost response without a returned callback error or a recovery blocker
still requires inspection and prevents new writes.

This contract establishes native geometry and persistence. Style, contact timing,
loop seams and target-engine export need their own motion/delivery checks.
The [verification guide](../../docs/verification.md) distinguishes tested targets
from capabilities that still require native acceptance.

## Focused checks

```powershell
uv run pytest -q tests/test_humanoid_mapping.py tests/test_kimodo_point_retarget.py tests/test_kimodo_native_import.py
```

These checks cover anatomical mapping, native membership, proportion/rest-pose
differences, root travel, twist, shared rigid anchors and retained stage ownership.
They complement native scene testing rather than replacing it.
