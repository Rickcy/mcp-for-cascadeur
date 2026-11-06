# Capability contracts

Tool discovery is available without a running application. Native reads and
writes require the installed Cascadeur application, its local script runner and
a compatible open scene. A tool's presence is not a claim of native acceptance,
physical plausibility or artistic quality. Measured acceptance is recorded in
[validation.md](validation.md).

## Application and animation operations

| Capability | Structured tools | Contract |
| --- | --- | --- |
| Live process and native capabilities | `app_status`, `app_capabilities` | Actual adapter/native build, Python, license, native tools and enum names. |
| Scene tabs and scene files | `scene_info`, `scene_list`, `scene_new`, `scene_activate`, `scene_open`, `scene_save` | Explicit tab indices, workspace-scoped files and fresh output verification. Reopen saved data to check persistence. |
| Object and native property inspection | `objects_list`, `object_inspect`, `object_hierarchy` | Exact native IDs, unambiguous names, behaviors, typed channels, settings and reference relationships. |
| All native object data | `object_data_read`, `object_data_set` | Exact owned data IDs or unique display names, including channels without behavior bindings. Typed writes check native readback; solver-driven outputs can refuse a generic edit. |
| Native selection | `selection_get`, `selection_set`, `tracks_select_frames` | Object selection and real native track/frame selection are distinct. Interval tools use the latter. |
| Timeline | `timeline_get`, `timeline_set_frame`, `timeline_set_range` | Stored frame count, playhead and separate native playback/visible boundaries. |
| Keys and interpolation | `tracks_list`, `tracks_set_keys`, `tracks_set_interpolation` | Explicit track IDs, locking checks, controlled allocation and retained section metadata. |
| Local and global poses | `pose_read`, `pose_read_sequence`, `pose_read_euler`, `pose_apply`, `pose_apply_sequence`, `pose_offset` | Real native values, bounded batch edits and requested-versus-observed readback. Driven rig outputs are better edited through character Point tools. |
| Transform authoring | `transforms_animate` | Per-object spaces, absolute quaternions or Euler deltas from an explicit reference frame, native interpolation and explicit timeline extension. |
| Native property writes | `object_properties_set` | Prevalidated native types and channel identities, with static-versus-animated scope disclosed. |
| Native motion measurement | `motion_measure` | Existing native trajectories, path length and maximum sampled step distance. This is separate from planning new motion. |
| Native mirroring | `native_mirror` | Actual native rig-aware mirror methods on explicit objects, using current-frame or selected-interval context. |
| Motion planning | `motion_sample`, `motion_mirror` | Deterministic position/easing/quaternion plans and mathematical reflection; these return data without editing Cascadeur. |
| Camera and review | `camera_get`, `camera_set`, `camera_frame`, `viewport_mode_set`, `viewport_capture` | Actual camera/display state and fresh PNG output. A screenshot is useful evidence but is not an automatic quality score. |
| FBX exchange | `fbx_import`, `fbx_export` | Native import/export, explicit workspace paths and native export entitlement. Validate receiving-application axes, skeleton and clip range separately. |
| Native actions | `actions_list`, `action_run` | Exact documented action IDs. Dispatch is not successful completion; some actions toggle state or depend on selection/license. |
| Native API discovery | `api_inspect`, `native_tools_list`, `native_tool_inspect` | Bounded public attributes, signatures and documentation. Inspection does not invoke discovered methods. |
| Advanced Python | `python_execute` | Explicitly enabled full-privilege native scripting. This is not a substitute for a structured feature or its tests. |

`tracks_retime` is the conservative independent-transform retimer. It is not the
character retimer and does not establish support for solver-driven rig channels.
It authors only native local position/rotation/scale inputs and animated
settings on standalone Basic/Transform objects; native updates produce derived
world outputs. It retains supported classic section metadata and verifies both
authored key samples and world poses. CLAMPED, baked/AI intervals, custom tangents
and non-key sections are outside this retiming contract.
`api_inspect` exposes actual `csc`, `app`, `view`, `scene`, `model` and `layers`
roots; the last two are the active scene's native viewers. `scene_save` returns
SHA256 when a fresh nonempty file is observed. A `pending` file receipt is not a
completed save.

## Native characters and semantics

`characters_list` identifies real `RigInfo` owner IDs. `character_skeleton` reads
the actual owned objects, Joint hierarchy, Point controls, native
`RigAdditionalInfo` relationships, track membership and relevant constraint
properties. It can omit per-key section details to bound responses.

The built-in semantic profile is deliberately strict. `rig_semantics` validates
the complete Cascy hierarchy and all 43 Point targets using exact unique native
names, types, membership and actual Joint-to-Point references. It checks the two
heel helpers by their exact names, native Point types and RigInfo membership;
it does not invent missing Box references. Unknown topology is rejected.
User-created `rig_profile_*` groups remain a separate feature and do not acquire
this native semantic validation merely by using similar role names.

| Role | Point slots |
| --- | --- |
| pelvis | center, orientation, left_hip, right_hip |
| chest | waist, waist_orientation, center, orientation, left/right_clavicle, left/right_clavicle_orientation, left/right_shoulder |
| head | neck, neck_orientation, center, direction, orientation |
| left/right_hand | center, direction, orientation |
| left/right_elbow | center, bend |
| left/right_foot | center, heel, toe, toe_direction, toe_orientation |
| left/right_knee | center, bend |

`character_pose` returns all writable Point positions and optional read-only
local/global Joint states. `semantic_pose_read` and
`semantic_pose_read_sequence` select named roles after full profile validation;
they read stored frames without moving the playhead. Driven body Joints are not
advertised as writable semantic controls.

`character_pose_apply` requires the complete native Point ID set.
`semantic_pose_apply`, `semantic_pose_apply_sequence` and `semantic_pose_offset`
merge partial role patches with each frame's real current targets. Ordinary
semantic batches contain at most eight increasing existing frames. They create
whole-character keys and use LINEAR intervals: authored easing is not preserved.
Offsets are relative and accumulate when repeated.

Explicit `sampled_motion=true` permits up to 64 complete semantic poses on
existing frames. Every role/slot is required, each native bound rigid Point
cluster must retain its geometry, and native playback/visible bounds must cover
the whole stored range. The result returns actual Point hashes rather than
repeating every large pose. Read actual poses separately when needed.

Checked character writes require a saved scene, exclusive unlocked character
tracks, supported native IK/curve state and a complete recovery checkpoint.
The current verified-contract limit is 121 stored frames. Additive stacks,
cycles, AI interpolation on character tracks, custom spatial tangents and CLAMPED_BEZIER are rejected
by the declared preflight. The actual solver result is checked after the native
transaction commits, including target residuals, connected-body anchors,
topology, static data, settings and relevant protected channels. Unverified
postconditions raise an error for the outer recovery transaction.

## Character curves and retiming

| Capability | Tools | Preservation and checks |
| --- | --- | --- |
| Character motion inspection | `character_motion_inspect` | Native connection geometry, mass-weighted/native COM, track metadata and stored-frame evidence. |
| Existing-key Point editing | `character_pose_preserving_curves`, `character_offset_preserving_curves` | Existing keys only; exact track/section metadata, labels, interpolation, easing and supported tangent state. Protected baked channels and unedited Point keys are checked across the clip. |
| Solver-aware character timing | `character_retime` | Complete synchronized character keys, native Point poses, authored finger channels/settings and native Joint-key readback. Optional contact stabilization has a separate declared scope. |
| Advisory edit eligibility | `character_edit_readiness` | The same declared native/curve/ownership/geometry prerequisites plus current recovery fence/capacity. It allocates no snapshot or history entry and cannot promise target feasibility. |

Preserving metadata does not preserve the edited trajectory's shape. Solvers can
move coupled Points and Joints; edit-impact reports distinguish direct targets
from observed coupled movement. A successful tool response is insufficient to
claim improved contact, motion timing or visual quality.

## Hands

`hand_pose_read` discovers both hands' five four-Joint finger chains through
native parent relationships. The first three segments expose 15 authored local
rotation channels per hand; terminal Joints are validated but not written.
`hand_pose_apply_sequence` accepts one to eight increasing existing frames of
named finger unit quaternions in WXYZ order.

Hand editing checks native finger track membership, including Box controls only
when their real `RigAdditionalInfo` Joint reference identifies a validated
finger. It verifies requested quaternion readback and the complete stored clip's
body Points, non-finger Joint poses, static/settings state and unrelated tracks.
Moving a palm Point group and curling fingers are separate operations.

## Recovery and durable poses

The recovery layer captures bounded native model state before eligible edits,
records transaction identity and verifies committed/rollback state. Persistent
failure records survive an adapter restart. A changed quarantine does not become
writable just because the application was restarted: a complete matching native
checkpoint is needed. Read diagnostics before deciding whether to restore.
Opaque asset payloads and arbitrary Python's external side effects are outside
the native model rollback scope.

`pose_snapshot` is a selected-object pose convenience. Full native checkpoints
and transaction receipts cover a different scope. `character_pose_snapshot`
writes a new integrity-checked JSON with every Point target and read-only Joint
verification, saved-scene identity and native rig fingerprint.
New durable files use format v2 and additionally preserve each validated
finger's native Euler representation, including full turns, together with its
quaternion. Restores read v1 files for compatibility and explicitly report that
the original Euler winding is unavailable in quaternion-only legacy files.
`character_pose_restore` validates that identity across sessions, restores Point
positions through the native character solver, restores all 30 validated
authored finger-local rotations from the recorded Joint state, and verifies all
recorded Joint poses. Existing section metadata, labels, interpolation and
weights remain exact; only the requested target frame may acquire native keys.
Untouched authored Point/finger keys and unrelated scene channels are checked.
Native adjacent interpolation is recomputed. This is one complete character
pose, not an entire clip backup. Point static global rotations remain protected.

`subtree_pose_read` discovers every real Joint descendant of an exact root.
`subtree_snapshot` captures every stored frame's local and global Joint pose;
the optional new workspace JSON is integrity checked. `subtree_pose_apply`
captures the same whole-clip scope automatically. `subtree_restore` restores
authored local channels, verifies both spaces at every recorded frame, and
preserves the current unrelated scene channels. It checks native object/channel
identity, parent hierarchy, track metadata, keys, scene identity and checksum.
`subtree_release_snapshot` frees a host-memory snapshot.

This FK scope is limited to 1-32 standalone unit-scale Joints, 2-121 stored
frames, exclusive unlocked tracks and exactly two LINEAR/FK endpoint keys.
RigInfo-driven body Joints, constraints, cycles and changed track layouts are
rejected. Ordinary single-frame subtree edits require an allocated successor;
the final stored frame remains readable. Whole-clip restoration replays all
recorded authored samples and verifies the terminal sample too. Scoped subtree
snapshots preserve unrelated edits; full native checkpoints recover a complete
bounded model and have a different purpose.

## Playback

`playback_start` checks a stationary playhead, sets real playback and visible
bounds, sets the start frame once and invokes native playback. A Qt timer only
observes frame changes; it never simulates playback through repeated frame
assignments. Playback is bounded to 20 seconds per session.

`playback_status` reports measured frames, range coverage and stop evidence.
`playback_stop` operates only on the adapter-owned session in its original
scene. External playback is not toggled. Native playback exposes a toggle rather
than a documented direct play-state getter, so the stop is reported verified
only after native frames remain stationary. Active or failed-unverified owned
playback blocks further edits.

## Acceptance status

Offline tests verify contracts, topology rejection, semantic merging, sampled
completeness/rigid cluster checks, hand quaternion prevalidation, durable pose
integrity and playback ownership/state transitions. They do not establish native
solver or graphics-driver behavior. Native success and failure/recovery tests
must be recorded separately for the installed build before expanding the
supported acceptance claim. See [validation.md](validation.md) for measured runs.
