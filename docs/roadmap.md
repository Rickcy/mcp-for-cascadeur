# Animation workflow roadmap

The agreed architecture uses Uthana for motion generation and video capture,
Gemini Flash or Qwen-VL for visual review, and this MCP bridge for applying,
editing and measuring animation in Cascadeur. Each review job selects one vision
provider. Uthana text/video generation, existing-rig upload and checked downloads
are implemented. Native Cascy transfer and persistence have live acceptance;
visual review and the remaining high-level workflow are planned work.
Released behavior and native evidence are recorded in
[capabilities.md](capabilities.md) and [validation.md](validation.md).

| Component | Responsibility |
| --- | --- |
| Uthana API | Text-to-motion, video-to-motion, locomotion, motion stitching and looping where supported by the selected endpoint. |
| Gemini Flash / Qwen-VL API | Review timestamped previews, compare candidates and propose edits linked to visible evidence. |
| MCP and Cascadeur | Rig mapping, editable animation, scoped changes, native measurements, recovery, preview and delivery. |

Start acceptance with a prepared Cascy character, a level floor and a 5–10 second
clip at 30 FPS. Use the same character, camera and travel speed when comparing
ordinary walking with a heavy gait. Extend support to other rigs after validating
the complete workflow on this character.

1. **Validate native execution routes.** Inventory the installed SDK, vendor
   Python commands, action IDs and Qt/QML interfaces for Mocap, Motion Generation,
   AutoPosing, AutoPhysics, contact cleaning, retargeting, unbaking and review
   capture. Identify parameters, prerequisites, progress, cancellation and the
   observable result for each operation. Record build and license requirements.
   Use discovered Qt properties and slots for operations exposed through QML.
   Queue UI callbacks and check object lifetime before invoking them. Schedule
   long operations through persistent jobs with native readback.

   Acceptance requires operation-level checks on disposable scenes, including
   timeout resolution, restart recovery, save/reopen and measured scene changes.
   Record each tested route in [native-api-notes.md](native-api-notes.md), with
   failures and remaining gaps. Feed verified native operations into generation,
   transfer, polishing and preview jobs.

2. **Define the clip and job contracts.** Record source FPS, time range, units,
   axes, rest pose, joint mapping, root motion, floor, provider parameters and
   file hashes. Persist cloud job IDs and native request IDs separately. Resume
   polling after reconnecting and resolve uncertain submissions before retrying.
   Keep provider credentials in runtime configuration and redact them from logs.

   Acceptance requires restart recovery and a trace from the request through its
   downloaded motion and native scene result.

3. **Support dense clips and editable rigs.** Establish a checked path from an
   external skeleton to native character controls. Preserve existing finger
   animation when the source contains body motion only. Prepare baked/FIXED
   input for scoped editing, with measured key reduction or unbaking where
   appropriate. Extend bounded capture, readback and recovery to the full target
   clip. Current character checks cover at most 121 stored frames, sampled
   authoring accepts 64 full poses, and retiming accepts eight synchronized keys.
   A 10-second clip exceeds these contracts even when writes are batched.
   Native Unbaking produced CLAMPED_BEZIER in live checks. Extend curve-preserving
   editing and recovery acceptance to that representation as part of this stage.

   Acceptance requires measured joint/contact error, protected-channel checks,
   save/reopen and verified restoration of a disposable native scene.

4. **Connect Uthana and deliver the first generated clip.** Add text, locomotion
   and video jobs with status, cancellation where supported, downloads and
   explicit failure reporting. Track the skeleton and parameters returned by each
   endpoint. Measure cold and warm request times, download time, native transfer
   and total delivery time on the same test clips.

   Acceptance requires a walking clip from text and a captured clip from a
   single-person video, each transferred to Cascy and saved as editable animation.

5. **Add measured analysis and visual review.** Extract phase-dependent foot
   contacts, contact sliding, ground penetration, root trajectory, discontinuities
   and loop boundaries. Capture timestamped previews from consistent views and
   send them to the selected vision API with the requested motion description.
   Return observations with frame ranges and supporting measurements. Evaluate
   locomotion in world-space and in-place modes using the appropriate contact
   interpretation.

   Acceptance uses labeled walking, running, jumping and asymmetric-motion clips,
   together with deliberately introduced contact and timing defects.

6. **Implement scoped revision and reference workflows.** Translate comments
   into edits to explicit intervals and body groups. Extract timing, selected
   poses or trajectory from a reference clip according to the requested result.
   Apply constraints through native editing and supported provider parameters.
   Check protected animation and transition boundaries after every edit. Preserve
   comparable candidates and stop according to the requested checks and job
   budget. Validate native AutoPosing, physics, retargeting and unbaking operations
   against the installed build, license and observed postconditions.

   Acceptance covers creation from text, scene analysis, a targeted edit,
   capture from video, new motion using a reference, and polishing an existing
   clip. For heavy walking, compare
   ordinary and heavy variants at equal travel speed and review both with a
   neutral character appearance.

7. **Package the workflow and measure delivery.** Expose high-level creation,
   analysis, revision and job-status operations through the existing uvx setup.
   Document provider configuration and usage costs. Verify native scene
   persistence, preview playback and exported animation in a receiving app.
   Publish measured end-to-end timings together with clip length, rig, providers
   and number of revision passes.

   Run tests for changed functionality and its direct integrations. Native
   acceptance uses disposable scenes. Automated provider contract checks use
   recorded or synthetic responses, with live API acceptance run separately.

Provider references are [Uthana text-to-motion](https://uthana.com/docs/api/capabilities/text-to-motion),
[video-to-motion](https://uthana.com/docs/api/capabilities/video-to-motion),
[locomotion](https://uthana.com/docs/api/capabilities/locomotion),
[Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
and [Qwen visual understanding](https://www.alibabacloud.com/help/en/model-studio/vision).
