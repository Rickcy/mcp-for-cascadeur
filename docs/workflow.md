# Smart MCP animation workflow

The native MCP layer reads and edits Cascadeur. Kimodo is an optional external
motion generator. Gemini or a local vision-language model provides visual
review. Each part has a separate role, evidence and failure state.

## Choose the input route

| Input | Route | Result and limits |
| --- | --- | --- |
| Existing animation | Native inspection and bounded Point/curve/finger edits | Keep source, unrelated channels and accepted timing protected. |
| Text description | Optional Kimodo pipeline | New body/root motion transferred to a separate Cascy scene; 2–120 frames at 30 FPS. |
| Body key poses in Cascadeur | Kimodo keypose mode | Generate a standalone interval constrained by captured native poses; verify returned anchors. |
| Video | `mocap_start` | Native Cascadeur Mocap, workspace MP4 or existing video reference; official model must be available. |
| Existing endpoint poses | `motion_generation_start` | Native pose-based Motion Generation; this is not a text endpoint. |
| Baked motion | `unbaking_start` | Classic editable intervals; approximation displacement must be measured. |
| Physics refinement | `autophysics_start` | Calculate/apply native physics with an explicit interval and all-frames/only-keys choice. |

Native menu/API discovery can expose more application functionality, but an
available method or dispatched action is not a checked workflow. The optional
Kimodo scripts are checkout CLIs, not additional registered native MCP tools.

## Inspect and protect the scene

Read `cascadeur://scene`, `app_status`, `app_capabilities`, `recovery_status`,
`characters_list`, `character_skeleton` and `tracks_list`. Resolve exact object
IDs, units, FPS, existing frame bounds and license/build support. Use
`object_inspect`, `object_data_read`, `native_tool_inspect` or `api_inspect` when
the required channel or operation is unknown.

Save an isolated native source and choose the recovery scope:

- `checkpoint_create`: bounded whole-model state for a sequence of edits.
- `character_pose_snapshot`: one durable body/finger pose with native identity.
- `subtree_snapshot`: supported independent FK subtree across stored frames.
- Native jobs/pipeline: owned scene copies and durable operation receipts.

A `.casc` file, an in-memory model checkpoint and one pose snapshot have different
purposes. Inspect `character_edit_readiness` before semantic body edits. Validated
Cascy `rig_semantics` is distinct from user-defined rig-profile groups.

## Blocking and generation constraints

Choose meaningful events: contact, weight acceptance, passing, push-off, takeoff
or landing. Set their timing, root path, facing, duration and desired root-motion
or in-place behavior. Uniform frame spacing is useful for transport checks, not
an artistic selection strategy. The integration does not automatically choose
the best phase keys, infer armor weight, or guarantee a seamless loop.

Kimodo keypose mode captures actual native body Joint transforms and Point
targets without moving the playhead. Conversion adapts proportions and coherent
SOMA geometry, centimeters/meters, floor origin, initial planar root and heading.
The exact inverse restores world placement after generation. Floor/canonical
input checks and an algebraic roundtrip run before GPU work; final native
readback checks positions and rotations after transfer.

Kimodo receives the prompt and all constraints for the whole interval. Its
postprocessing applies the anchors. The returned Cascy clip contains every
generated frame, not just the input keys. Body constraints do not preserve
separate finger articulation, facial animation or prop/armor simulation.

For candidate comparison, calculate 1–4 variants sequentially with recorded
seeds and retain each NPZ, FBX and manifest. Source masks diagnose predicted
contacts; native measurement must separately inspect the retargeted result.

## Editing and refinement

Author body motion through Point controls; solver-driven body Joints are
read-only outputs. Use `hand_pose_apply_sequence` for validated finger-local
rotations. Moving a palm does not curl fingers.

For existing keys, `character_pose_preserving_curves` and
`character_offset_preserving_curves` retain supported curve metadata.
`character_retime` accepts 2–8 synchronized keys; `tracks_retime` has a narrower
independent-transform scope. CLAMPED_BEZIER remains a native interpolation type.
Preserving metadata does not preserve the old path after a changed pose.

Run Unbaking when editable classic intervals are needed and measure its
approximation error. Apply AutoPhysics only with a declared scope, then compare
the result with the accepted source. Inspect available fulcrum, contact and
penetration data/tools before choosing an exact native action. There is no
universal verified automatic penetration/foot-lock cleanup tool in this release.

Correct the smallest diagnosed problem. Preserve unrelated controls, fingers,
root travel and authored keys. Measure the native result, then review the changed
motion. Candidate selection does not constitute an autonomous artistic repair
loop; generic stylistic repair and game-readiness certification are not provided.

## Visual review providers

The visual reviewer analyzes the resulting animation. It does not generate
skeletal motion or directly edit Cascadeur. Render front, side and three-quarter
views with the same duration and framing. Declare source FPS, sampled-frame
cadence and any held preview frames. Stills/contact sheets can establish pose
evidence but cannot establish temporal quality.

Both providers return a common validated review: candidate IDs, usability,
confidence, naturalness/prompt/contact/continuity scores, timestamped observations,
limitations and an optional preferred candidate. SHA256 binds the review to its
actual media. The pipeline combines visual review with approximate contact
diagnostics and requires successful native geometry, persistence and any anchors.
If evidence is missing/stale, confidence is insufficient, native checks fail or
no candidate is suitable, the outcome is `needs_revision` with no automatic winner.

Use one comparative request for the retained candidates. Repeat it after a
meaningful correction or for a specific unresolved issue, rather than after
every tool call. A model's preference is an assessment, not ground truth.

### MCP review request

`visual_review_status` reads public provider/model settings and credential
availability. `animation_review` runs visual analysis in the external MCP process,
without invoking Cascadeur or editing the scene. Pass `request_path`, an absolute
JSON file inside the configured workspace, and optionally a new `output_path`.
Provider/model/base-URL/transport overrides are available; keys are never MCP
arguments.

```json
{
  "prompt": "A person walks forward naturally",
  "candidates": [
    {
      "id": "candidate-01",
      "scene": "D:/Animation/kimodo/run/candidate-01/native/session/result.casc",
      "receipt": "D:/Animation/kimodo/run/candidate-01/native/session/session.json",
      "validation": {"constraints_required": false},
      "media": [
        {"path": "D:/Animation/kimodo/run/candidate-01/review/front.mp4", "view": "front", "render_sample_step": 2},
        {"path": "D:/Animation/kimodo/run/candidate-01/review/side.mp4", "view": "side", "render_sample_step": 2},
        {"path": "D:/Animation/kimodo/run/candidate-01/review/three-quarter.mp4", "view": "three-quarter", "render_sample_step": 2}
      ],
      "metrics": {
        "native_contact_speed_m_s": 0.05,
        "native_contact_coverage_ratio": 0.3
      }
    }
  ]
}
```

Use actual pipeline paths and measured metrics. Supported native Point-transfer
receipts, plans and saved-scene hashes gate automatic selection; supplied
`validation` booleans alone do not establish acceptance. If the receipt is absent
or unsupported, the tool still provides visual findings but refuses automatic
acceptance. Native geometry/persistence cannot be established from video alone.

### Provider settings

Use private CLI configuration and hidden credential input:

```powershell
mcp-for-cascadeur review configure --provider gemini --model gemini-flash-latest
mcp-for-cascadeur review key set gemini
mcp-for-cascadeur review status --json
```

Windows encrypts saved secrets with DPAPI for the current user. For another
account/platform, provide its own key through the environment. Configuration and
credentials remain outside the repository. The provider/model returned by the
service is recorded in the review.

| Environment setting | Meaning |
| --- | --- |
| `VISUAL_REVIEW_PROVIDER` | `gemini` or `openai-compatible` |
| `VISUAL_REVIEW_MODEL` | Gemini model or exact locally served model ID |
| `VISUAL_REVIEW_BASE_URL` | Local compatible API, default `http://127.0.0.1:8000/v1` |
| `VISUAL_REVIEW_MEDIA_TRANSPORT` | `video` or ordered timestamped `frames` |
| `VISUAL_REVIEW_API_KEY` | Credential override for the selected provider; process environment only |
| `VISUAL_REVIEW_SAMPLING_FPS` | Explicit temporal sampling request |
| `VISUAL_REVIEW_MAX_TOTAL_FRAMES` | Total explicit frame-transport budget |
| `VISUAL_REVIEW_MAX_FRAMES_PER_VIDEO` | Per-video frame-transport budget |
| `VISUAL_REVIEW_FRAME_WIDTH` | Frame-transport resize bound |
| `GEMINI_API_KEY`, `GOOGLE_API_KEY` | Gemini credential fallbacks |
| `GEMINI_MODEL` | Legacy Gemini model fallback |

Provider-specific environment overrides take precedence over saved settings.
Never copy keys into prompts, model containers, scenes, run records or source.

### Local Qwen3-VL

Provision a compatible local multimodal server separately, then configure its
exact model name and transport:

```powershell
mcp-for-cascadeur review configure --provider openai-compatible --model '<local-model-id>' --base-url http://127.0.0.1:8000/v1 --media-transport frames
```

Native video transport requires the server's `video_url` support and appropriate
multimodal/context limits. Qwen's vLLM video example uses video frame processing
options; current vLLM configurations may require explicit trust for request
multimodal kwargs. Use the official [Qwen3-VL examples](https://github.com/QwenLM/Qwen3-VL)
and [vLLM multimodal reference](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
for the served runtime.

Frame transport extracts ordered timestamped frames on CPU and sends image
inputs; this supports runtimes whose video interface differs. Its frame/resolution
budget must be explicit. An insufficient budget fails instead of silently
reducing evidence. Start with a small known clip/candidate set, profile memory
and increase evidence only when it fits. A sparse sequence limits judgments of
fine motion continuity even if its JSON report is valid.

The default explicit frame budget is 96 total, 48 per video, with a 384-pixel width
bound. Approximate required frames are `candidates × views × seconds × FPS`.
For example, three candidates, three views, four seconds and 12 FPS need about
432 frames, exceeding that default. Configure a measured larger budget, choose
a supported video transport, or deliberately change sampling and disclose its
quality limitations. A smaller resize bound can obscure feet and hand detail.

Re-evaluate a completed run without another generation/transfer/render:

```powershell
uv run --no-project --with numpy --with imageio-ffmpeg python -B integrations/kimodo/pipeline.py review --output '<MCP workspace>\kimodo\run' --provider openai-compatible --model '<served-model-id>' --base-url http://127.0.0.1:8000/v1 --media-transport video
```

Provider/model/sampling profiles and reviewed media identities distinguish cached
reviews. Changing provider does not reuse another model's assessment as its own.
Compare providers on equal evidence before claiming comparable quality.

Local Qwen3-VL shares the contract and acceptance gates. Equivalent quality is
unproven until benchmarked on the same normal/defective motion clips and compared
with animator labels. Do not advertise equal quality from schema compatibility.

## 12 GB GPU operation

Generate candidates one at a time. Complete CPU motion conversion, verify the
generator is idle, then stop/release its service before Cascadeur review and
local VLM inference. The integration does not automatically load/unload arbitrary
local model servers. Configure and stop those runtimes explicitly.

Quantization, context, visual frame count/resolution, native viewport use and
other applications all affect VRAM. A 12 GB card is not a guarantee that a chosen
Qwen checkpoint and evidence budget will fit. Record measured peak memory and
the sampling configuration used for any quality comparison.

## Delivery and recovery

Save a new native result, reopen and compare the relevant stored data. Observe
bounded native playback and verify its stop. Timeline progression is not a
display frame-time measurement. Use `fbx_export_status` and
`fbx_export_verified` when licensed, then inspect the receiving application's
skeleton, axes, scale, root extraction, clip range and loop seam.

After an uncertain write, inspect `transaction_status` or the durable stage
receipt. Do not replay relative edits/toggles. Native cancellation retains the
fence until the worker has stopped and rollback is verified. A restart is not an
unlock; use the appropriate native-job or model recovery operation.

Kimodo `pipeline.py resume` reuses verified artifacts/native/review stages. It
does not resubmit generation or resume an interrupted diffusion pass. Remove
only owned temporary files after the worker is proven stopped; keep accepted
scenes, all retained variants, useful review evidence and recovery records.
