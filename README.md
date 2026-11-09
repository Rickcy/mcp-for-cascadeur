# MCP for Cascadeur

Smart MCP connects an AI assistant to the live Cascadeur application. Inspect
native scenes and rigs, author body and finger animation, preserve curves,
retime keys, run native animation tools and verify saved results. Optional
Kimodo generation and provider-neutral visual review extend the native workflow.

## Install

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/)
and a Cascadeur build containing **Scripts > MCP > Start script server**.
The supported native-job target is Windows Cascadeur **2026.2.3, build 44356**
with the validated Cascy rig. Generic discovery can inspect other installations;
their native editing behavior must be verified separately.

```powershell
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur@release/0.1.0 mcp-for-cascadeur setup --install-startup
```

Setup discovers Cascadeur, installs a stable host runtime outside the uv cache,
and configures the supported MCP client. Existing settings are preserved and
backed up. Restart Cascadeur, open a scene, and reconnect the MCP client. The
welcome screen alone does not provide scene-idle execution events.

```powershell
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur@release/0.1.0 mcp-for-cascadeur doctor
```

The release is installed from its Git tag; these commands do not require a PyPI
publication. For checkout/wheel installation, client JSON and custom paths, see
[installation](docs/installation.md).

## What is available

| Area | Operations |
| --- | --- |
| Scene and rig inspection | Application/build/license, native objects and channels, hierarchy, RigInfo, Point bindings, tracks, keys, native tools and API signatures |
| Body animation | Complete or semantic Point poses, bounded pose sequences, relative corrections, motion plans and native mirroring |
| Hands and independent FK | Validated finger-local rotations; separate guarded FK-subtree editing and restoration |
| Curves and timing | Keys, native interpolation including CLAMPED_BEZIER, curve-preserving Point edits and synchronized character retiming |
| Native animation tools | Video Mocap, pose-based Motion Generation, Animation Unbaking and AutoPhysics with progress and recovery |
| Recovery | Whole-model checkpoints, durable poses, request receipts, verified rollback and persistent recovery fences |
| Playback and delivery | Observed native playback, camera/viewport capture, native save/reopen and licensed verified FBX export |
| Optional Kimodo pipeline | Text or native body key poses → Kimodo-SOMA-RP-v1.2 → editable Cascy; retain and compare up to four candidates |
| Visual review | Gemini or a compatible local Qwen3-VL service; common report schema, multi-view motion evidence and explicit uncertainty |

The packaged MCP server supplies native structured tools plus
`visual_review_status` and `animation_review`. Visual review runs in the external
MCP process and can be used while Cascadeur is disconnected. The optional Kimodo
pipeline is a separate checkout-based CLI that uses the installed native runner;
it is not a `kimodo_generate` MCP tool. Full Python access is opt-in through
`setup --allow-python` and has Cascadeur's application privileges.

## The animation workflow

1. Inspect the actual scene, native capabilities, recovery state, rig and timeline.
2. Choose the input route: existing animation, native video Mocap, native endpoint
   inbetweening, or external Kimodo text/keypose generation.
3. Save an isolated source copy and capture the appropriate checkpoint. Plan body
   poses and timing from motion events; handle finger articulation separately.
4. Generate or apply bounded edits. Read actual solved results, protected channels,
   geometry and contact diagnostics. Poll native jobs to verified completion.
5. Render the resulting motion from several views and send the comparison to the
   configured visual reviewer. Correct only a diagnosed problem, then recheck it.
6. Save, reopen and verify the result; observe playback and verify any licensed
   export in the receiving application. Retain useful outputs and recovery records.

The complete operating instructions are shipped inside the MCP package:

| Resource | Purpose |
| --- | --- |
| `cascadeur://guide` | Native editing and recovery rules |
| `cascadeur://workflow` | Full Smart MCP route selection, generation, review and delivery flow |
| `cascadeur://capabilities` | Machine-readable route boundaries and limits |
| `cascadeur://tools` | Exact registered parameter contracts |
| `cascadeur://scene` | Fresh native scene context |

The `animate_character` prompt applies this workflow to a brief. An assistant
must report measured outcomes and remaining uncertainty; tool success alone
does not establish natural movement or game readiness.

## Configure visual review

Gemini and local models perform **visual analysis**, while Kimodo generates
motion and Cascadeur edits/solves it. The review provider returns the same
validated candidate IDs, usability, confidence, scores, findings, limitations
and preference. Matching JSON does not establish equal animation judgment.

```powershell
mcp-for-cascadeur review configure --provider gemini --model gemini-flash-latest
mcp-for-cascadeur review key set gemini
mcp-for-cascadeur review status --json
```

The key command uses hidden input. Windows stores the secret encrypted for the
current user outside the repository. Environment variables remain available on
all supported platforms. Do not put a real key in source, client arguments,
Docker files, prompts or committed examples.

An MCP client calls `animation_review` with an absolute workspace JSON request
and optional new output path. It returns review, selection and native-evidence
status without editing the scene. See the [request contract](docs/workflow.md#mcp-review-request).

For an already running local service:

```powershell
mcp-for-cascadeur review configure --provider openai-compatible --model '<local-model-id>' --base-url http://127.0.0.1:8000/v1 --media-transport frames
```

Provisioning that service/model is separate from configuring the MCP integration.
Use the video or timestamped-frame transport supported by the runtime. Measure
VRAM and benchmark a replacement on the same clips before trusting its rankings.
See [visual-review configuration](docs/workflow.md#visual-review-providers).

## Optional Kimodo generation

Kimodo runs in Linux GPU containers; Cascadeur remains on Windows. The supported
checkpoint is **Kimodo-SOMA-RP-v1.2**, with a real 4-bit LLM2Vec encoder. The
Indie-compatible route uses this project's Point retarget instead of Pro-only
native Retargeting Copy/Paste.

The checked transfer covers **2–120 frames at 30 FPS**, body motion and root
placement. Finger articulation and seamless loops need separate work. On a 12 GB
GPU, candidates run sequentially and the generator is released before native
review/local VLM inference. See [Kimodo setup and commands](integrations/kimodo/README.md).

## Limits and verification

Native body edits have a 121-stored-frame contract including a read-only terminal
guard. Character retiming accepts 2–8 synchronized keys. Native animation jobs
have a separate 512-stored-frame limit including their guard. A job's larger
budget does not expand the character-edit contract.

Native actions report dispatch. Structured jobs verify operation-specific
completion. After an uncertain write, inspect its receipt rather than retrying.
A restart does not clear a failed recovery fence. Saved scenes, checkpoints and
single-pose snapshots serve different purposes.

See [capability contracts](docs/capabilities.md), [workflow](docs/workflow.md),
[verification](docs/verification.md) and the [native API reference](docs/native-api-notes.md).

## Updating and removal

Install the intended release source and repeat setup, then restart Cascadeur and
the MCP client so imported native modules reload. `mcp-for-cascadeur uninstall`
removes the recorded startup hook and client entry, preserving scenes, models
and user settings. Remove a persistent uv installation separately with
`uv tool uninstall mcp-for-cascadeur`.

Contributors should follow [CONTRIBUTING.md](CONTRIBUTING.md) and run only affected
tests and immediate integration checks.

## Licenses

The integration code is MIT licensed. Cascadeur, its sample assets, NVIDIA Kimodo,
MotionCorrection, Blender and model weights retain their respective licenses and
notices. They are not relicensed by this repository's MIT license.
