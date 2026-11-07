# Uthana cloud motion

The MCP process can submit Uthana text/video jobs and download motion into its
configured workspace. Cascadeur's embedded Python does not load the Uthana SDK
or credentials. Downloading motion and importing/retargeting it are separate steps.

Configure the key using a hidden prompt:

```powershell
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur mcp-for-cascadeur credentials set uthana
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur mcp-for-cascadeur credentials status uthana --json
```

Windows encrypts the key for the current Windows account with DPAPI under the
application's user data directory. `config.json`, MCP client configuration and
job records contain no API key. If the terminal cannot hide input, the command
refuses to read a key. An inherited `UTHANA_API_KEY` takes precedence; use
`credentials set uthana --from-env` to encrypt that existing value on Windows.
Other platforms use the environment variable without a plaintext storage fallback.
Do not put keys in command arguments or tool calls. Gemini credentials can be
stored with `credentials set gemini`; that does not itself enable a Gemini tool.

Read `uthana_account` for account/rate information. The text, video and custom
character submission tools require an explicit cost ceiling and unique request ID.
Keep the returned job ID, poll `uthana_job_status`, and use `uthana_jobs_list`
after reconnecting. A lost submission response can mean the provider accepted
work; inspect the recorded outcome before retrying a paid request.

The cost ceiling gates our estimate before submission; Uthana does not enforce
it as a billing limit. Text-to-motion pricing uses generated output duration,
which can include an endpoint sample beyond the requested interval. New text
jobs reserve one additional output second in the estimate. At $0.10/second,
a 4-second request therefore needs a $0.50 ceiling; an 8-second request needs
$0.90. Receipts report both `requested_seconds` and `reserved_seconds`. This
reserve is our budgeting policy, not an official rounding rule or guaranteed
maximum charge. Existing receipts retain their original quote when replayed.

Use `uthana_download` with an absolute destination inside the workspace after
completion. The downloaded artifact remains available for review before native
import. Provider acceptance or a successful download does not establish motion
quality or correct retargeting.

## Tools

| Tool | Contract |
| --- | --- |
| `uthana_account` | Read account allowances and current model prices. No generation. |
| `uthana_character_upload` | Upload an existing workspace FBX/GLB with auto-rigging disabled. Returns a target character ID. |
| `uthana_text_start` | Submit text-to-motion 3.0, with a requested length of 4–10 whole seconds. |
| `uthana_video_start` | Submit video-to-motion 2.1 from a local, unfragmented MP4/MOV lasting 2–60 seconds. Unsupported or contradictory timing is rejected before submission. |
| `uthana_job_status` | Read a checksummed receipt and optionally poll its known upstream job. No generation replay. |
| `uthana_jobs_list` | List local receipts without a network connection or credentials. |
| `uthana_download` | Download completed motion as FBX/GLB at 24, 30 or 60 FPS to a new workspace path. Verify format and SHA-256 before publication. |

The input file's byte snapshot and SHA-256 are recorded before upload. A repeated
`request_id` returns its existing receipt; changing its input or budget is an
error. An interrupted submission with no upstream ID remains unresolved. The
integration has no upstream cancellation endpoint and cannot prove that an
interrupted remote generation stopped.

## Cascy transfer

Use a disposable copy of the installed `samples/Cascy.casc` inside the configured
workspace. Retain its linked textures when copying the sample.

1. Open the copy and export its neutral model with `fbx_export(mode="model")`.
   This exports a target rig without animation; it does not require changing the
   sample's one-frame playback range.
2. Upload this FBX with `uthana_character_upload`. Retain the returned
   `character_id` and the upload receipt.
3. Pass that character ID to `uthana_text_start` or `uthana_video_start`. For
   example, request four seconds of walking at a natural pace, with a $0.50
   estimate ceiling and a unique request ID. Video inference uses its own source
   skeleton; the recorded character ID selects the export target.
4. Poll until `status="completed"` and a motion identity is recorded. Download
   with the same character ID, `output_format="fbx"` and `fps=30`.
5. Inspect the FBX in a separate scratch scene before replacing animation on the
   original rig. `fbx_import(mode="scene", set_offset=false)` preserves source
   placement for comparison. The internal `host.importing` validation helpers
   compare Joint hierarchy, measured lengths, scales and parent coordinate
   transforms; they do not infer a bind pose from frame zero.
6. Open a fresh copy of the original Cascy and use `fbx_import(mode="animation")`.
   A matching skeleton is required. Keep the downloaded FBX and original `.casc`.
7. Check the native RigInfo, Point controls, poses and clip bounds. Save to a new
   `.casc`, create a complete checkpoint, reopen the saved file and compare the
   model. Test editing on another copy before treating imported motion as
   editable through the character tools.

Native import projects animation onto the existing rig. Matching names and
proportions do not establish exact Joint-pose preservation. Live Cascy checks
found measurable wrist/twist adjustments even when full save/reopen comparison
had zero differences. Import accuracy and native persistence are separate
results; see [native evidence](validation.md#uthana-text-and-video-transfer).

The first acceptance covers this prepared Cascy on Cascadeur 2026.2.3 build
44356. Arbitrary skeleton mapping, long-clip checked editing, locomotion/stitch
tools, visual review and automatic motion repair are separate milestones.

To remove the encrypted local copy:

```powershell
mcp-for-cascadeur credentials delete uthana
```

This leaves any inherited environment variable unchanged. No Uthana API request
is sent by credential setup, tool discovery, or native bridge installation.
