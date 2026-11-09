# Kimodo generation and visual review

This optional checkout integration runs **Kimodo-SOMA-RP-v1.2** and a real
4-bit LLM2Vec encoder in Linux GPU containers. Cascadeur stays on Windows. It
supports text generation and generation constrained by native body poses,
then transfers the motion to the character selected in Cascadeur. The panel
creates one animation without video capture or visual review. Optional MCP/CLI
workflows can retain multiple candidates and compare them with a visual reviewer.

The current checkout adds a native **Smart Motion** panel and `kimodo_*` MCP
tools over this same pipeline. Long operations run in a durable external worker;
Cascadeur's application thread is reserved for native scene operations. The
published `release/0.1.0` tag predates the panel and Kimodo MCP tools. Follow the
[current checkout installation](../../docs/installation.md#current-checkout-and-smart-motion)
to use these additions.

## Requirements and assets

- Docker Desktop's Linux/WSL2 engine with NVIDIA GPU support.
- The installed MCP bridge, a saved scene and trusted Python enabled for native
  transfer. Raw generation needs no running Cascadeur or native calibration.
- Compatible Kimodo source, MotionCorrection, skeleton/skin/FBX assets and the
  merged LLM2Vec encoder checkpoint, obtained under their respective licenses.
- The pinned motion checkpoint below. Models and vendor assets are not included
  in this repository or its wheel.

Expected layout:

```text
.local/kimodo/
  source/                         # Kimodo, MotionCorrection and export assets
  models/
    text-encoders/generated/LLM2Vec-Meta-Llama-3-8B-Instruct-merged/
    checkpoints/Kimodo-SOMA-RP-v1.2/
```

`prepare_archive.ps1 -Archive '<compatible archive>'` prepares source/encoder
assets from a local compatible archive. It skips unchanged completed extraction
and refuses to merge a changed/incomplete extraction. It does not extract old
motion checkpoints. The source archive, original unmerged Llama weights, its
Linux virtual environment and unrelated models are not runtime requirements.

Download the public v1.2 motion checkpoint separately:

```powershell
uvx hf download nvidia/Kimodo-SOMA-RP-v1.2 --revision 0e3c89024d896064b6851201373f7834784a386b --local-dir .local/kimodo/models/checkpoints/Kimodo-SOMA-RP-v1.2
```

The runtime verifies the motion weights against SHA256
`fdcef624140a12c0a4840192ce0b75cafb4e73228a893da3864b4a990d3e919e`.
This public motion checkpoint does not require a Hugging Face account. Encoder
access and redistribution remain subject to the encoder/base-model licenses.
See the [NVIDIA model card](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.2).

Only this motion checkpoint is supported. Models mount read-only outside the
image. Runtime downloads and implicit Hugging Face authentication are disabled.
The encoder must be real LLM2Vec with 4-bit NF4 quantization; there is no hash-text
fallback. The initial image build downloads pinned runtime dependencies.

## Optional visual review for MCP/CLI

The Smart Motion panel does not use a visual reviewer or need its credentials.
For an assistant-driven MCP/CLI review, configure it outside the repository:

```powershell
mcp-for-cascadeur review configure --provider gemini --model gemini-flash-latest
mcp-for-cascadeur review key set gemini
mcp-for-cascadeur review status --json
```

Or point it to an already provisioned local Qwen3-VL-compatible service:

```powershell
mcp-for-cascadeur review configure --provider openai-compatible --model '<served-model-id>' --base-url http://127.0.0.1:8000/v1 --media-transport frames
```

The local service/model is a separate prerequisite. Choose a compatible video or
timestamped-frame transport and an explicit evidence budget. See
[provider settings and memory limits](../../docs/workflow.md#visual-review-providers).
Credentials can also come from process environment; they never belong in source,
Docker images, scene files, manifests or reports.

## Start, inspect and stop the generator

Run from the repository root:

```powershell
powershell -NoProfile -File integrations/kimodo/kimodo.ps1 -Action start -Build
powershell -NoProfile -File integrations/kimodo/kimodo.ps1 -Action status
powershell -NoProfile -File integrations/kimodo/kimodo.ps1 -Action logs
powershell -NoProfile -File integrations/kimodo/kimodo.ps1 -Action stop
```

Subsequent starts can omit `-Build`. `-Action build` only builds the image.
`start` waits for real encoder/backend health, verifies the checkpoint and all
required encoder shards, and reports the resolved exchange path. The image is
`mcp-for-cascadeur-kimodo:1.2.0-cu126`; the backend listens on
`http://127.0.0.1:9552`, with the encoder reachable only inside its Compose network.

Models default to `.local/kimodo/models`. The exchange directory is `kimodo`
inside the configured MCP workspace, or the local Documents fallback printed by
the launcher. Override with `-ModelsPath`, `-ExchangePath` and `-Port`.
`-TimeoutSeconds` bounds waiting; a timeout never resubmits a generation.

`stop` removes this project's containers/network, retaining models, source,
outputs and runtime caches. `-PurgeRuntime` additionally removes its named runtime
volumes. The launcher does not install drivers or change WSL/system settings.

## Panel and MCP entry points

Install the current package with its `animation` extra, run setup and configure
the prepared checkout path. Open **Scripts > MCP > Smart Motion** or use
`mcp-for-cascadeur motion panel`. Choose the scene character and timeline range,
enter a prompt and generate one animation. Existing keys or the current pose can
be added as anchors. Advanced controls include raw export and Docker service
management; there are no VLM, score or candidate-count controls. See its
[panel guide](../../docs/smart-motion.md).

`kimodo_generate` exposes the shared pipeline plus optional multi-candidate and
visual-review settings. It returns
a job ID immediately. Poll `kimodo_job_status`; the worker continues if the panel
closes or the MCP client disconnects. `kimodo_jobs`, `kimodo_cancel`,
`kimodo_resume`, `kimodo_review`, `kimodo_select`, `kimodo_runtime` and
`kimodo_import_run` manage the same retained jobs. Configuration and readiness
use `kimodo_configure` and `kimodo_status`.

The installed CLI accepts a JSON request with the `kimodo_generate` fields:

```powershell
mcp-for-cascadeur motion generate --request '<generation-request.json>' --json
mcp-for-cascadeur motion job <job_id> --json
```

The native and raw routes accept a total duration of at least two frames, with
no fixed 120/1800-frame cap. Long requests use Kimodo's native sequential
continuation in balanced inference segments of at most 240 frames. Each segment
uses body history, root translation and heading from the previous one. The exact
requested total length is retained. This limits attention memory on the 12 GB
setup; time, host memory and disk usage still grow with clip length. Generation
is at 30 FPS. Waiting timeout can be set from 30 to 86400 seconds.

For the current-scene route, supply `target_scene`, `character_id` and optionally
`rig_mapping` (canonical body role to native Joint ID/name). The pipeline reads
that character's actual native Point rig and creates a pinned calibration before
GPU work. Native key poses additionally use `source_scene`, increasing
`keyframes` and the neutral SOMA `template_motion`. The source interval is
recorded by `first_frame`; the retained standalone result starts at zero.
The saved native target must use 30 FPS and centimetres. Its settings are checked
before GPU work; unsupported timebases require an explicit resampling workflow.

The panel submits `samples: 1`, `no_render: true`, `no_review: true` and
`auto_activate: true`. Geometry, persistence and required anchors still gate
acceptance. These jobs neither load a visual-review configuration nor capture
videos. The source copy and completed result are retained.

MCP/CLI callers may instead enable rendering/review, request 1–4 sequential
candidates or use `generate_only: true` for NPZ/FBX output without a native target.
Prepared `constraints` are raw-route NPZ files inside the exchange directory,
hash-bound at submission. Legacy runs using explicit calibration and
`template_scene` remain supported. See the
[shared parameter and job contract](../../docs/workflow.md#smart-motion-panel-mcp-and-cli).

## Full text-to-motion run

Choose a saved target scene and an exact character ID reported by Cascadeur.
The pipeline calibrates its native body rig before generation. This command
matches the panel's one-animation, no-video/no-VLM workflow:

```powershell
uv run python -B integrations/kimodo/pipeline.py run `
  --prompt 'A person walks forward naturally at a steady pace' `
  --frames 360 --samples 1 --seed 420 `
  --target-scene '<MCP workspace>\MyCharacter.casc' `
  --character-id '<native character ID>' `
  --no-render --no-review `
  --output '<MCP workspace>\kimodo\walk'
```

Use `--rig-mapping '<JSON object>'` for unresolved custom humanoid names. The
mapping follows actual native membership; a bone-only FBX must first have a
usable Cascadeur Point rig. UE5 and UEFN names can be recognized automatically.
The target keeps its native object names and body proportions.

The run directory must be new and inside the configured workspace's `kimodo`
folder. The pipeline records inputs, starts the generator, submits exactly one
job, retains each completed candidate, performs CPU conversion, releases the
idle generator, then transfers and verifies native results. The no-render route
opens a result after geometry/persistence/required-anchor checks. Optional MCP
render/review workflows add their own media and visual acceptance gates.

The lower-level pipeline also exposes:

| Option | Meaning |
| --- | --- |
| `--text-weight`, `--constraint-weight` | Finite guidance weights from 0 to 10, default 2.0 |
| `--no-cfg` | Disable classifier-free guidance |
| `--sample-index` | Default output alias, smaller than `--samples`; all variants remain retained |
| `--render-sample-step` | Native pose sampling from 1 to 30, default 2; use 1 for every frame |
| `--no-review` | Save native variants and videos without calling the visual API |
| `--no-render` | Save/open native animation with no video capture or VLM; panel default |
| `--no-activate` | Retain a recommendation without final selected-scene activation |
| `--generate-only` | Raw NPZ/FBX route; no native target required |
| `--constraints` | Prepared exchange NPZ for the raw route; mutually exclusive with native keypose capture |
| `--cancel-file` | Cooperative stop signal within `workspace/kimodo`; no force-kill |

Weights and CFG affect model guidance, not a guaranteed physical property.
Render sampling is recorded in the evidence; a sparse video cannot establish
fine motion continuity. Provider/model/transport/sampling budgets use the
options listed by `pipeline.py run --help` and the same reviewer configuration.

## Generate between captured body poses

Add these arguments to the run command:

```powershell
  --source-scene '<MCP workspace>\posed-animation.casc' --keyframes 0,30,60,90,119
```

Frame indices are zero-based existing native frames. The first/last selected
frames must span `--frames`. There is no fixed total-frame cap. In the panel,
range boundaries are included automatically and existing keys can be added with
one button. Choose additional poses from meaningful phases of the movement.

Capture reads actual native body Joint transforms and all 43 Point targets
without moving the playhead. It converts proportions, coherent SOMA geometry,
units, floor origin and initial root/heading, with an exact inverse for return.
Conversion/canonical input checks run before GPU work. After transfer and
save/reopen, actual anchor position/rotation error is measured. Default anchor
acceptance is 1 cm and 5 degrees; the observed errors are retained in the report.

The result is a new standalone interval on a separate copy of the chosen target scene. Its body poses
are stored frame by frame. The source animation is preserved; finger articulation,
facial animation, attached-object physics and seamless looping are outside this
transfer contract.

## Optional MCP/CLI candidates and visual selection

`--samples` accepts 1–4 candidates, generated sequentially with seeds
`(base_seed + index) mod 2**32`. Each retains NPZ, FBX and a manifest with pinned
input/output checksums. A common candidate index is saved after each complete
candidate, so completed results survive a later failure.

The panel always requests one result; the controls below remain available to MCP
and lower-level CLI callers.

Each result transfers to its own native scene and passes full-frame rigid-target
and solver readback checks. The scene is saved/reopened and anchors are checked
when required. Optional native videos show front, side and three-quarter views. The
renderer observes scene updates between PNG captures, encodes on CPU and cleans
owned scratch frames only after its native worker stops. Review videos disclose
sampled/held-frame cadence and preserve the clip's duration.

When rendering and visual review are explicitly enabled, the visual provider
compares all candidates in one request and returns the common
validated report. Selection combines visual review (75%) and approximate contact
diagnostics (25%), with mandatory geometry/persistence/anchor/media checks.
`needs_revision` means that no candidate can be selected from the available
evidence. Selection is not an automatic repair loop or a game-readiness certificate.
Local Qwen3-VL compatibility does not establish quality parity with Gemini; use
the [reviewer acceptance procedure](../../docs/verification.md#visual-reviewer-acceptance).

The lower-level launcher can generate without transfer/review:

```powershell
powershell -NoProfile -File integrations/kimodo/kimodo.ps1 -Action generate -Prompt 'A person walks forward naturally' -Frames 120 -Samples 3 -Seed 420 -ReleaseGpuAfter
```

`-Constraints '<exchange-relative constraints.npz>'` supplies prepared anchors.
`-SampleIndex` chooses a requested default alias while retaining all variants;
it is not a quality ranking. `-ReleaseGpuAfter` stops only an idle generator after
artifact verification. Omit it only when deliberately retaining GPU models.

## Status and recovery

`run.json` is the pipeline outcome, with generation/native/render receipts and
the review/selection paths. `progress.json` carries the current stage and actual
backend or renderer counts where available; other stages remain indeterminate.
Native Point transfer additionally reports its readback-confirmed frame cursor
after each bounded batch, without dispatching a write merely to update progress.
The panel/MCP worker stores its durable job journal separately in the workspace.

```powershell
mcp-for-cascadeur motion cancel <job_id> --json
mcp-for-cascadeur motion resume <job_id> --json
mcp-for-cascadeur motion review <job_id> --json
mcp-for-cascadeur motion select <job_id> candidate-02 --json
mcp-for-cascadeur motion import '<existing pipeline run>' --json
```

Cancellation is cooperative. Generation is observed to terminal completion,
completed candidates and required conversions are retained, and the GPU is
released only after proven idle. An in-flight native stage must stop safely
before the request is reported cancelled. This can take longer than clicking
Cancel. Uncertain native outcomes remain fenced and are not replayed.

For an interrupted host process using the lower-level CLI:

```powershell
uv run --no-project --with numpy --with imageio-ffmpeg python -B integrations/kimodo/pipeline.py resume --output '<MCP workspace>\kimodo\walk-candidates'
```

Resume verifies pinned input/output hashes and reuses completed native scenes,
and any videos/reviews requested by that run. Panel continuation keeps video
capture and VLM review disabled. It never starts Docker, resubmits generation,
reconverts motion, or repeats an uncertain native write. An interrupted diffusion
pass cannot continue from that pass. An unknown import/render owner requires
inspection before continuing. Preserve recovery files until that outcome is known.
Cooperatively cancelled native runs can resume verified saved stages. Raw runs
retain their source artifacts without a native resume or visual-review route.

After a terminal synchronous Point-transfer error, explicit continuation can
inspect the idle native process and pinned source/plan, then create a fresh owned
transfer attempt. It reuses generated motion without another GPU request. The
old failed copy and uncertain marker remain untouched; this is a new disposable
attempt from source, not rollback or replay. Lost transport without a returned
callback error, an active owner or a recovery blocker remains refused.

Individual stages are available through `workflow.py capture`, `workflow.py
transfer`, `native_import.py status` and `motion_review.py --help`. The
[Point transfer procedure](POINT_RETARGET.md) works on Indie; the optional
[native FBX retarget route](CASCADEUR.md) requires Pro.

## 12 GB memory and motion diagnostics

The 4-bit encoder, generator, native viewport and other applications share VRAM.
Generate sequentially, finish CPU conversions, prove idle state and release the
generator before local visual inference. Configure/unload that separate visual
runtime explicitly. Profile actual memory for its model/context/frame budget;
no arbitrary local model/evidence combination is guaranteed to fit 12 GB.

`compare_motion.py` compares SOMA NPZ clips without inference. It reports root
travel and foot speed under predicted-contact and height masks, including their
coverage. These are source-skeleton diagnostics. Measure the native target result
separately; Joint centers, contact labels, foot rolling and mesh penetration do
not describe the same thing. Review actual motion before use in a game.

## Licenses

The integration code is MIT. Kimodo, MotionCorrection, Blender, skeleton/skin
assets and encoder/model weights retain their upstream licenses and notices.
Keep those notices when redistributing compatible source, images or model assets.
