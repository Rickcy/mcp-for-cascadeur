# Kimodo generation and visual review

This optional checkout integration runs **Kimodo-SOMA-RP-v1.2** and a real
4-bit LLM2Vec encoder in Linux GPU containers. Cascadeur stays on Windows. It
supports text generation and generation constrained by native Cascy body poses,
retains candidates, transfers each to Cascy and compares the native results with
a configurable visual reviewer.

The core MCP package supplies native tools independently of this checkout. The
scripts below are CLI orchestration over the installed native runner, not a
registered `kimodo_generate` MCP tool.

## Requirements and assets

- Docker Desktop's Linux/WSL2 engine with NVIDIA GPU support.
- The installed MCP bridge, a saved scene and trusted Python enabled.
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

## Configure visual review

Configure the reviewer through the installed CLI, outside the repository:

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

## Full text-to-motion run

Prepare a neutral Cascy calibration with `point_retarget.calibrate`, a SOMA
retarget template from `prepare_retarget_motion.py`, and the installed Cascy scene.
The calibration describes the unanimated target. The SOMA template supplies its
neutral skeleton/axes independently of the example motion it contains. Reuse
these inputs rather than recalibrating from an animated pose.

```powershell
uv run --no-project --with numpy --with imageio-ffmpeg python -B integrations/kimodo/pipeline.py run `
  --prompt 'A person walks forward naturally at a steady pace' `
  --frames 120 --samples 3 --seed 420 `
  --calibration '<neutral Cascy calibration.json>' `
  --template-motion '<SOMA retarget template.json>' `
  --template-scene '<installed samples\Cascy.casc>' `
  --output '<MCP workspace>\kimodo\walk-candidates'
```

The run directory must be new and inside the configured workspace's `kimodo`
folder. The pipeline records inputs, starts the generator, submits exactly one
job, retains each completed candidate, performs CPU conversion, releases the
idle generator, then transfers, renders and reviews the native results. The
selected candidate is activated in Cascadeur only after acceptance checks.

## Generate between captured body poses

Add these arguments to the run command:

```powershell
  --source-scene '<MCP workspace>\posed-animation.casc' --keyframes 0,30,60,90,119
```

Frame indices are zero-based existing native frames. The first/last selected
frames must span `--frames`, up to **120 frames at 30 FPS**. Choose poses from
meaningful motion phases; the integration does not automatically choose them.

Capture reads actual native body Joint transforms and all 43 Point targets
without moving the playhead. It converts proportions, coherent SOMA geometry,
units, floor origin and initial root/heading, with an exact inverse for return.
Conversion/canonical input checks run before GPU work. After transfer and
save/reopen, actual anchor position/rotation error is measured. Default anchor
acceptance is 1 cm and 5 degrees; the observed errors are retained in the report.

The result is a new standalone interval on a separate Cascy scene. Its body poses
are stored frame by frame. The source animation is preserved; finger articulation,
facial animation, attached-object physics and seamless looping are outside this
transfer contract.

## Candidates and selection

`--samples` accepts 1–4 candidates, generated sequentially with seeds
`(base_seed + index) mod 2**32`. Each retains NPZ, FBX and a manifest with pinned
input/output checksums. A common candidate index is saved after each complete
candidate, so completed results survive a later failure.

Each result transfers to its own native scene and passes full-frame rigid-target
and solver readback checks. The scene is saved/reopened and anchors are checked
when required. Native videos show front, side and three-quarter views. The
renderer observes scene updates between PNG captures, encodes on CPU and cleans
owned scratch frames only after its native worker stops. Review videos disclose
sampled/held-frame cadence and preserve the clip's duration.

The visual provider compares all candidates in one request and returns the common
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
the review/selection paths. For an interrupted host process:

```powershell
uv run --no-project --with numpy --with imageio-ffmpeg python -B integrations/kimodo/pipeline.py resume --output '<MCP workspace>\kimodo\walk-candidates'
```

Resume verifies pinned input/output hashes and reuses completed native scenes,
videos and matching reviews. It never starts Docker, resubmits generation,
reconverts motion, or repeats an uncertain native write. An interrupted diffusion
pass cannot continue from that pass. An unknown import/render owner requires
inspection before continuing. Preserve recovery files until that outcome is known.

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
coverage. These are source-skeleton diagnostics. Measure the native Cascy result
separately; Joint centers, contact labels, foot rolling and mesh penetration do
not describe the same thing. Review actual motion before use in a game.

## Licenses

The integration code is MIT. Kimodo, MotionCorrection, Blender, skeleton/skin
assets and encoder/model weights retain their upstream licenses and notices.
Keep those notices when redistributing compatible source, images or model assets.
