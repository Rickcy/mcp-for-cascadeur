# Smart Motion in Cascadeur

Smart Motion creates one animation on the character chosen in the open scene.
The panel uses the same durable external job service as the Kimodo MCP tools.
Generation runs outside Cascadeur; the native QtQuick window stays responsive.
Closing the panel hides it and does not cancel an accepted job.

## Open the panel

Install the current checkout and register the native menu:

```powershell
uv tool install --force ".[animation]"
mcp-for-cascadeur setup --install-startup --local-tool --allow-python
```

Restart Cascadeur and choose **Scripts > MCP > Smart Motion**. An assistant can
also call `smart_motion_panel`. The published `release/0.1.0` tag predates this
panel; these instructions apply to the current checkout.

The **Сервис / Service** tab needs the prepared project directory. Generation
between existing poses also needs the SOMA skeleton template. Docker, model
weights and the local text encoder must already be prepared; see
[Kimodo setup](../integrations/kimodo/README.md).

## Create an animation

1. Open the scene containing your character and select its controls. On
   **Создать / Create**, check the **Персонаж / Character** field. Choose the
   intended character explicitly if the scene contains several rigs.
2. Select the desired timeline range and press **Взять с таймлайна / Use timeline**.
   Its first and last frames are inclusive. With no usable range selection, the
   panel reads the scene's playback interval; the two fields remain editable.
3. Describe the movement and press **Создать анимацию / Create animation**.

The panel saves a source copy containing current edits, calibrates the selected
rig and generates one result. The retained source copy becomes the current
document through native Save As; the original file is preserved. Transfer uses
a separate working scene and the finished interval starts at frame 0. For
example, selecting frames 50–409 produces a 360-frame result, numbered 0–359.
This route generates at 30 FPS.
The saved target scene must also use 30 FPS and centimetres; unsupported native
timebases are rejected before GPU generation rather than silently resampled.

Panel jobs do not request Gemini or another VLM, capture review videos, choose
between several generated animations or display visual scores. They retain
native geometry, required-anchor and save/reopen checks. After completion, the
saved animation opens for normal Cascadeur editing.

## Preserve key poses

Enable **Движение между ключевыми позами / Move between key poses**. The beginning
and end of the chosen range are included automatically. Use
**Взять ключевые кадры / Use keyframes** to add existing keys from that
character's tracks, or place the playhead on an important pose and press
**Добавить текущую позу / Add current pose**. The text field allows manual
adjustment when needed.

A text prompt is optional in this mode. The captured constraints contain the
actual body poses, not just frame numbers.
The pipeline converts them into the generator's skeleton and compares the
returned native positions and rotations after save/reopen. Failed required
anchor checks prevent automatic acceptance. Finger and face animation are
outside this body-generation contract.

## Choose or map a skeleton

The target comes from the scene rather than a fixed Cascy template. Mapping uses
the native character's Joint, Point and RigAdditionalInfo membership. UE5,
UEFN and other recognizable humanoid names are resolved to common body roles.
The character's original names and native proportions are retained.
Use a suitable anatomical reference at frame 0 for calibration. An arbitrary
posed or crouched first frame is not a substitute for the rig's neutral reference.

If a custom humanoid cannot be recognized, open **Настроить скелет / Set up
skeleton** and assign each required body role to the correct Joint. Missing or
ambiguous bindings are shown before generation. A plain imported bone hierarchy
must first become a usable Cascadeur character with its native Point controls;
changing names alone does not supply a rig. Nonhumanoid anatomies need their own
mapping and transfer implementation.

## Long animations and advanced controls

There is no fixed 120- or 1800-frame total-duration limit. Kimodo generates long
clips through its own sequential continuation, using inference segments of at
most 240 frames. Body-history constraints, root translation and heading connect
the segments; their lengths add up to the exact requested total. The segment
size bounds attention memory on the 12 GB setup and does not shorten the clip.
Very long requests still consume time, GPU/host memory and disk space.

**Дополнительные параметры / Advanced** contains seed, diffusion steps and text
and pose guidance. It also exposes **Только файлы NPZ / FBX / Files only** for raw
export without native transfer. The service tab holds the port, explicit waiting
timeout and Docker maintenance controls. The supported waiting timeout is
30–86400 seconds; a timeout records an uncertain outcome and does not repeat the
accepted generation.

## Results and recovery

**Результаты / Results** retains job history, native checks and saved paths.
Choose **Открыть анимацию / Open animation**, **Воспроизвести / Play** or
**Папка результата / Output folder**. Playback is owned by this integration;
**Остановить / Stop** stops that playback. Technical logs are optional.

Progress reports measured backend counts and actual stages. Some native stages
remain indeterminate. During Point transfer the counter reports the actual
verified cursor, such as “Перенесено 20 из 540 кадров”, after each completed batch.
Cancellation waits for an in-flight GPU/native operation
to reach a safe boundary. **Продолжить после сбоя / Continue after failure**
reuses verified retained stages without resubmitting generation. It preserves
the panel's no-video/no-VLM behavior. An unknown native write must be inspected
before continuation; the panel does not clear a recovery fence.

If a synchronous native transfer returned an error, explicit continuation can
create a fresh isolated copy after live idle/process and immutable-source
inspection. It reuses the generated motion. The failed working copy and its
uncertain receipt remain untouched for inspection; this is a new attempt from
the saved source, not a claimed rollback or replay of the failed write. A live
worker, unresolved transport loss or recovery blocker prevents this route.

The lower-level MCP/CLI still provides multiple candidates and optional visual
review for assistant-driven workflows. Those options are not part of the
panel. See [the API workflow](workflow.md#smart-motion-panel-mcp-and-cli).
