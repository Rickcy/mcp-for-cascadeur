# Native Cascadeur import

This route requires **Cascadeur Pro** for its native Retargeting Copy/Paste
actions. It cannot complete on an Indie license. Indie uses the separate
`point_retarget.py` route through the public Python API. Paths below are examples;
use the actual workspace and installed Cascadeur path from the MCP configuration.

`cascadeur_bridge.py` imports a completed Kimodo generation. It does not start
Docker, download models or call a model from the native application thread.
Call the Docker service first, wait for a successful job, then use the local
FBX and its generation manifest.

## Command-line entry point

Use `native_import.py` from the prepared repository checkout.
It uses `RunnerClient`'s loopback HTTP transport and the application's native
`/run` entry point. It does not wrap scene creation, switching or closure in a
`python_execute` transaction owned by the original scene. Instead, the bridge
records its own isolated scene copies and durable stage receipts.

```powershell
$cli = '.\integrations\kimodo\native_import.py'
python -B $cli prepare `
  --manifest 'D:\Animation\kimodo\generated.json' `
  --fbx 'D:\Animation\kimodo\generated.fbx' `
  --workspace 'D:\Animation\kimodo' `
  --template 'D:\Animation\kimodo\Kimodo.casc'

# Use the receipt_path returned above. Execute one stage and inspect the UI.
python -B $cli step --receipt '<absolute session.json path>'
python -B $cli status --receipt '<absolute session.json path>' --native
```

`--cascy-template` optionally overrides the installed Cascy sample. `step --stage
open_target` can name the exact next stage; a different or repeated stage is
rejected. `--expected-pid` pins the intended Cascadeur process; the first native
stage also records that PID in the session for later calls. All requests use a
maximum 35-second HTTP timeout and no automatic retry.

Before each native write, the CLI checks a stationary playhead with two separate
read requests, then rechecks the recovery, native-job and owned-playback guards
on the application thread immediately before dispatch. An active core recovery
transaction or disabled trusted Python setting blocks the operation. Existing
MCP recovery locks are never cleared by this integration.

`complete --receipt '<path>'` advances all remaining stages as separate native
requests, printing each result. It stops at the first failure. Use `step` during
the first run on an installation so each operation can be observed independently.

An uncertain transport result leaves `.native-transport.json` and blocks further
steps even if the native receipt still reports the preceding phase. `status
--native` only inspects it. If the receipt proves that the uncertain stage did
finish successfully, an explicit `status --native --acknowledge-completed` can
record that inspection and permit the following stage. It never repeats the
uncertain write. A remaining `.native-import.lock` belongs to another CLI process
or an interrupted invocation and requires inspection of that process first.
Successful completion removes both temporary transport files.

The current route accepts SOMA motion at 30 FPS and native scenes in centimetres.
The installed Cascy template has that timebase. `scene_settings()` reads it from
the saved `.casc` container; Windows' bundled `tar.exe` handles the Zstandard ZIP
compression unsupported by the embedded Python's `zipfile`.

## Stages

Import the module in Cascadeur's script-server execution environment:

```python
import importlib.util

path = r"D:\Projects\mcp-for-cascadeur\integrations\kimodo\cascadeur_bridge.py"
spec = importlib.util.spec_from_file_location("kimodo_cascadeur_bridge", path)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
```

`prepare` only copies files and may also run outside Cascadeur. Its arguments:

```python
result = bridge.prepare(
    workspace_root=r"D:\Animation\kimodo",
    cascy_scene=r"C:\Path\Cascadeur\samples\Cascy.casc",
    kimodo_scene=r"D:\Animation\kimodo\Kimodo.casc",
    generation_manifest=r"D:\Animation\kimodo\generated.json",
    fbx_path=r"D:\Animation\kimodo\generated.fbx",
)
receipt = result["receipt_path"]
```

Run **one native stage per script-server request**, in this order:

| Function | Result |
| --- | --- |
| `open_target(receipt)` | Opens a new Cascy working copy and checks its real Point semantics. |
| `open_source(receipt)` | Opens a separate copy of the supplied `Kimodo.casc`. |
| `import_fbx(receipt)` | Imports verified FBX animation into that source rig. |
| `retarget_copy(receipt)` | Uses Cascadeur's native Retargeting Copy for the generated interval. |
| `prepare_target_interval(receipt)` | Allocates the Cascy interval and its animation keys. |
| `retarget_paste(receipt)` | Uses native Retargeting Paste and measures Cascy root displacement. |
| `save_result(receipt)` | Saves `result.casc`, its hash, and Point values across every motion frame. |
| `reopen_prepare(receipt)` | Activates the retained source tab without closing any scene. |
| `reopen_result(receipt, callback_scene=...)` | In a separate request, closes the inactive result tab, reopens the file and compares every Point coordinate. |
| `cleanup(receipt, callback_scene=...)` | In a later request, removes disposable files and closes the inactive source tab. |

Modules need not remain in memory between calls: the receipt and unique native
tab paths identify the session. Retargeting Copy/Paste uses Cascadeur's native
clipboard, so do not run another retarget operation between those stages.
The native CLI forwards the real `scene` supplied by the vendor's scene-idle
callback. Reopen and cleanup require that context; switching the active tab
inside one callback does not make its original domain scene safe to delete.

The existing user's scene tabs are never imported into, saved, closed or used as
the target. A session retains only `before.casc`, `result.casc` and `session.json`
after successful cleanup; the original downloaded templates and generated FBX
remain caller-owned. The result stays open for review.

## Verification and failure handling

The generation manifest must name `text_encoder.kind = "llm2vec"`, include the
actual FBX SHA256, `fps = 30` and an integer `num_frames` between 2 and 240. Hash
fallback is rejected. The bridge does not infer semantic text conditioning from
the presence of a prompt field.

`inspect(receipt)` reports the durable phase, native tabs, root displacement,
saved timebase and reopen comparison. A stage writes `running:<stage>` before
dispatching its native operation. Exceptions persist `needs_review:<stage>`.
Neither state is automatically replayed; a script-server timeout can occur after
the native action has already started. Inspect the owned tabs and receipt first.
`before.casc` remains an unmodified recovery copy. Opening it as a new working
copy is safer than trying to replay an uncertain paste into the same scene.

Reopen verification covers all native Point controls at every requested frame,
with a 0.01 cm coordinate tolerance and exact semantic/topology fingerprints.
It is a persistence check, not an artistic quality score. Root translation must
exceed 1 cm. Foot contacts, loop seams, body mechanics and the requested motion
style still need review.

After reopening, use the existing `rig_semantics` and
`character_edit_readiness(frames=[30], operation="semantic_pose_offset",
roles=["left_hand"])` MCP tools to check edit readiness. An actual edit test
belongs in another disposable copy: snapshot a pose, apply a small semantic
offset, verify the resulting control position, then restore and compare it.
Do not alter a reviewed final scene just to demonstrate editability.

For playback, use the existing `playback_start(first=0, last=119, timeout=6)` and
poll `playback_status`. Accept measured `continuous_playback_observed` and
`stop_verified`; scripted frame changes do not demonstrate native playback.

## Local contract checks

Only the bridge's tests are required for its file/receipt changes:

```powershell
python -B -m unittest discover -s tests -p test_kimodo_cascadeur_bridge.py -v
python -B -m unittest discover -s tests -p test_kimodo_native_import.py -v
```

These tests do not load Cascadeur or establish that retargeting worked. Native
acceptance must execute the stages above on the installed application.
