# MCP for Cascadeur

Inspect native rigs, author body and finger animation, preserve curve metadata,
retime character keys and recover interrupted edits through MCP. Structured tools
cover generic controllers and validated native character semantics. Optional
Python execution exposes the rest of Cascadeur's API. The registry contains 103
structured tools and the optional Python tool.

The stdio process uses the official MCP Python SDK. Native operations run through
the script runner shipped with Cascadeur, on its main thread. Setup copies this
project's host modules to a persistent user directory; cleaning the uv cache does
not remove the host bridge. It installs no Python dependencies or replacement Qt
libraries inside Cascadeur.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Cascadeur
first. This release requires a build containing **Scripts > MCP > Start script
server**. Native development and acceptance use Windows Cascadeur **2026.2.3**.
Other operating systems have Python contract checks, not native acceptance.

From PowerShell or another terminal:

```powershell
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur mcp-for-cascadeur setup --install-startup
```

This discovers Cascadeur, installs a startup hook, and adds an MCP entry to Codex.
Existing client settings are preserved and backed up. No clone or manual virtual
environment is required. Git installs are recorded in the client configuration
with their exact commit hash. The package is not published to PyPI yet.

Restart Cascadeur, **open or create a scene**, then reconnect the MCP server in
Codex. The welcome screen alone does not emit the scene-idle events needed to
execute scripts. A normal terminal check is:

```powershell
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur mcp-for-cascadeur doctor
```

Useful setup options:

```powershell
# Enable the full Python API tool as well as structured tools.
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur mcp-for-cascadeur setup --install-startup --allow-python

# Choose the application and the directory allowed for scene, image and JSON files.
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur mcp-for-cascadeur setup --cascadeur-home "C:\Path\Cascadeur" --workspace "D:\Animation" --install-startup
```

Without `--install-startup`, setup generates `start_bridge.py` and prints its
location. Execute that file in Cascadeur's Python console after opening a scene.
Use `--client none` when managing client settings yourself. For Claude Desktop,
Cursor or another stdio client, add:

```json
{
  "mcpServers": {
    "cascadeur": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Rickcy/mcp-for-cascadeur", "mcp-for-cascadeur"]
    }
  }
}
```

Use an absolute path to `uvx` if your desktop client cannot find it. Settings and
the stable host runtime live in the user's application-data directory under
`mcp-for-cascadeur`. `MCP_FOR_CASCADEUR_HOME` overrides this location.

For a downloaded wheel, keep the external tool installed and use its registered
executable instead of a temporary package source:

```powershell
uv tool install "C:\Downloads\mcp_for_cascadeur-0.2.0-py3-none-any.whl"
uvx mcp-for-cascadeur setup --install-startup --local-tool
```

## Animation workflow

The [animation workflow roadmap](docs/roadmap.md) describes the planned Uthana
integration, Gemini Flash / Qwen-VL review, and native editing and validation.

1. Read `app_status`, `app_capabilities`, `objects_list` and `tracks_list`.
2. Use `object_inspect` to find actual body, hand and finger channels. IDs avoid
   ambiguous names. Bind convenient names using `rig_profile_create`.
3. Save a native scene copy and create a durable `checkpoint_create` before a
   sequence of edits. Use `character_edit_readiness` to inspect exact blockers.
4. Read poses, build keyframes, and preview a plan with `motion_sample` or
   `motion_mirror`. Plans contain ordinary JSON and do not edit the scene.
5. Use `character_offset_preserving_curves` for existing character keys,
   `character_retime` for synchronized character timing, or `pose_apply_sequence`
   for generic controller authoring. Timeline extension is explicit. Readback reports
   target error and solver adjustments; a successful write can still alter other
   controls. `position_tolerance` and `rotation_tolerance` explicitly set readback
   acceptance (scene units and radians); they do not change the native solver.
6. Sample inbetweens, capture body and hand views, and reopen a saved `.casc`.
   Check contacts, silhouettes and exported animation in the receiving app.

Tool groups:

| Area | Tools |
| --- | --- |
| Discovery | Application/build/license capabilities, object and behavior inspection, hierarchy, native tool signatures, public API introspection |
| Scene | Tabs, loading/saving, selection, workspace file checks |
| Animation | Controller poses and batches, relative offsets, keys, interpolation, timeline bounds, typed behavior-property writes |
| Motion plans | Easing, shortest-arc quaternion SLERP, reflection and object mapping |
| Rig profiles | Named groups for body controls, hands, fingers or props; scene/topology validation |
| Characters | RigInfo ownership, semantic body groups, authored finger rotations, curve-preserving edits and synchronized retiming |
| Native jobs | Video Mocap, pose-based Motion Generation, Animation Unbaking and AutoPhysics, with durable progress and recovery |
| Recovery | Persistent pre-write journal, verified rollback, whole-model checkpoints, durable character poses and recovery fences |
| Review and exchange | Native playback observation, camera framing, fresh PNG images, FBX import/export and staged hash-checked publication |
| Advanced | Documented native actions and optional Python with `csc`, `app`, `view`, `scene`, `workspace` |

The server exposes `cascadeur://guide`, `cascadeur://tools`, `cascadeur://scene`
and an `animate_character` prompt.
Run `mcp-for-cascadeur tools --json` from an installed tool environment to inspect
all parameter contracts.

## Native animation jobs

The following structured tools run on Windows Cascadeur **2026.2.3, build 44356**
with a validated Cascy rig. They check the installed build and native scene
before dispatch. Other builds and arbitrary humanoid rigs need separate
acceptance; similar controller names do not establish compatibility.

| Start tool | Operation |
| --- | --- |
| `motion_generation_start` | Generate native motion from existing keyed endpoint poses and a trajectory. This is pose-based generation, not a text-prompt endpoint. |
| `mocap_start` | Process a workspace MP4 or an existing native reference plane. The official Mocap model must be installed; downloading it requires `allow_model_download=true`. |
| `unbaking_start` | Convert baked/FIXED character motion to classic editable intervals, including CLAMPED_BEZIER, and report observed changes. |
| `autophysics_start` | Calculate and apply native AutoPhysics with the explicit `all_frames` or `only_keys` choice, then measure the result. |

Each start returns a `job_id`. Poll `native_job_status`, or use
`native_jobs_list`, until a terminal outcome is observed. Progress reports native
stages and available observations; it remains indeterminate when the application
provides no reliable fraction. A queued action or closed dialog is not completion.

Jobs preserve a hashed source `.casc` and use a separate working scene before
editing. The durable receipt owns the scene and blocks competing writes until
completion or verified recovery. Final checks protect static data, unrelated
objects and animation outside the declared interval before publishing a fresh
output file. They do not certify artistic quality or physical plausibility.

`native_job_cancel` requests cancellation. The write fence remains until native
work has stopped and restoration is verified. An uncertain stop can require an
application restart followed by `native_job_recover`. Status and list tools can
read checksummed local journal records when Cascadeur or its transport is
unavailable; this offline observation does not restart or replay native work.

Native jobs accept at most **512 stored frames**, including the following
terminal guard frame. Existing character pose/curve edits retain their
121-frame contract, and `character_retime` accepts 2–8 synchronized keys.
Unbaking can produce more keys than this retimer accepts.

For a prepared scene with endpoint keys at frames 0 and 30 and a stored successor,
the development smoke example creates a fresh scene copy and records job evidence:

```powershell
uv run python examples/native_job_smoke.py motion_generation --scene "C:\Animation\posed-walk.casc" --first 0 --last 30
```

## Scope

This is an initial development release. It does not guarantee finished animation
from a prompt. Generic menu actions report dispatch; the affected native tool,
selection and license still determine the result. Native jobs add their own
operation-specific completion and protected-state checks.
General Python execution has the same privileges as Cascadeur and is optional.

`tracks_retime` handles standalone transforms. `character_retime` uses authored
Point positions, validated finger-local rotations and native settings; driven
Joint outputs are solved and checked. Its bounded contract requires synchronized
keys and supported interpolation. Curve-preserving tools retain section metadata
and protect unrelated channels; solver-coupled movement is reported separately.
CLAMPED_BEZIER remains its native interval type through these edits, classic
retiming and checkpoint restoration. Exact metadata preservation does not promise
the edited trajectory's original shape; custom spatial tangents remain excluded.

Checked model edits persist a full bounded checkpoint before mutation. Failure
triggers checked Undo and a guarded manual restoration fallback. A failed or
interrupted recovery blocks writes across native process restarts until the full
recorded model is verified. Read-only diagnostics and saving a new quarantine copy
remain available. There is no force-unlock operation.

After a timeout, inspect `transaction_status` using the returned request ID and
read the actual scene before retrying. The native runner has a 30-second request
deadline. Arbitrary external Python side effects and opaque asset payload edits
are outside model rollback. Keep the vendor HTTP runner bound to loopback.

See [capability contracts](docs/capabilities.md) and [measured native acceptance](docs/validation.md)
for bounds, rejected curve states and tested behavior.

## Updating and removal

Repeat setup with `uvx --refresh --from <source> ... setup`, then restart both
Cascadeur and the MCP client. Files on disk do not reload modules already imported
into the native process. Modified or unrelated startup hooks are preserved.

`mcp-for-cascadeur uninstall` removes this project's recorded startup hook and
Codex entry. It keeps scenes, workspace files and settings. A permanent tool
installation, if used, can then be removed with `uv tool uninstall mcp-for-cascadeur`.

## Development

```powershell
uv sync --extra dev
uv run mcp-for-cascadeur --help
uv run ruff check src tests examples
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [native API notes](docs/native-api-notes.md)
and [validation results](docs/validation.md) for test scope and native evidence.

Code is MIT licensed. Cascadeur and its installed assets retain their own licenses.
