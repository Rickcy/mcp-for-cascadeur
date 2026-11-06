# MCP for Cascadeur

Inspect rigs, edit controller animation and review the result through MCP.
There are 49 structured tools, plus optional Python execution for the rest of
Cascadeur's API. Control groups are user-defined and work with inspected native
rigs rather than one hard-coded character layout.

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

## Animation workflow

1. Read `app_status`, `app_capabilities`, `objects_list` and `tracks_list`.
2. Use `object_inspect` to find actual body, hand and finger channels. IDs avoid
   ambiguous names. Bind convenient names using `rig_profile_create`.
3. Save a native scene copy. `pose_snapshot` captures a selected pose, while
   `scene_save` preserves a whole clip.
4. Read poses, build keyframes, and preview a plan with `motion_sample` or
   `motion_mirror`. Plans contain ordinary JSON and do not edit the scene.
5. Apply `pose_apply_sequence`. Timeline extension is explicit. Readback reports
   target error and solver adjustments; a successful write can still alter other
   controls.
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
| Recovery | Selected-pose snapshots in memory or JSON; explicit restore and readback |
| Review and exchange | Camera framing, viewport modes, fresh PNG images, FBX import/export with native license checks |
| Advanced | Documented native actions and optional Python with `csc`, `app`, `view`, `scene`, `workspace` |

The server also exposes `cascadeur://guide` and `cascadeur://tools` resources.
Run `mcp-for-cascadeur tools --json` from an installed tool environment to inspect
all parameter contracts.

## Scope

This is an initial development release. It does not guarantee finished animation
from a prompt. Physics, AutoPosing and retargeting menu actions report dispatch;
the affected native tool, selection and license still determine the result.
General Python execution has the same privileges as Cascadeur and is optional.

Structured retiming currently supports standalone, unparented Basic/Transform
objects. It rejects constrained rig tracks rather than treating derived global
channels as independent authored channels. Use inspected native actions or Python
for rig-specific retiming. Pose writes support native Transform channels, including
controller and finger channels; solvers may adjust requested targets.

Pose snapshots do not preserve curve metadata or an entire clip. Native Undo is
not advertised as a verified rollback mechanism. After a timeout or lost native
connection, the write outcome is unknown: read back before retrying a relative
edit or action. Keep the vendor HTTP runner bound to loopback.

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
