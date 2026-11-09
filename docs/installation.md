# Installation and configuration

## Requirements

- Python 3.11+ and uv for the external MCP process.
- Cascadeur with **Scripts > MCP > Start script server**; native jobs target
  Windows 2026.2.3 build 44356 and the validated Cascy sample.
- An MCP client capable of launching a stdio server.
- A dedicated workspace for scenes, exports, images and recovery records.

The external package does not install replacement Qt/Python packages inside
Cascadeur. Native handlers use the application's own runtime on its main thread.

## Current checkout and Smart Motion

The Smart Motion panel and `kimodo_*` MCP tools are newer than the published
`release/0.1.0` tag. Until a release containing them is published, install the
current prepared checkout with its animation dependencies:

```powershell
uv tool install --force ".[animation]"
mcp-for-cascadeur setup --install-startup --local-tool --allow-python
mcp-for-cascadeur motion status --json
```

Setup installs **Scripts > MCP > Smart Motion** and records the external tool
Python used by its worker service. Restart Cascadeur after installation. The
panel uses the Qt already shipped with Cascadeur; do not install PySide or model
dependencies into the application's Python environment.

Configure the prepared Kimodo checkout and inputs once:

```powershell
mcp-for-cascadeur motion configure --checkout 'D:\Projects\mcp-for-cascadeur' --template-motion 'D:\Animation\soma-template.json'
```

These paths refer to existing assets; configuration does not download weights or
accept model licenses. The panel calibrates the chosen character from its native
scene, saves an isolated source, and requires trusted Python. The SOMA template
is needed for body anchors. Legacy API callers may still configure an explicit
calibration and target template. Raw generation needs the prepared checkout and GPU
runtime but does not require a running Cascadeur or native calibration.

Open the menu entry or call `mcp-for-cascadeur motion panel`. Prompt, anchors,
generation settings, saved results and runtime controls use the same
service as MCP. See [Smart Motion](smart-motion.md) and the
[shared job workflow](workflow.md#smart-motion-panel-mcp-and-cli).

## Earlier tagged release

The existing tag installs the earlier native bridge and standalone visual
reviewer. It does not include Smart Motion or the `kimodo_*` job tools:

```powershell
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur@release/0.1.0 mcp-for-cascadeur setup --install-startup
```

Setup preserves and backs up existing client settings. The native host modules
are copied to stable application data, so removing the uv cache does not remove
the bridge. Restart Cascadeur and open a scene before connecting the MCP client.
Client tool calls allow 300 seconds for visual review; setup preserves a longer existing timeout.

Useful options:

```powershell
# Full-privilege Python tool in addition to structured native tools.
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur@release/0.1.0 mcp-for-cascadeur setup --install-startup --allow-python

# Explicit application and workspace.
uvx --from git+https://github.com/Rickcy/mcp-for-cascadeur@release/0.1.0 mcp-for-cascadeur setup --cascadeur-home 'C:\Path\Cascadeur' --workspace 'D:\Animation' --install-startup
```

Without `--install-startup`, setup prints a generated `start_bridge.py`; execute
that file in Cascadeur's Python console after opening a scene. `--client none`
keeps MCP client configuration under your control.

## Generic stdio client

```json
{
  "mcpServers": {
    "cascadeur": {
      "command": "uvx",
      "args": [
        "mcp-for-cascadeur"
      ]
    }
  }
}
```

Use the absolute `uvx` path when a desktop client does not inherit the terminal's
PATH. This example uses the permanently installed tool with the `animation`
extra. A temporary `uvx --from` installation of the earlier tag does not expose
the current panel or Kimodo MCP jobs. Keep the native HTTP runner on loopback. The MCP stdio process supplies the
structured tools; the native runner itself is not an authenticated network service.

## Checkout or wheel

```powershell
uv tool install --force ".[animation]"
mcp-for-cascadeur setup --install-startup --local-tool --allow-python
```

Or install the wheel built from the current checkout with `uv tool install` and
its `animation` extra.
This avoids relying on a temporary source path. Installation from Git/wheel does
not imply the package is published to PyPI. Optional Kimodo scripts require the
prepared source/model checkout and are documented [separately](../integrations/kimodo/README.md).

## Private settings and review credentials

Application data lives under `mcp-for-cascadeur` in the user's local application
data directory. `MCP_FOR_CASCADEUR_HOME` overrides it. Scene/output paths are
restricted to the configured workspace. Provider configuration and credentials
belong to private application data or the process environment, outside Git.

```powershell
mcp-for-cascadeur review configure --provider gemini --model gemini-flash-latest
mcp-for-cascadeur review key set gemini
mcp-for-cascadeur review status --json
mcp-for-cascadeur review key delete gemini
```

`key set` prompts without echo. On Windows the stored secret is protected by
DPAPI for the current user. Other platforms use environment-supplied credentials.
The status command reports configuration and credential availability, not the key.
Do not put credentials in client arguments, source, committed examples, Docker
images, prompts or animation reports. See [provider settings](workflow.md#visual-review-providers).

## Connection checks

Timestamped-frame review requires the `animation` extra for its CPU FFmpeg
decoder. A persistent tool installation keeps those dependencies available to
the configured MCP process:

```powershell
uv tool install --force ".[animation]"
mcp-for-cascadeur setup --install-startup --local-tool --allow-python
```

This installs the review client and decoder. The local Qwen server and its model
are provisioned separately; its exact served model ID is configured with
`review configure`.

```powershell
mcp-for-cascadeur doctor
mcp-for-cascadeur tools --json
```

Read `cascadeur://guide`, `cascadeur://workflow`, `cascadeur://capabilities`
and `cascadeur://scene`. Check `app_status`, `app_capabilities` and
`recovery_status` before editing. If transport is unavailable, open a scene and
start its script server. Do not resubmit a timed-out write: inspect its request
ID or durable receipt first. Restart Cascadeur after updating imported modules.

## Removal

`mcp-for-cascadeur uninstall` removes recorded integration entries, including
the owned Smart Motion menu, while retaining user scenes, models and settings.
Wait for any Smart Motion worker to reach a safe terminal state before updating
or removing its installed tool. Remove a permanent uv tool with
`uv tool uninstall mcp-for-cascadeur`. Model/cache cleanup is a separate explicit
operation; never delete the animation workspace to uninstall the bridge.
