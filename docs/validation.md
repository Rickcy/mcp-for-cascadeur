# Validation

Local environment: Windows, Cascadeur 2026.2.3 (build 44356), embedded Python
3.11.0, external Python 3.12.14. Native checks use disposable copies of the
application's installed samples. No sample assets are distributed here.

## Python and distribution checks

The 132 feature tests cover contracts, HTTP transport, official MCP SDK stdio,
settings, installation, scene tools, animation, motion math and rig profiles.
They include invalid nested inputs, unknown write outcomes, swallowed native
callback errors, metadata preservation and timeline allocation order.

The wheel was installed into a fresh environment outside the checkout. Help,
tool discovery, SDK initialize/resources/argument validation and
`uvx --isolated --from <wheel>` succeeded without access to repository imports.
The registry contains 50 tools; the server exposes 49 when Python is disabled.
`uv pip check`, Ruff and wheel/sdist builds pass.

CI runs Python checks on Windows and Linux with Python 3.11 and 3.12. It does not
install Cascadeur and is not evidence of native support on Linux.

The initial branch's four hosted matrix jobs completed successfully. Motion
planning, reflection and the documented action catalog also work without a
connection to the native application.

## Native checks

| Check | Observed result |
| --- | --- |
| Startup and diagnostics | Managed hook starts the installed runner after a scene opens; health and live process/scene status respond. |
| Capability discovery | Native build, license flags, tool names, interpolation modes and viewport modes returned by the installed application. |
| Cube scene via MCP SDK | Read controls, snapshot, sample seven motion poses, extend animation, write/read back, set playback bounds, frame the camera, capture PNG, save and reopen `.casc`. |
| Cube pose readback | Maximum position error approximately `0.000011` scene units. |
| Cascy controllers | All 43 native Point controllers read and written on the original frame; maximum position error approximately `0.000933`. |
| Cascy timeline extension | Three poses at frames 0, 12 and 24, with a rigid group translation of 15 scene units; native readback matches within `0.001`. |
| Cascy roundtrip | Seven sampled poses, selected-pose restoration, PNG, native save/reopen and all 43 Point controls checked again after loading. Maximum reopened error `0.001039`, against an explicit `0.002` scene-unit tolerance. |
| Rig profiles | Body group bound to 43 native IDs; profile and grouped pose read both work after reopening the saved scene. |
| Finger controller | Right thumb Box controller rotated 15 degrees in local space, read back, then restored to its original quaternion. |
| FBX export | Native animation export produced a fresh 677,664-byte FBX under the tested installation's export entitlement. Import into another application is not covered by this check. |
| Forced disk reload | A uniquely named copy of the saved Cascy file loaded through the native parser; frames 0, 12 and 24 read back with maximum position error `0.000933`. |

Sampling that Cascy motion at seven frames produced a maximum solver residual of
approximately `0.001039` scene units, just above the default `0.001` readback
tolerance. The native smoke example explicitly uses `0.002` for Cascy and records
the chosen tolerance. Changing this parameter affects the reported match flag,
not the written targets or solver behavior.

The first development test exposed an invalid native allocation order: creating
fixed keys outside stored frames could abort the application. Extension now
allocates a terminal section, resizes native data and generates the update before
using key helpers. A separate native Cascy test established that the requested
last written frame needs a following terminal frame for the rig solver. The
allocation includes that terminal frame; playback can still end at the requested
last frame. Regression tests preserve this ordering and boundary.

Native readback verifies requested channels, not every solver-coupled control,
contact, physical plausibility or artistic quality. Arbitrary rigs and every
interpolation mode have not been accepted. Whole-rig retiming is deliberately
outside the structured retiming tool's supported scope.

## Repeating native acceptance

With Cascadeur open on a scene and setup complete:

```powershell
uv run python examples/native_smoke.py --sample Cube
uv run python examples/native_smoke.py --sample Cascy
```

Each run creates a new validation directory inside the configured workspace,
opens a copy of the selected installed sample, and records a JSON report beside
the `.casc`, selected-pose JSON and PNG outputs. It never edits the vendor sample.
The saved scene is copied to a new path before loading, so an already-open tab
cannot substitute for reading the saved file.
