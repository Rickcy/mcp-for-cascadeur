# Package validation: 0.2.0

Package acceptance used Windows and an independent CPython 3.12.14 environment
outside the checkout. `PYTHONPATH` was unset, and the imported package was
confirmed to reside in that environment's `site-packages`.

## Results

| Check | Result |
| --- | --- |
| Wheel and source archive | `uv build` succeeds. |
| Dependency consistency | `uv pip check` succeeds for the development environment and fresh wheel environment. |
| Installed CLI | Module help succeeds; `tools --json` returns 96 tools. |
| Isolated launcher | `uvx --isolated --from <wheel> mcp-for-cascadeur tools --json` returns 96 tools. |
| Official MCP SDK | Initializes version 0.2.0 and discovers 95 enabled tools with advanced Python disabled. |
| Resources and prompts | Three resources and one prompt; guide/contracts read successfully. |
| Scene resource transport | Reads through an isolated HTTP fixture, without contacting Cascadeur. |
| Input validation | Boolean input for an integer frame is rejected before any HTTP submission. |
| Runner timeout | An emulated native timeout retains its request ID, reports an unknown outcome, recommends `transaction_status`, and does not retry the edit. |
| Installation side effects | Help/discovery create no user configuration; SDK acceptance uses disposable settings. |
| Package contents | All 26 packaged Python modules match the checkout and installed wheel byte for byte, including the final Static-getter, selection and native-Euler fixes. |
| Installed native runtime | All 20 host/runtime modules match the checkout byte for byte; live diagnostics report adapter 0.2.0. |
| Published Git installation | Isolated `uvx` discovery from commit `a4d4a5b765aa13fb035459be68dd3f9634fb898a` returns 96 tools without setup or configuration changes. |

The HTTP fixture validates the packaged protocol and resource wiring. It is not
native animation evidence. Actual Cascadeur acceptance is recorded separately
in [validation.md](validation.md).

The wheel contains this project's Python modules and distribution metadata.
It contains no vendor binaries, native scenes, FBX files, screenshots,
`__pycache__`, dependency source trees or hardcoded local user paths. Metadata
declares Python `>=3.11`, MIT licensing and Eugene Kudenko as author; the MIT
license is included under the wheel's license directory.

## Repeating package acceptance

Build and install the wheel into a fresh environment outside the checkout,
then use that environment's Python to run:

```text
python <checkout>/validation/package_smoke.py --expected-tools 96
```

The script rejects imports from outside its Python environment. It runs the
official SDK against a stdio child and its own loopback HTTP fixture, with
temporary settings. It does not launch, inspect or edit the native application.

CI explicitly lists the affected feature and integration test modules, runs
Ruff over package/tests/examples/validation, builds the distribution, checks
isolated `uvx` discovery and repeats the fresh installed-wheel SDK check on
Windows/Linux with Python 3.11/3.12. These CI jobs do not install Cascadeur and
therefore do not establish native application support on Linux.

## Normal installation

An end user can run setup directly from a Git revision through `uvx`; no manual
checkout or virtual environment is required. Setup records a Git installation's
exact commit for the generated client command, and copies the host runtime to
stable per-user storage independently of the disposable uv cache. Starting the
bridge automatically remains the explicit `--install-startup` setup option.

Developer editable installations use `setup --local-tool`. A local wheel path
is not written into another user's client configuration; remote wheel and Git
installations retain their reproducible remote source.
