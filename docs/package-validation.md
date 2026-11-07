# Package validation: 0.2.0 with Uthana

Package acceptance used Windows and an independent CPython 3.12.14 environment
outside the checkout. `PYTHONPATH` was unset, and the imported package was
confirmed to reside in that environment's `site-packages`.

## Results

| Check | Result |
| --- | --- |
| Wheel and source archive | `uv build` succeeds. |
| Dependency consistency | `uv pip check` succeeds for the development environment and fresh wheel environment. |
| Installed CLI | Module help succeeds; `tools --json` returns 111 tools: 104 native definitions and seven external Uthana definitions. |
| Isolated launcher | `uvx --isolated --from <wheel> mcp-for-cascadeur tools --json` returns 111 tools. |
| Official MCP SDK | Initializes version 0.2.0 and discovers 110 enabled tools with advanced Python disabled. |
| Resources and prompts | Three resources and one prompt; guide/contracts read successfully. |
| Scene resource transport | Reads through an isolated HTTP fixture, without contacting Cascadeur. |
| Input validation | Boolean input for an integer frame and invalid cloud submission arguments are rejected before any HTTP/API submission. |
| Cloud boundary | All seven Uthana tools have `openWorldHint=true`; native tools retain `false`. Cloud discovery and a local receipt list require no credential, Uthana API request or native runner call. |
| Offline native status | A native transport failure falls back to the checksummed durable job journal and retains an unknown execution state for an unfinished job. |
| Runner timeout | An emulated native timeout retains its request ID, reports an unknown outcome, recommends `transaction_status`, and does not retry the edit. |
| Installation side effects | Help/discovery create no user configuration; SDK acceptance uses disposable settings. |
| Credentials | Windows DPAPI CurrentUser roundtrip succeeds with a test fixture; CLI refuses an echoing prompt fallback and never prints key values. Settings contain no API keys. |
| Package contents | All 34 packaged Python modules match the frozen checkout, wheel archive and fresh installed package byte for byte. |
| Installed native runtime | All 24 host/runtime modules match the frozen checkout byte for byte. Cloud modules, credentials and the Uthana SDK are excluded from this copied runtime. |

The frozen wheel is 185,781 bytes, SHA-256
`be4f0395baea6822742f383816b3387adb1ece173a5ffbf2337bc39b329f3b32`.
Package acceptance uses only a disposable loopback HTTP fixture and clears
inherited Uthana/Gemini credentials from its SDK child environment.

The HTTP fixture validates the packaged protocol and resource wiring. It is not
native animation evidence. Actual Cascadeur acceptance is recorded separately
in [validation.md](validation.md).

The wheel contains this project's Python modules and distribution metadata.
It contains no vendor binaries, native scenes, FBX files, screenshots,
`__pycache__`, dependency source trees, `.env` files, DPAPI credential records or
hardcoded local user paths. The external dependency range is `uthana>=0.8,<0.9`;
the accepted SDK is 0.8.0. It is not installed into Cascadeur. Metadata
declares Python `>=3.11`, MIT licensing and Eugene Kudenko as author; the MIT
license is included under the wheel's license directory.

## Repeating package acceptance

Build and install the wheel into a fresh environment outside the checkout,
then use that environment's Python to run:

```text
python <checkout>/validation/package_smoke.py
```

The script rejects imports from outside its Python environment. It runs the
official SDK against a stdio child and its own loopback HTTP fixture, with
temporary settings. It derives the expected count from the installed native and
external tool manifests; `--expected-tools 111` can additionally pin that count.
It does not launch, inspect or edit the native application or call a cloud API.

CI explicitly lists the affected feature and integration test modules, runs
Ruff over package/tests/examples/validation, builds the distribution, checks
isolated `uvx` discovery and repeats the fresh installed-wheel SDK check on
Windows/Linux with Python 3.11/3.12. These CI jobs do not install Cascadeur and
therefore do not establish native application support on Linux.

## Prior published release

Commit `a4d4a5b765aa13fb035459be68dd3f9634fb898a` was accepted separately with
96 CLI definitions and 95 enabled SDK tools. Its 26 packaged Python modules and
20 installed host/runtime modules matched that checkout byte for byte. Isolated
Git-based `uvx` discovery succeeded without setup or configuration changes.
Those counts describe that earlier release, before native jobs and Uthana tools.

## Normal installation

An end user can run setup directly from a Git revision through `uvx`; no manual
checkout or virtual environment is required. Setup records a Git installation's
exact commit for the generated client command, and copies the host runtime to
stable per-user storage independently of the disposable uv cache. Starting the
bridge automatically remains the explicit `--install-startup` setup option.

Developer editable installations use `setup --local-tool`. A local wheel path
is not written into another user's client configuration; remote wheel and Git
installations retain their reproducible remote source.
