# Contributing

Use Python 3.11 or newer and `uv sync --extra dev`.

Run tests for the feature being changed and its immediate integrations. For a
native edit, add a disposable-scene acceptance check as well as contract tests.
The hosted checks validate packaging and Python behavior; they do not run Cascadeur.

The external server must not import `csc`, NumPy or Qt. Native modules under
`host/` may import those packages inside handlers. The installed script runner
owns main-thread scheduling. Keep tool schemas and preflight validation consistent
on both sides, check the result of native modifications, and report uncertain
write outcomes without automatic retries.

Use exact object IDs or unambiguous names. Preserve untouched key metadata.
Prepare an entire batch before modifying a scene. Read back the written channels;
a successful native call is insufficient to establish animation quality.

For local checks:

```powershell
uv run ruff check src tests examples
uv run pytest tests/test_animation.py tests/test_motion.py tests/test_stdio.py
uv build
```

Native smoke checks use copies of installed vendor samples in the configured
workspace. These assets are never included in the package or repository.

```powershell
uv run python examples/native_smoke.py --sample Cube
uv run python examples/native_smoke.py --sample Cascy
```
