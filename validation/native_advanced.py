"""Run bounded acceptance cases in the user's installed Cascadeur.

Fixture copies and receipts live in the configured workspace, outside the repo.
Raw runner access is limited to installing this build and deliberate fault probes.
"""

from __future__ import annotations

import ast
import json
import time
from urllib.request import ProxyHandler, Request, build_opener

from mcp_for_cascadeur.installation import install_runtime
from mcp_for_cascadeur.runner import RunnerClient
from mcp_for_cascadeur.settings import load_config

CONFIG = load_config()
CLIENT = RunnerClient(CONFIG.endpoint, CONFIG.workspace, allow_python=True, timeout=120)
OPENER = build_opener(ProxyHandler({}))


def raw(code):
    request = Request(
        CONFIG.endpoint + "/run",
        data=json.dumps({"code": code}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with OPENER.open(request, timeout=120) as response:
        value = json.load(response)
    if not value.get("ok"):
        raise RuntimeError(value)
    return ast.literal_eval(value["value"]) if value.get("value") else value


def upgrade():
    bootstrap = install_runtime(CONFIG)
    # Drop this package only. The vendor HTTP server remains in place.
    code = (
        "import sys, importlib\n"
        "for name in list(sys.modules):\n"
        "    if name == 'mcp_for_cascadeur' or name.startswith('mcp_for_cascadeur.'):\n"
        "        del sys.modules[name]\n"
        "importlib.invalidate_caches()\n"
        f"exec(compile(open({str(bootstrap)!r}, encoding='utf-8').read(), {str(bootstrap)!r}, 'exec'))\n"
    )
    return raw(code)


def call(name, **arguments):
    started = time.monotonic()
    result = CLIENT.call(name, arguments)
    print(
        json.dumps(
            {"tool": name, "seconds": round(time.monotonic() - started, 2), **result}, ensure_ascii=False
        ),
        flush=True,
    )
    return result


def probe(code):
    """Trusted read-only diagnostics or explicitly documented fault injection."""
    expression = (
        "__import__('json').dumps(__import__('mcp_for_cascadeur.host', fromlist=['python_execute'])"
        f".python_execute({code!r}), allow_nan=False)"
    )
    return json.loads(raw(expression))["value"]


if __name__ == "__main__":
    upgrade()
    call("app_status")
    call("recovery_status")
    call("characters_list")
    call("rig_semantics")
    call("hand_pose_read")
