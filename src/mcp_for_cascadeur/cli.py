"""Command-line setup, diagnostics and stdio entry point."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from . import installation
from .settings import load_config, save_config


def _output(value: dict | list, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    if isinstance(value, list):
        for item in value:
            print(item.get("name", str(item)) if isinstance(item, dict) else item)
        return
    for key, item in value.items():
        print(f"{key}: {json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-for-cascadeur", description="Connect MCP clients to Cascadeur's installed Python API."
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="Run the MCP stdio server (the default).")
    setup = commands.add_parser("setup", help="Install a stable host runtime and configure an MCP client.")
    setup.add_argument("--cascadeur-home", type=Path)
    setup.add_argument("--workspace", type=Path)
    setup.add_argument("--endpoint")
    setup.add_argument("--client", choices=("codex", "none"), default="codex")
    setup.add_argument("--install-startup", action="store_true")
    setup.add_argument("--allow-python", action="store_true", default=None)
    setup.add_argument(
        "--local-tool",
        action="store_true",
        help="Use an already installed uv tool in the client configuration.",
    )
    setup.add_argument(
        "--source",
        default=None,
        help="Package source for the generated client's uvx --from argument.",
    )
    setup.add_argument("--json", action="store_true")
    doctor = commands.add_parser("doctor", help="Read the runner health and live application status.")
    doctor.add_argument("--json", action="store_true")
    uninstall = commands.add_parser(
        "uninstall", help="Remove this project's startup hook and client entry; preserve scenes and settings."
    )
    uninstall.add_argument("--cascadeur-home", type=Path)
    uninstall.add_argument("--client", choices=("codex", "none"), default="codex")
    uninstall.add_argument("--json", action="store_true")
    tools = commands.add_parser("tools", help="List tools without connecting to Cascadeur.")
    tools.add_argument("--json", action="store_true")
    review = commands.add_parser("review", help="Configure the visual reviewer without putting keys in source.")
    review_commands = review.add_subparsers(dest="review_command", required=True)
    configure = review_commands.add_parser("configure", help="Save nonsecret provider settings outside the checkout.")
    configure.add_argument("--provider", choices=("gemini", "openai-compatible"))
    configure.add_argument("--model")
    configure.add_argument("--base-url")
    configure.add_argument("--media-transport", choices=("video", "frames"))
    configure.add_argument("--sampling-fps", type=float)
    configure.add_argument("--max-total-frames", type=int)
    configure.add_argument("--max-frames-per-video", type=int)
    configure.add_argument("--frame-width", type=int)
    configure.add_argument("--json", action="store_true")
    review_status = review_commands.add_parser("status", help="Show provider settings and key availability, never its value.")
    review_status.add_argument("--json", action="store_true")
    key = review_commands.add_parser("key", help="Manage a current-user encrypted API key on Windows.")
    key_commands = key.add_subparsers(dest="key_command", required=True)
    for operation in ("set", "delete"):
        entry = key_commands.add_parser(operation)
        entry.add_argument("provider", choices=("gemini", "openai-compatible"))
        entry.add_argument("--json", action="store_true")
        if operation == "set":
            entry.add_argument("--from-env", metavar="VARIABLE", help="Read an existing environment variable instead of prompting.")
    motion = commands.add_parser("motion", help="Manage the jobs shared by Smart Motion and MCP.")
    motion_commands = motion.add_subparsers(dest="motion_command", required=True)
    for operation in ("status", "jobs", "panel"):
        entry = motion_commands.add_parser(operation)
        entry.add_argument("--json", action="store_true")
    entry = motion_commands.add_parser("configure")
    for name in ("checkout", "calibration", "template-motion", "template-scene"):
        entry.add_argument("--" + name)
    entry.add_argument("--json", action="store_true")
    entry = motion_commands.add_parser("generate")
    entry.add_argument("--request", required=True, type=Path, help="JSON with the same options as kimodo_generate.")
    entry.add_argument("--json", action="store_true")
    for operation in ("job", "cancel", "resume", "review", "select"):
        entry = motion_commands.add_parser(operation)
        entry.add_argument("job_id")
        if operation == "select":
            entry.add_argument("candidate_id")
        entry.add_argument("--json", action="store_true")
    entry = motion_commands.add_parser("runtime")
    entry.add_argument("action", choices=("status", "start", "stop", "build", "logs"))
    entry.add_argument("--json", action="store_true")
    entry = motion_commands.add_parser("import")
    entry.add_argument("output", type=Path)
    entry.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "serve"
    try:
        if command == "review":
            from . import review_settings

            if args.review_command == "status":
                result = review_settings.public_review_status()
            elif args.review_command == "configure":
                result = review_settings.configure_review(**{
                    key: getattr(args, key) for key in review_settings.DEFAULTS
                })
            elif args.key_command == "delete":
                result = review_settings.delete_review_key(args.provider)
            else:
                if args.from_env:
                    value = os.environ.get(args.from_env)
                    if not value:
                        raise ValueError("The named API key environment variable is empty")
                else:
                    if not sys.stdin.isatty():
                        raise ValueError("Use an interactive hidden prompt or --from-env VARIABLE")
                    value = getpass.getpass("API key (hidden): ")
                result = review_settings.set_review_key(args.provider, value)
                value = None
            _output(result, args.json)
            return 0
        config = load_config()
        if command == "motion":
            from .motion_tools import tools as motion_tools

            contracts = {tool.name: tool for tool in motion_tools(config)}
            operation = args.motion_command
            if operation == "panel":
                from .runner import RunnerClient

                result = RunnerClient(config.endpoint, config.workspace, allow_python=config.allow_python).call(
                    "smart_motion_panel", {}
                )
            else:
                name = {"job": "job_status", "import": "import_run"}.get(operation, operation)
                values = {}
                if operation == "configure":
                    values = {name: getattr(args, name) for name in
                              ("checkout", "calibration", "template_motion", "template_scene") if getattr(args, name)}
                elif operation == "generate":
                    if args.request.stat().st_size > 1024 * 1024:
                        raise ValueError("Motion request exceeds 1 MiB")
                    values = json.loads(args.request.read_text(encoding="utf-8"))
                elif operation == "runtime":
                    values = {"action": args.action}
                elif operation == "import":
                    values = {"output": str(args.output.resolve())}
                elif operation not in {"status", "jobs"}:
                    values = {"job_id": args.job_id}
                    if operation == "select":
                        values["candidate_id"] = args.candidate_id
                definition = contracts["kimodo_" + name]
                definition.validate(values)
                result = definition.handler(**values)
            _output(result, args.json)
            return 0
        if command == "serve":
            from .server import main as serve

            serve(config)
            return 0
        if command == "tools":
            from .server import tool_definitions

            _output(
                [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                        "read_only": tool.read_only,
                    }
                    for tool in tool_definitions(replace(config, allow_python=True)).values()
                ],
                args.json,
            )
            return 0
        if command == "setup":
            home = installation.detect_cascadeur_home(args.cascadeur_home or config.cascadeur_home)
            changes = {"cascadeur_home": home}
            for name in ("workspace", "endpoint", "allow_python"):
                if getattr(args, name) is not None:
                    changes[name] = getattr(args, name)
            config = replace(config, **changes)
            config.workspace.mkdir(parents=True, exist_ok=True)
            bridge = installation.install_runtime(config)
            result = {
                "ok": True,
                "cascadeur_home": str(home),
                "workspace": str(config.workspace),
                "start_bridge": str(bridge),
                "panel": installation.install_panel_menu(config, bridge),
            }
            if args.install_startup:
                result["startup"] = installation.install_startup(config, bridge)
            if args.client == "codex":
                result["codex"] = installation.configure_codex(
                    source=None if args.local_tool else args.source or installation.default_package_source()
                )
            result["settings"] = str(save_config(config))
            result["next"] = (
                "Restart Cascadeur and reconnect the MCP client."
                if args.install_startup
                else f"Run {bridge} in Cascadeur's Python console, then run mcp-for-cascadeur doctor."
            )
            _output(result, args.json)
            return 0
        if command == "doctor":
            from .runner import RunnerClient

            client = RunnerClient(config.endpoint, config.workspace, allow_python=config.allow_python)
            health = client.health()
            status = client.call("app_status", {})
            _output(
                {"ok": True, "endpoint": config.endpoint, "health": health, "app_status": status}, args.json
            )
            return 0
        if command == "uninstall":
            result = {
                "ok": True,
                "startup": installation.uninstall_startup(args.cascadeur_home or config.cascadeur_home),
                "panel": installation.uninstall_panel_menu(args.cascadeur_home or config.cascadeur_home),
            }
            if args.client == "codex":
                result["codex"] = installation.configure_codex(remove=True)
            _output(result, args.json)
            return 0
    except Exception as exc:
        result = {"ok": False, "code": getattr(exc, "code", type(exc).__name__), "error": str(exc)}
        if hasattr(exc, "execution_state"):
            result["execution_state"] = exc.execution_state
        if getattr(args, "json", False):
            _output(result, True)
        else:
            print(f"{result['code']}: {exc}", file=sys.stderr)
        return 1
    return 0
