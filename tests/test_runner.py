import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError

import pytest

from mcp_for_cascadeur.runner import RunnerClient, RunnerError, validate_endpoint


@pytest.fixture
def native_server():
    state = {
        "payload": {"ok": True, "result": {"native": True}, "execution_state": "completed"},
        "messages": [],
        "calls": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply({"ok": True, "version": "0.3.1"})

        def do_POST(self):
            state["calls"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.reply(
                {"ok": True, "value": repr(json.dumps(state["payload"])), "messages": state["messages"]}
            )

        def reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield RunnerClient(f"http://127.0.0.1:{server.server_port}"), state
    server.shutdown()
    server.server_close()
    thread.join()


def test_native_expression_round_trip(native_server):
    client, state = native_server
    assert client.health()["version"] == "0.3.1"
    assert client.call("app_status", {})["result"] == {"native": True}
    assert ".dispatch('app_status', {}" in state["calls"][0]["code"]


def test_native_error_messages_override_success(native_server):
    client, state = native_server
    state["messages"] = [{"level": "Error", "text": "Solver failed"}]
    with pytest.raises(RunnerError) as caught:
        client.call("app_status", {})
    assert caught.value.code == "NATIVE_EVENT_ERROR"
    assert caught.value.execution_state == "unknown"


def test_adapter_errors_keep_execution_state(native_server):
    client, state = native_server
    state["payload"] = {
        "ok": False,
        "error": {"code": "ValueError", "message": "Invalid pose", "execution_state": "not_started"},
    }
    with pytest.raises(RunnerError) as caught:
        client.call("app_status", {})
    assert caught.value.execution_state == "not_started"


def test_invalid_arguments_do_not_reach_application(native_server):
    client, state = native_server
    with pytest.raises(RunnerError, match="unknown fields"):
        client.call("app_status", {"unexpected": True})
    assert not state["calls"]


def test_motion_plans_work_without_a_native_request(native_server):
    client, state = native_server
    result = client.call(
        "motion_sample",
        {
            "keyframes": [
                {"frame": 0, "poses": [{"object": "hand", "position": [0, 0, 0]}]},
                {"frame": 2, "poses": [{"object": "hand", "position": [2, 0, 0]}]},
            ],
            "easing": "linear",
        },
    )
    assert result["executor"] == "local"
    assert result["result"]["keyframes"][1]["poses"][0]["position"] == [1, 0, 0]
    assert not state["calls"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8765",
        "http://example.com:8765",
        "http://user:secret@localhost:8765",
        "http://localhost:8765/run",
    ],
)
def test_endpoint_must_be_local(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


@pytest.mark.parametrize("wrapped", [False, True])
def test_lost_connection_after_post_does_not_claim_write_was_not_started(wrapped):
    failure = ConnectionResetError("The peer reset the connection after accepting the request")
    if wrapped:
        failure = URLError(failure)

    class BrokenConnection:
        def open(self, *args, **kwargs):
            raise failure

    client = RunnerClient()
    client._opener = BrokenConnection()
    with pytest.raises(RunnerError) as caught:
        client._request("/run", {"code": "submitted_operation()"})
    assert caught.value.code == "CONNECTION_LOST"
    assert caught.value.execution_state == "unknown"


def test_refused_connection_reports_no_submission():
    class RefusedConnection:
        def open(self, *args, **kwargs):
            raise URLError(ConnectionRefusedError("No listener"))

    client = RunnerClient()
    client._opener = RefusedConnection()
    with pytest.raises(RunnerError) as caught:
        client._request("/run", {"code": "not_submitted()"})
    assert caught.value.code == "NOT_CONNECTED"
    assert caught.value.execution_state == "not_started"


def test_local_runner_redirect_is_not_followed(native_server):
    target_client, _ = native_server

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", target_client.endpoint + "/health")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        redirected = RunnerClient(f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(RunnerError) as caught:
            redirected.health()
        assert caught.value.code == "HTTP_ERROR"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
