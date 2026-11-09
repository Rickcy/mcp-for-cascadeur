"""Protect the domain scene retained by the vendor scene-idle callback."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "integrations" / "kimodo"


def load(name):
    spec = importlib.util.spec_from_file_location("kimodo_reopen_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = load("cascadeur_bridge")
points = load("point_retarget")
cli = load("native_import")


class View:
    def __init__(self, path):
        self.path = Path(path)
        self.domain = object()

    def get_path_name(self):
        return str(self.path)

    def name(self):
        return self.path.name

    def domain_scene(self):
        return self.domain


class KimodoReopenStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kimodo-reopen-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.root.joinpath("before.casc").write_bytes(b"untouched recovery copy")
        self.root.joinpath("working.casc").write_bytes(b"working copy")
        self.root.joinpath("source.casc").write_bytes(b"source copy")
        self.root.joinpath("result.casc").write_bytes(b"saved generated scene")
        self.result = View(self.root / "result.casc")
        self.source = View(self.root / "source.casc")
        self.user = View(self.root.parent / "user-project.casc")
        self.views = [self.user, self.source, self.result]
        self.current = self.result
        self.removed = []
        self.callback = self.result.domain_scene()
        self.manager = SimpleNamespace(
            scenes=lambda: self.views,
            current_scene=lambda: self.current,
            set_current_scene=self.activate,
            remove_application_scene=self.remove,
        )
        datasource = SimpleNamespace(load_scene=self.open)
        self.app = SimpleNamespace(get_data_source_manager=lambda: datasource)
        self.snapshot = {
            "fingerprint": "native-semantic-rig",
            "positions": {"pelvis_MainPoint": [[0, 100, 0], [0, 100, 20]]},
        }
        self.addCleanup(patch.stopall)
        patch.object(bridge, "_application", return_value=(None, self.app, self.manager)).start()
        patch.object(
            bridge, "_snapshot", side_effect=lambda view, count: copy.deepcopy(self.snapshot)
        ).start()
        patch.object(points, "_bridge", return_value=bridge).start()

    def activate(self, view):
        self.current = view

    def remove(self, view):
        self.assertIsNot(view, self.current)
        self.assertIsNot(view.domain_scene(), self.callback)
        self.views.remove(view)
        self.removed.append(view)

    def open(self, path):
        view = View(path)
        self.views.append(view)
        self.current = view
        return True

    def receipt(self, point_route):
        record = {
            "schema": "kimodo-point-session-v1" if point_route else "kimodo-cascadeur-import-v1",
            "directory": str(self.root),
            "phase": "points_saved" if point_route else "saved",
            "events": [],
            "active_target": "result",
            "num_frames": 2,
            "generation": {"num_frames": 2},
            "before_sha256": hashlib.sha256(self.root.joinpath("before.casc").read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256(self.root.joinpath("result.casc").read_bytes()).hexdigest(),
            "expected": self.snapshot,
        }
        receipt = self.root / "session.json"
        receipt.write_text(json.dumps(record), encoding="utf-8")
        return receipt

    def test_active_scene_cannot_be_closed(self):
        with self.assertRaisesRegex(RuntimeError, "active tab"):
            bridge._close_inactive_view(self.result, self.callback)
        self.assertEqual(self.removed, [])

    def test_switching_active_tab_does_not_release_original_callback_scene(self):
        self.current = self.source
        with self.assertRaisesRegex(RuntimeError, "scene held"):
            bridge._close_inactive_view(self.result, self.callback)
        self.assertEqual(self.removed, [])

    def test_missing_callback_context_never_allows_scene_closure(self):
        with self.assertRaisesRegex(RuntimeError, "callback"):
            bridge._close_inactive_view(self.source, None)
        self.assertEqual(self.removed, [])

    def test_point_reopen_and_cleanup_keep_each_actual_callback_anchor_alive(self):
        receipt = self.receipt(True)
        prepared = points.reopen_prepare(receipt)
        self.assertEqual(prepared["phase"], "points_reopen_ready")
        self.assertEqual(self.removed, [])
        anchor = self.current
        self.assertEqual(anchor.path.name, "before.casc")
        self.assertIn(self.result, self.views)
        # A separate native callback now owns the retained before.casc domain.
        self.callback = anchor.domain_scene()
        verified = points.reopen(receipt, callback_scene=self.callback)
        self.assertEqual(verified["phase"], "points_verified")
        self.assertTrue(verified["point_coordinates_verified"])
        self.assertIn(anchor, self.views)
        self.assertEqual(self.current.path.name, "result.casc")
        self.assertEqual(self.removed, [self.result])
        self.callback = self.current.domain_scene()
        completed = points.cleanup(receipt, callback_scene=self.callback)
        self.assertEqual(completed["phase"], "completed")
        self.assertNotIn(anchor, self.views)
        self.assertIn(self.user, self.views)
        self.assertEqual(self.root.joinpath("before.casc").read_bytes(), b"untouched recovery copy")

    def test_point_reopen_rejects_old_idle_context_even_after_prepare_switch(self):
        receipt = self.receipt(True)
        points.reopen_prepare(receipt)
        with self.assertRaisesRegex(RuntimeError, "scene_idle callback"):
            points.reopen(receipt, callback_scene=self.callback)
        self.assertEqual(self.removed, [])

    def test_pro_route_also_requires_separate_source_anchor_request(self):
        receipt = self.receipt(False)
        prepared = bridge.reopen_prepare(receipt)
        self.assertEqual(prepared["phase"], "reopen_ready")
        self.assertEqual(self.removed, [])
        self.assertIs(self.current, self.source)
        self.callback = self.source.domain_scene()
        verified = bridge.reopen_result(receipt, callback_scene=self.callback)
        self.assertEqual(verified["phase"], "verified")
        self.assertIn(self.source, self.views)
        self.callback = self.current.domain_scene()
        completed = bridge.cleanup(receipt, callback_scene=self.callback)
        self.assertEqual(completed["phase"], "completed")
        self.assertIn(self.user, self.views)
        self.assertNotIn(self.source, self.views)

    def test_existing_completed_receipts_have_no_new_stage(self):
        self.assertEqual(cli._next_stage({"phase": "completed"}), (None, None))


if __name__ == "__main__":
    unittest.main()
