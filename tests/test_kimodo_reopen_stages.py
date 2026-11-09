"""Protect the domain scene retained by the vendor scene-idle callback."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
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

    def save(self, path):
        if Path(path) != self.path:
            raise AssertionError("A close preparation must retain the owned source path")


class KimodoReopenStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kimodo-reopen-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for name in ("before", "working", "source", "result"):
            self.save_scene(name)
        self.before_bytes = self.root.joinpath("before.casc").read_bytes()
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
            remove_application_scene=lambda view: self.fail("Use the native data-source close path"),
        )
        self.datasource = SimpleNamespace(load_scene=self.open, close_scene=self.remove)
        self.app = SimpleNamespace(get_data_source_manager=lambda: self.datasource)
        self.snapshot = {
            "fingerprint": "native-semantic-rig",
            "positions": {"pelvis_MainPoint": [[0, 100, 0], [0, 100, 20]]},
        }
        self.addCleanup(patch.stopall)
        patch.object(bridge, "_application", return_value=(None, self.app, self.manager)).start()
        patch.object(
            bridge, "_snapshot", side_effect=lambda view, count: copy.deepcopy(self.snapshot)
        ).start()
        patch.object(
            points, "snapshot", side_effect=lambda view, count, record: copy.deepcopy(self.snapshot)
        ).start()
        patch.object(points, "_bridge", return_value=bridge).start()

    def activate(self, view):
        self.current = view

    def save_scene(self, name, linked=None):
        with zipfile.ZipFile(self.root / (name + ".casc"), "w") as archive:
            archive.writestr("tool/linked_scenes.json", json.dumps({"Data": linked or {}}))

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
            "point_bindings": {},
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

    def test_saved_inactive_scene_uses_datasource_close_and_retains_other_tabs(self):
        bridge._close_inactive_view(self.source, self.callback)
        self.assertEqual(self.removed, [self.source])
        self.assertEqual(self.views, [self.user, self.result])
        self.assertIs(self.current, self.result)

    def test_linked_scene_expansion_is_rejected_before_native_close(self):
        self.save_scene("source", {"scene": "result.casc"})
        with self.assertRaisesRegex(RuntimeError, "linked scenes"):
            bridge._close_inactive_view(self.source, self.callback)
        self.assertEqual(self.removed, [])

    def test_missing_or_malformed_link_information_never_dispatches_close(self):
        for content in (None, b"not JSON", b'{"Data": []}'):
            with self.subTest(content=content):
                with zipfile.ZipFile(self.source.path, "w") as archive:
                    if content is not None:
                        archive.writestr("tool/linked_scenes.json", content)
                with self.assertRaisesRegex(RuntimeError, "linked scenes"):
                    bridge._close_inactive_view(self.source, self.callback)
        self.assertEqual(self.removed, [])

    def test_pending_native_close_stops_reopen_instead_of_loading_a_duplicate(self):
        receipt = self.receipt(True)
        points.reopen_prepare(receipt)
        self.callback = self.current.domain_scene()
        opened_before = list(self.views)
        self.datasource.close_scene = lambda view: None
        with self.assertRaisesRegex(RuntimeError, "has not completed"):
            points.reopen(receipt, callback_scene=self.callback)
        self.assertEqual(self.views, opened_before)
        record = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(record["phase"], "needs_review:points_verified")

    def test_unexpected_native_removal_of_another_tab_is_detected(self):
        def unsafe_close(view):
            self.views.remove(view)
            self.views.remove(self.user)

        self.datasource.close_scene = unsafe_close
        with self.assertRaisesRegex(RuntimeError, "another tab"):
            bridge._close_inactive_view(self.source, self.callback)

    def test_native_close_cannot_silently_change_the_active_anchor(self):
        def activate_another_tab(view):
            self.views.remove(view)
            self.current = self.user

        self.datasource.close_scene = activate_another_tab
        with self.assertRaisesRegex(RuntimeError, "another tab"):
            bridge._close_inactive_view(self.source, self.callback)

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
        self.assertEqual(self.root.joinpath("before.casc").read_bytes(), self.before_bytes)

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
