"""Qt render state machine checks with native and timer doubles."""

import builtins
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "integrations/kimodo"


def load(name):
    spec = importlib.util.spec_from_file_location("kimodo_render_test_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = load("native_render")
NativeType = renderer._Native
workflow = load("workflow")
PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 100 + b"\x00\x00\x00\x00IEND\xaeB`\x82"


class FakeNative:
    def __init__(self, scene):
        self.scene = str(scene)
        self.frame = 0
        self.restored = 0
        self.prepared = []
        self.timer_stopped = False
        self.timer_deleted = False
        self.disconnected = False

    def inspect(self):
        return {"scene": self.scene, "frame": self.frame, "frame_count": 4, "fingerprint": "rig",
                "file_stamp": [100, 1], "camera": {"position": [0, 0, 500], "target": [0, 100, 0], "mode": "Mesh"}}

    def validate_output(self, output):
        return Path(output)

    def prepare(self, frame, position, target):
        self.frame = frame
        self.prepared.append(frame)

    def capture(self, path):
        Path(path).write_bytes(PNG)

    def restore(self, task):
        self.restored += 1
        self.frame = task["original"]["frame"]
        return {"restored": True, "frame": self.frame}

    def timer(self, callback):
        return SimpleNamespace(start=lambda: None, stop=lambda: setattr(self, "timer_stopped", True),
                               deleteLater=lambda: setattr(self, "timer_deleted", True),
                               timeout=SimpleNamespace(disconnect=lambda callback: setattr(self, "disconnected", True)))

    def connect_shutdown(self, callback):
        self.shutdown = callback

    def disconnect_shutdown(self, callback):
        self.shutdown = None


class NativeRenderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.scene = self.root / "source.casc"
        self.scene.write_bytes(b"scene")
        self.output = self.root / "render"
        self.output.mkdir()
        self.environment = FakeNative(self.scene)
        self.factory = patch.object(renderer, "_Native", return_value=self.environment)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.previous = getattr(builtins, renderer.SLOT, None)
        setattr(builtins, renderer.SLOT, None)
        self.addCleanup(setattr, builtins, renderer.SLOT, self.previous)

    def start(self):
        return renderer.start({"scene": str(self.scene), "output": str(self.output),
                               "roots": [[0, 100, frame * 2] for frame in range(4)], "frame_step": 2})

    def test_render_waits_for_pngs_then_restores_and_disposes_timer(self):
        self.start()
        task = getattr(builtins, renderer.SLOT)
        now = task["_started"]
        for _ in range(100):
            now += 0.11
            renderer._tick(task, now=now)
            if task["state"] != "running":
                break
        self.assertEqual(task["state"], "completed")
        self.assertEqual(task["completed"], 9)
        self.assertEqual(task["frame_indices"], [0, 2, 3])
        self.assertEqual(self.environment.prepared, [0, 2, 3] * 3)
        self.assertEqual(self.environment.restored, 1)
        self.assertEqual(self.environment.frame, 0)
        self.assertTrue(self.environment.timer_stopped)
        self.assertTrue(self.environment.timer_deleted)
        self.assertTrue(self.environment.disconnected)
        self.assertIsNone(task["_environment"])
        self.assertIsNone(task["_callback"])
        self.assertTrue(all(Path(item["path"]).is_file() for item in task["images"]))

    def test_user_playhead_change_cancels_further_captures_and_restores(self):
        self.start()
        task = getattr(builtins, renderer.SLOT)
        self.environment.frame = 3
        renderer._tick(task)
        self.assertEqual(task["state"], "failed")
        self.assertIn("playhead", task["error"])
        self.assertEqual(task["completed"], 0)
        self.assertEqual(self.environment.restored, 1)

    def test_timeout_and_cancellation_are_terminal_without_restart(self):
        public = self.start()
        task = getattr(builtins, renderer.SLOT)
        renderer._tick(task, now=task["_started"] + 901)
        self.assertEqual(task["state"], "failed")
        self.assertIn("timeout", task["error"])
        renderer.cancel(public["id"], public["receipt"])
        self.assertEqual(self.environment.restored, 1)

    def test_second_active_task_and_existing_receipt_are_refused(self):
        public = self.start()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            self.start()
        renderer.cancel(public["id"], public["receipt"])
        with self.assertRaises(FileExistsError):
            self.start()

    def test_missing_owner_reports_interrupted_and_never_resumes(self):
        public = self.start()
        setattr(builtins, renderer.SLOT, None)
        observed = renderer.status(public["id"], public["receipt"])
        self.assertEqual(observed["state"], "interrupted")
        self.assertEqual(self.environment.prepared, [])

    def test_incomplete_png_cannot_advance_frame(self):
        image = self.root / "incomplete.png"
        image.write_bytes(PNG[:-12])
        ready, signature = renderer._complete_png(image, None)
        self.assertFalse(ready)
        ready, _ = renderer._complete_png(image, signature)
        self.assertFalse(ready)
        image.write_bytes(PNG)
        ready, signature = renderer._complete_png(image, None)
        self.assertFalse(ready)
        self.assertTrue(renderer._complete_png(image, signature)[0])

    def test_native_prepare_uses_view_mode_and_requested_camera(self):
        calls = []
        native = object.__new__(NativeType)
        native.common = SimpleNamespace(current_context=lambda: (
            None, None, None, SimpleNamespace(set_current_frame=lambda frame: calls.append(("frame", frame)))
        ))
        native.scene_api = SimpleNamespace(
            viewport_mode_set=lambda mode: calls.append(("mode", mode)),
            camera_set=lambda **values: calls.append(("camera", values)),
        )
        native.prepare(9, [400, 165, 400], [0, 105, 0])
        self.assertEqual(calls, [("frame", 9), ("mode", "View"),
                                 ("camera", {"position": [400, 165, 400], "target": [0, 105, 0]})])

    def test_native_restore_attempts_mode_and_frame_after_camera_failure(self):
        calls = []

        def fail_camera(camera):
            raise RuntimeError("camera restore failed")

        viewport = SimpleNamespace(camera_struct=lambda: SimpleNamespace(), set_camera_struct=fail_camera,
                                   set_mode_visualizers=lambda mode: calls.append(("mode", mode)))
        scene = SimpleNamespace(set_current_frame=lambda frame: calls.append(("frame", frame)))
        view = SimpleNamespace(get_path_name=lambda: str(self.scene), domain_scene=lambda: scene,
                               active_viewport=lambda: SimpleNamespace(domain_viewport=lambda: viewport))
        app = SimpleNamespace(get_scene_manager=lambda: SimpleNamespace(scenes=lambda: [view]))
        csc = SimpleNamespace(view=SimpleNamespace(ViewportMode=SimpleNamespace(__members__={"Points": "points"})))
        native = object.__new__(NativeType)
        native.common = SimpleNamespace(current_context=lambda **kwargs: (csc, app, view, scene))
        native.characters = SimpleNamespace(resolve_character=lambda *args, **kwargs: SimpleNamespace(fingerprint="rig"))
        task = {"scene": str(self.scene), "fingerprint": "rig", "original": {
            "frame": 3, "camera": {"position": [0, 0, 500], "target": [0, 100, 0], "mode": "Points"}}}
        with patch.dict(sys.modules, {"numpy": SimpleNamespace(asarray=lambda value, **kwargs: value, float32="f32")}):
            result = native.restore(task)
        self.assertFalse(result["restored"])
        self.assertIn("camera restore failed", result["errors"][0])
        self.assertEqual(calls, [("mode", "points"), ("frame", 3)])

    def test_existing_png_is_not_overwritten_after_prepare(self):
        self.start()
        task = getattr(builtins, renderer.SLOT)
        now = task["_started"] + 0.2
        renderer._tick(task, now=now)
        path = Path(task["images"][0]["path"])
        path.write_bytes(b"keep existing data")
        renderer._tick(task, now=now + 0.2)
        self.assertEqual(task["state"], "failed")
        self.assertEqual(path.read_bytes(), b"keep existing data")

    def test_encoding_holds_samples_for_exact_source_duration_including_final_frame(self):
        images = []
        for index, frame in enumerate((0, 2, 3)):
            path = self.root / f"{frame}.png"
            path.write_bytes(bytes([65 + index]))
            images.append({"frame": frame, "path": str(path)})
        video = self.root / "video.mp4"

        def encode(command, **kwargs):
            self.assertEqual(kwargs["input"], b"AABC")
            self.assertEqual(command[command.index("-frames:v") + 1], "4")
            video.write_bytes(b"encoded")

        with patch.object(workflow.subprocess, "run", side_effect=encode):
            workflow._encode_rendered_images(images, 4, video, "ffmpeg")


if __name__ == "__main__":
    unittest.main()
