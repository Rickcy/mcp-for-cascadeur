"""Modeless Cascadeur frontend for the external Smart Motion job service.

Only scene inspection, an explicitly requested save-copy and owned playback run
here. Generation, HTTP orchestration, credentials and model dependencies stay in
the external interpreter. Network requests use Qt asynchronously.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

_window = None
_launcher = None
_failed_panels = []


def _data_home():
    override = os.environ.get("MCP_FOR_CASCADEUR_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return base / "mcp-for-cascadeur"


def _private_json(name):
    path = _data_home() / name
    if path.stat().st_size > 65536:
        raise ValueError("Smart Motion private configuration is too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Smart Motion private configuration must be an object")
    return value


def _connection():
    record = _private_json("smart-motion-service.json")
    endpoint, token = record.get("endpoint", ""), record.get("token", "")
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment or parsed.port is None):
        raise ValueError("Smart Motion requires its local service endpoint")
    if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_.~-]{20,256}", token):
        raise ValueError("Smart Motion service authentication is not configured")
    return endpoint.rstrip("/"), token


def _start_service():
    global _launcher
    if _launcher is not None and _launcher.poll() is None:
        return
    settings = _private_json("smart-motion-panel.json")
    python = Path(settings.get("external_python", ""))
    if not python.is_absolute() or not python.is_file():
        raise ValueError("Repeat MCP setup to register the external Smart Motion interpreter")
    environment = dict(os.environ, MCP_FOR_CASCADEUR_HOME=str(_data_home()))
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    _launcher = subprocess.Popen(
        [str(python), "-m", "mcp_for_cascadeur.motion_service"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def parse_keyframes(text):
    """Parse zero-based native anchor frames without silently sorting user input."""
    values = [item for item in re.split(r"[,;\s]+", text.strip()) if item]
    if len(values) < 2 or any(not re.fullmatch(r"\d{1,7}", item) for item in values):
        raise ValueError("Укажите минимум два номера кадра, например: 0, 30, 60, 90, 119")
    frames = [int(item) for item in values]
    if any(first >= last for first, last in zip(frames, frames[1:])):
        raise ValueError("Опорные кадры должны возрастать без повторений")
    return frames


def candidate_rows(job):
    """Retain saved outputs; visual-model results do not belong to this UI."""
    selection = job.get("selection") or {}
    candidates = job.get("candidates") or job.get("run", {}).get("candidates", [])
    return [dict(item, selected=selection.get("selected_id") == item["id"]) for item in candidates]


def progress_display(job):
    progress = job.get("progress") or {}
    fraction = progress.get("fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction):
        fraction = None
    elif not 0 <= fraction <= 1:
        fraction = None
    return {"state": job.get("state", "unknown"),
            "stage": progress.get("stage") or job.get("stage") or job.get("action", ""),
            "fraction": fraction, "message": progress.get("message") or job.get("error") or ""}


def _preview_request(run, stored_frames):
    if type(stored_frames) is not int or stored_frames < 2:
        raise ValueError("В сцене недостаточно кадров для воспроизведения")
    frames = run.get("frames", stored_frames)
    if type(frames) is not int or frames < 2:
        raise ValueError("В сохранённой задаче указана неверная длительность")
    frames = min(frames, stored_frames)
    # Slow viewports still need time to reach the final frame. The native
    # observer stops at that frame; this is only its bounded watchdog budget.
    return {"first": 0, "last": frames - 1, "timeout": max(30, frames / 3 + 10)}


def _job_label(job):
    title = job.get("prompt") or _stage_caption(job.get("action", "run"))
    state = {"queued": "В очереди", "running": "Выполняется", "completed": "Готово", "failed": "Ошибка",
             "cancel_requested": "Ожидание отмены", "cancelling": "Ожидание отмены", "cancelled": "Отменено",
             "interrupted": "Прервано"}.get(job.get("state"), "Сохранённая задача")
    created = job.get("created_at")
    stamp = ""
    if type(created) in (int, float) and math.isfinite(created):
        try:
            stamp = time.strftime("%d.%m %H:%M", time.localtime(created))
        except (OSError, OverflowError, ValueError):
            pass
    return " · ".join(part for part in (title[:65], state, stamp) if part)


def _completion_notice(job):
    state = job.get("state")
    if state == "completed":
        actions = {"select": "Анимация открыта.", "runtime_start": "Kimodo запущен.",
                   "runtime_stop": "GPU освобождён.", "runtime_build": "Docker-образ собран."}
        text = actions.get(job.get("action"))
        if text is None:
            text = ("Файлы анимации готовы и сохранены." if (job.get("run") or {}).get("mode") == "generated_only"
                    else "Анимация готова и сохранена.")
        return text, False
    if state == "failed":
        detail = str(job.get("error") or "Подробности доступны в журнале.")
        return "Не удалось завершить задачу: " + detail, True
    if state == "cancelled":
        return "Задача отменена. Сохранённые результаты доступны в истории.", False
    if state == "interrupted":
        return "Обработка прервана. Проверьте сохранённые результаты в истории.", True
    return None


def _notice_after_update(previous_state, job, *, task_notice):
    if task_notice and previous_state != job.get("state"):
        return _completion_notice(job)
    return None


def _settings_payload(value):
    if not isinstance(value, dict):
        raise ValueError("Настройки должны быть объектом")
    paths = {"checkout", "calibration", "template_motion", "template_scene"}
    return {key: item for key, item in value.items() if key != "review"
            and (key not in paths or not isinstance(item, str) or item.strip())}


def _panel_request(value, snapshot=None):
    """Build a single native result without rendering or visual-model calls."""
    if not isinstance(value, dict):
        raise ValueError("Параметры генерации должны быть объектом")
    payload = dict(value)
    for name in ("provider", "model", "base_url", "media_transport", "sampling_fps", "max_total_frames",
                 "max_frames_per_video", "frame_width", "review_enabled", "render_enabled", "auto_activate"):
        payload.pop(name, None)
    if payload.get("seed") in (None, ""):
        payload["seed"] = uuid.uuid4().int & 0xFFFFFFFF
    if type(payload["seed"]) is not int or not 0 <= payload["seed"] <= 0xFFFFFFFF:
        raise ValueError("Seed должен быть целым числом от 0 до 4294967295; оставьте поле пустым для нового движения")
    first, last = payload.pop("first_frame", None), payload.pop("last_frame", None)
    if type(first) is not int or type(last) is not int or first < 0 or last <= first:
        raise ValueError("Выберите диапазон минимум из двух кадров")
    payload.update(frames=last - first + 1, first_frame=first, samples=1, sample_index=0,
                   no_review=True, no_render=True, no_activate=False)
    anchors = payload.pop("use_keyposes", False)
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("Описание движения должно быть текстом")
    if not prompt.strip():
        if anchors and not payload.get("generate_only"):
            prompt = "A person moves naturally between the specified key poses."
        else:
            raise ValueError("Опишите движение или включите опорные позы из сцены")
    payload["prompt"] = prompt.strip()
    text = payload.pop("keyframes", "")
    if not payload.get("generate_only"):
        if not payload.get("character_id"):
            raise ValueError("Выберите персонажа из текущей сцены")
        if not snapshot:
            raise ValueError("Не удалось сохранить исходную сцену")
        payload["target_scene"] = snapshot
        if anchors:
            values = [item for item in re.split(r"[,;\s]+", str(text).strip()) if item]
            if any(not re.fullmatch(r"\d{1,10}", item) for item in values):
                raise ValueError("Номера опорных кадров должны быть целыми и неотрицательными")
            frames = sorted({first, last, *map(int, values)})
            if frames[0] < first or frames[-1] > last:
                raise ValueError("Опорные позы должны находиться внутри выбранного диапазона")
            payload.update(source_scene=snapshot, keyframes=frames)
    else:
        payload.pop("character_id", None)
        payload.pop("rig_mapping", None)
    return payload


def _load_rig_profile(rig):
    try:
        data = _private_json("smart-motion-rigs.json")
    except (OSError, ValueError):
        return {}
    profiles = data.get("profiles")
    if data.get("format") != "smart-motion-rigs-v1" or not isinstance(profiles, dict):
        return {}
    record = profiles.get(rig.get("fingerprint"))
    if not isinstance(record, dict):
        return {}
    mapping, roles = record.get("mapping"), {row["id"] for row in rig.get("roles", [])}
    if (not isinstance(mapping, dict) or set(mapping) != roles
            or any(not isinstance(value, str) or not value or len(value) > 256 for value in mapping.values())):
        return {}
    return dict(mapping)


def _save_rig_profile(rig):
    """Keep one bounded, nonsecret preference file for verified manual mappings."""
    if not rig.get("ready") or not re.fullmatch(r"[a-f0-9]{64}", rig.get("fingerprint", "")):
        return False
    try:
        data = _private_json("smart-motion-rigs.json")
    except (OSError, ValueError):
        data = {}
    profiles = data.get("profiles")
    profiles = {key: row for key, row in profiles.items()
                if isinstance(row, dict) and type(row.get("updated_at")) in (int, float)
                and math.isfinite(row["updated_at"])} if isinstance(profiles, dict) else {}
    profiles[rig["fingerprint"]] = {"character_id": rig["character_id"], "mapping": dict(rig["mapping"]),
                                   "updated_at": time.time()}
    while len(profiles) > 32:
        oldest = min(profiles, key=lambda key: profiles[key].get("updated_at", 0)
                     if isinstance(profiles[key], dict) else 0)
        profiles.pop(oldest)
    encoded = json.dumps({"format": "smart-motion-rigs-v1", "profiles": profiles}, ensure_ascii=False)
    while len(encoded.encode("utf-8")) > 65536 and len(profiles) > 1:
        oldest = min(profiles, key=lambda key: profiles[key].get("updated_at", 0)
                     if isinstance(profiles[key], dict) else 0)
        profiles.pop(oldest)
        encoded = json.dumps({"format": "smart-motion-rigs-v1", "profiles": profiles}, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError("Профиль скелета слишком велик")
    from ..atomic import replace_file

    target = _data_home() / "smart-motion-rigs.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        replace_file(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _character_label(scene, rig, scene_name, name):
    viewer, model = scene.behaviour_viewer(), scene.model_viewer()
    owned = {str(identifier) for identifier in rig.owned}
    meshes = sorted({model.get_object_name(owner) for information in viewer.get_behaviours("MeshObject")
                     if str(owner := viewer.get_behaviour_owner(information)) in owned}, key=str.casefold)
    if meshes:
        return ", ".join(meshes[:2]) + (f" +{len(meshes) - 2}" if len(meshes) > 2 else "")
    scene_title = Path(scene_name).stem
    return scene_title + " · " + name if scene_title else name


def _motion_context(character_id=""):
    """Resolve native character membership and the real timeline selection."""
    from . import characters, common

    _, _, view, scene = common.current_context()
    native_records = characters.characters_list()["characters"]
    selected = {str(item) for item in scene.selector().selected().ids}
    records, rigs = [], []
    for native_record in native_records:
        rig = characters.resolve_character(scene, native_record["id"])
        record = dict(native_record, display_name=_character_label(scene, rig, str(view.name()), native_record["name"]))
        records.append(record)
        rigs.append((record, rig))
    if character_id:
        matches = [item for item in rigs if item[0]["id"] == character_id]
    else:
        matches = [item for item in rigs if selected.intersection(map(str, item[1].owned))]
        if not matches and len(rigs) == 1:
            matches = rigs
    chosen = matches[0] if len(matches) == 1 else None
    boundary = view.animation_boundary()
    first, last = int(boundary.first_frame), int(boundary.last_frame)
    selection = scene.get_layers_selector().selection()
    interval = selection.frames_interval()
    selected_range = bool(list(selection.item_ids())) and interval.first >= 0 and interval.last > interval.first
    if selected_range:
        first, last = int(interval.first), int(interval.last)
    keys = []
    if chosen:
        layers = scene.layers_viewer()
        keys = sorted({int(frame) for identifier in chosen[1].tracks
                       for frame in layers.layer(identifier).key_frame_indices() if first <= frame <= last})
    result = {"characters": records, "character_id": chosen[0]["id"] if chosen else "",
              "first": first, "last": last, "frame": int(scene.get_current_frame()), "keys": keys,
              "range_source": "selection" if selected_range else "playback",
              "scene_path": str(view.get_path_name() or ""), "scene_name": str(view.name())}
    if chosen:
        from .humanoid import discover_mapping

        rig = discover_mapping(scene, chosen[0]["id"])
        profile = _load_rig_profile(rig)
        result["rig"] = discover_mapping(scene, chosen[0]["id"], profile) if profile else rig
    return result


def _can_resume(job):
    run = job.get("run") or {}
    return (run.get("format") == "kimodo-cascadeur-run-v1" and run.get("mode") != "generated_only"
            and run.get("state") in {"running", "failed", "interrupted", "cancelled"})


def _stage_caption(stage):
    captions = {
        "prepared": "Подготовка", "checking_cascadeur": "Проверка Cascadeur",
        "cascadeur_ready": "Cascadeur готов", "capturing_keyposes": "Захват опорных поз",
        "keyposes_captured": "Опорные позы сохранены", "starting_runtime": "Загрузка Kimodo",
        "submitting_generation": "Отправка задания Kimodo", "generation_accepted": "Задание Kimodo принято",
        "generation": "Генерация движения", "generation_complete": "Движение сгенерировано",
        "generated": "Генерация завершена", "converting_candidates": "Подготовка вариантов к переносу",
        "releasing_gpu": "Освобождение GPU", "native_render": "Съёмка кадров",
        "encoding": "Подготовка видео", "candidate_complete": "Вариант сохранён",
        "awaiting_review": "Анимация сохранена", "activating_selection": "Открытие анимации",
        "completed": "Готово", "resume_inputs_verified": "Сохранённые данные проверены",
        "run": "Создание движения", "resume": "Продолжение сохранённых этапов",
        "select": "Открытие анимации",
        "runtime_start": "Загрузка Kimodo", "runtime_stop": "Освобождение GPU",
        "runtime_build": "Сборка Docker-образа", "runtime_logs": "Чтение журнала Kimodo",
        "runtime_status": "Проверка состояния Kimodo",
    }
    match = re.fullmatch(r"(rendering|transferring|reusing)_candidate-(\d+)", stage or "")
    if match:
        prefix = {"rendering": "Съёмка анимации", "transferring": "Перенос на персонажа",
                  "reusing": "Проверка сохранённого варианта"}[match[1]]
        return prefix + " " + match[2]
    return captions.get(stage, stage or "Готово")


def _stage_label(job):
    progress = progress_display(job)
    title = _stage_caption(progress["stage"])
    if progress["state"] == "completed":
        return title
    state = {"queued": "В очереди", "running": "Выполняется", "cancel_requested": "Ожидание безопасной отмены",
             "cancelling": "Ожидание безопасной отмены", "failed": "Ошибка", "interrupted": "Прервано",
             "cancelled": "Отменено", "unknown": ""}.get(progress["state"], progress["state"])
    return f"{state} · {title}" if state else title


def _native_call(name, arguments):
    from . import dispatch

    response = dispatch(name, arguments)
    if response.get("ok") is not True:
        raise RuntimeError(response.get("error", {}).get("message", "Native operation did not complete"))
    return response["result"]


class _DeferredQmlDisposal:
    """Delete each Qt object naturally before posting the next deletion."""

    def __init__(self, objects, *, valid, finished):
        self.objects, self.valid, self.finished = tuple(objects), valid, finished
        self.index, self.waiting, self.done = 0, False, False

    def start(self):
        if self.waiting or self.done:
            return
        while self.index < len(self.objects):
            item = self.objects[self.index]
            if item is None or not self.valid(item):
                self.index += 1
                continue
            self.waiting = True
            expected = self.index

            def destroyed(*_ignored, expected=expected):
                if self.waiting and self.index == expected:
                    self.waiting = False
                    self.index += 1
                    self.start()

            item.destroyed.connect(destroyed)
            item.deleteLater()
            return
        self.done = True
        self.finished()


def _dispose_failed_panel(panel, *, valid, queue_delete):
    """Keep a failed QObject constructor alive through normal deferred cleanup."""
    _failed_panels.append(panel)

    def released(*_ignored):
        _failed_panels[:] = [item for item in _failed_panels if item is not panel]

    panel.destroyed.connect(released)

    def finished():
        panel.window, panel.component, panel.engine = None, None, None
        # Keep the Python wrapper retained until QObject.destroyed. Queueing
        # parent deletion also lets the QQmlEngine destructor return first.
        queue_delete(panel)

    panel._qml_cleanup = _DeferredQmlDisposal(
        (panel.window, panel.component, panel.engine), valid=valid, finished=finished)
    panel._qml_cleanup.start()


def _scene_info(*, save_copy=False):
    from . import common

    _, _, view, scene = common.current_context()
    if save_copy:
        folder = common.get_settings()["workspace"] / "kimodo/sources"
        folder.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^\w.-]+", "_", str(view.name()))[:60].removesuffix(".casc") or "scene"
        result = _native_call("scene_save", {"path": str(folder / f"{name}-{uuid.uuid4().hex[:10]}.casc")})
        path = result["path"]
    else:
        path = view.get_path_name()
        if not path or not Path(path).is_file():
            raise ValueError("Сначала сохраните текущую сцену или создайте её копию кнопкой рядом")
    return {"path": str(Path(path).resolve()), "frame": int(scene.get_current_frame()),
            "frames": int(scene.layers_viewer().frames_count())}


def _make_window():
    from PySide6.QtCore import (
        Property,
        QByteArray,
        QCoreApplication,
        QEvent,
        QMetaObject,
        QObject,
        Qt,
        QTimer,
        QUrl,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtNetwork import QNetworkAccessManager, QNetworkProxy, QNetworkRequest
    from PySide6.QtQml import QQmlComponent, QQmlEngine

    class Panel(QObject):
        changed = Signal()
        configChanged = Signal()
        sceneContextChanged = Signal()
        resultRequested = Signal()
        anchorsChanged = Signal(str)
        pathSelected = Signal(str, str)

        def __init__(self):
            super().__init__()
            self.network = QNetworkAccessManager(self)
            self.network.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
            self.requests = {}
            self.job_id, self.connected, self.active_job = None, False, None
            self.job_data, self.rows, self._jobs, self._config = {}, [], [], {}
            self.selected_candidate = ""
            self._scene_context = {"characters": [], "character_id": "", "rig": {}}
            self._rig_mapping = {}
            self._technical_log = False
            self.start_attempt, self.settings_loaded = 0.0, False
            self.shutting_down = False
            self._close_pending, self._close_reissued = False, False
            self._qml_cleanup = None
            self.notice, self.last_error, self.connection_label = "", False, "Подключение…"
            self._notice_is_task = False
            self.window, self.component, self.engine = None, None, None
            try:
                self.engine = QQmlEngine(self)
                self.engine.rootContext().setContextProperty("motion", self)
                self.component = QQmlComponent(self.engine, self)
                self.component.setData(QByteArray(QML_SOURCE.encode("utf-8")), QUrl("smart-motion.qml"))
                if self.component.isError():
                    raise RuntimeError("Smart Motion QML: " + "; ".join(error.toString() for error in self.component.errors()))
                self.window = self.component.create()
                if self.window is None:
                    raise RuntimeError("Smart Motion QML could not be created: " + "; ".join(
                        error.toString() for error in self.component.errors()))
            except Exception:
                from shiboken6 import isValid

                self.shutting_down = True
                self.network.blockSignals(True)
                self.network.deleteLater()
                _dispose_failed_panel(self, valid=isValid, queue_delete=lambda panel: QMetaObject.invokeMethod(
                    panel, "deleteLater", Qt.ConnectionType.QueuedConnection))
                raise
            self.window.visibleChanged.connect(self._visibility_changed)
            self.timer = QTimer(self)
            self.timer.setInterval(1100)
            self.timer.timeout.connect(self._poll)
            self._app = QCoreApplication.instance()
            self._quit_callback = self._shutdown
            self._app.aboutToQuit.connect(self._quit_callback)
            self._owner = None
            owners = [window for window in QGuiApplication.topLevelWindows()
                      if window.objectName() == "MainWindow"]
            if len(owners) == 1:
                self._owner = owners[0]
                self.window.setTransientParent(self._owner)
                self._owner.installEventFilter(self)

        @Property("QVariantMap", notify=configChanged)
        def config(self):
            return self._config

        @Property("QVariantMap", notify=sceneContextChanged)
        def sceneContext(self):
            return self._scene_context

        @Slot(str)
        def refreshScene(self, character_id=""):
            def execute():
                if self.active_job:
                    raise ValueError("Дождитесь завершения текущей задачи")
                context = _motion_context(character_id)
                if context["character_id"] != self._scene_context.get("character_id"):
                    self._rig_mapping = {}
                elif self._rig_mapping and context.get("rig"):
                    from . import common
                    from .humanoid import discover_mapping

                    mapping = dict(context["rig"].get("mapping") or {})
                    mapping.update(self._rig_mapping)
                    context["rig"] = discover_mapping(common.current_context()[3], context["character_id"], mapping)
                context["revision"] = int(self._scene_context.get("revision", 0)) + 1
                self._scene_context = context
                self.sceneContextChanged.emit()
            self._guard(execute)

        @Slot(str, str)
        def setRigBinding(self, role, identifier):
            from . import common
            from .humanoid import discover_mapping

            def execute():
                rig = self._scene_context.get("rig") or {}
                if (role not in {row["id"] for row in rig.get("roles", [])}
                        or identifier not in {row["id"] for row in rig.get("joints", [])}):
                    raise ValueError("Выберите кость текущего персонажа")
                self._rig_mapping[role] = identifier
                mapping = dict(rig.get("mapping") or {})
                mapping.update(self._rig_mapping)
                _, _, _, scene = common.current_context()
                mapping = discover_mapping(scene, self._scene_context["character_id"], mapping)
                self._scene_context = dict(self._scene_context, rig=mapping)
                if mapping.get("ready"):
                    _save_rig_profile(mapping)
                self.sceneContextChanged.emit()
            self._guard(execute)

        @Slot(int, int)
        def useRangeKeys(self, first, last):
            from . import characters, common

            def execute():
                _, _, _, scene = common.current_context()
                rig = characters.resolve_character(scene, self._scene_context.get("character_id"))
                layers = scene.layers_viewer()
                keys = sorted({first, last, *(int(frame) for identifier in rig.tracks
                               for frame in layers.layer(identifier).key_frame_indices() if first <= frame <= last)})
                self.anchorsChanged.emit(", ".join(map(str, keys)))
            self._guard(execute)

        @Property("QVariantMap", notify=changed)
        def job(self):
            return self.job_data

        @Property("QVariantList", notify=changed)
        def jobs(self):
            return self._jobs

        @Property("QVariantList", notify=changed)
        def candidates(self):
            values = []
            for row in self.rows:
                state = "Сохранена" if row.get("scene") else "NPZ / FBX" if row.get("output") else "Подготовка"
                check_names = {"geometry": "Геометрия", "persistence": "Сохранение", "constraints": "Позы"}
                checked = [label for name, label in check_names.items() if row.get("validation", {}).get(name) is True]
                values.append({"id": row["id"], "state": state, "checks": ", ".join(checked) or "Ожидание"})
            return values

        @Property(str, notify=changed)
        def selectedCandidate(self):
            return self.selected_candidate

        @Property(bool, notify=changed)
        def selectedHasScene(self):
            return any(row.get("scene") for row in self.rows if row["id"] == self.selected_candidate)

        @Property(bool, notify=changed)
        def canResume(self):
            return _can_resume(self.job_data)

        @Property(str, notify=changed)
        def connectionStatus(self):
            return self.connection_label

        @Property(bool, notify=changed)
        def isConnected(self):
            return self.connected

        @Property(bool, notify=changed)
        def busy(self):
            return bool(self.active_job) or any(method == "POST" and (route == "/jobs" or
                   re.search(r"/(?:select|resume)$", route)) for method, route in self.requests)

        @Property(str, notify=changed)
        def message(self):
            return self.notice

        @Property(bool, notify=changed)
        def messageIsError(self):
            return self.last_error

        @Property(bool, notify=changed)
        def messageIsTask(self):
            return self._notice_is_task

        @Property(bool, notify=changed)
        def technicalLogVisible(self):
            return self._technical_log

        @Slot(bool)
        def setTechnicalLog(self, visible):
            if self._technical_log != visible:
                self._technical_log = visible
                self.changed.emit()

        @Property("QVariantMap", notify=changed)
        def progress(self):
            return progress_display(self.job_data)

        @Property(str, notify=changed)
        def stageLabel(self):
            return _stage_label(self.job_data)

        @Property(str, notify=changed)
        def details(self):
            job = self.job_data
            row = next((item for item in self.rows if item["id"] == self.selected_candidate), {})
            run = job.get("run") or {}
            lines = []
            if run.get("frames"):
                lines.append(f"Длительность: {run['frames']} кадров")
            if row:
                lines.append("Сцена: " + str(row.get("scene") or "Только файлы анимации"))
                if not row.get("scene"):
                    lines.extend(f"{label}: {row[name]}" for label, name in
                                 (("NPZ", "output"), ("FBX", "output_fbx")) if row.get(name))
            if job.get("error"):
                lines.extend(["", "Ошибка: " + str(job["error"])])
            if self._technical_log and job.get("log_tail"):
                lines.extend(["", "Последние сообщения:", str(job["log_tail"])[-5000:]])
            return "\n".join(lines)

        def _message(self, text, *, error=False, task=False):
            self.notice, self.last_error = str(text)[:3500], error
            self._notice_is_task = task
            self.changed.emit()

        def _guard(self, action):
            try:
                return action()
            except Exception as error:
                self._message(str(error), error=True)
                return None

        @Slot(str)
        def submit(self, serialized):
            def execute():
                if self.busy:
                    raise ValueError("Дождитесь завершения текущей задачи")
                options = json.loads(serialized)
                options["character_id"] = self._scene_context.get("character_id", "")
                if not options.get("generate_only") and not options["character_id"]:
                    raise ValueError("Выберите персонажа в текущей сцене")
                mapping = dict((self._scene_context.get("rig") or {}).get("mapping") or {})
                mapping.update(self._rig_mapping)
                if mapping:
                    options["rig_mapping"] = mapping
                # Validate before creating a source copy; the submitted snapshot
                # includes current manual poses while the original stays intact.
                _panel_request(options, "pending-native-snapshot")
                snapshot = None if options.get("generate_only") else _scene_info(save_copy=True)["path"]
                payload = _panel_request(options, snapshot)
                def submitted(job):
                    self._started(job)
                    self.resultRequested.emit()
                self._request("POST", "/jobs", payload, submitted)
            self._guard(execute)

        @Slot(str)
        def saveSettings(self, serialized):
            self._guard(lambda: self._request("POST", "/settings", _settings_payload(json.loads(serialized)), self._settings_saved))

        def _settings_saved(self, result):
            self._settings_received(result)
            self._message("Настройки сохранены")

        @Slot()
        def refresh(self):
            self._refresh_jobs()
            self._request("GET", "/settings", callback=self._settings_received)

        @Slot(str)
        def selectJob(self, identifier):
            if re.fullmatch(r"[a-f0-9]{32}", identifier):
                self.job_id = identifier
                self._request("GET", "/jobs/" + identifier, callback=self._update_job)

        @Slot(str)
        def chooseCandidate(self, identifier):
            if any(row["id"] == identifier for row in self.rows):
                self.selected_candidate = identifier
                self.changed.emit()

        @Slot()
        def openCandidate(self):
            def execute():
                row = next((item for item in self.rows if item["id"] == self.selected_candidate), {})
                if not row.get("scene"):
                    raise ValueError("Выберите сохранённую анимацию")
                self._request("POST", f"/jobs/{self.job_id}/select", {"candidate_id": row["id"]},
                              self._started)
            self._guard(execute)

        @Slot()
        def resume(self):
            if self.job_id:
                self._guard(lambda: self._request("POST", f"/jobs/{self.job_id}/resume",
                                                 {"no_review": True, "no_render": True, "no_activate": False}, self._started))

        @Slot()
        def cancel(self):
            if self.active_job:
                self._request("POST", f"/jobs/{self.active_job['job_id']}/cancel", {},
                              lambda result: self._message("Отмена запрошена. Текущий GPU/native этап должен безопасно завершиться.", task=True))

        @Slot(str)
        def runtime(self, action):
            if action in {"start", "stop", "status", "logs", "build"}:
                def result(value):
                    if "job_id" in value:
                        self._started(value)
                    else:
                        self._message(json.dumps(value, ensure_ascii=False, indent=2), task=True)
                self._request("POST", "/runtime", {"action": action}, result)

        @Slot(str, str)
        def fileSelected(self, target, url):
            path = QUrl(url).toLocalFile()
            if path and target in {"checkout", "calibration", "template_motion", "template_scene", "source_scene", "constraints", "import_run"}:
                if target == "import_run":
                    self._request("POST", "/jobs/import", {"output": path}, self._started)
                else:
                    self.pathSelected.emit(target, path)

        @Slot(str, int, int)
        def addAnchor(self, text, first, last):
            from . import common

            def execute():
                _, _, _, scene = common.current_context()
                if not first <= int(scene.get_current_frame()) <= last:
                    raise ValueError("Текущий кадр находится за пределами выбранного диапазона")
                frames = [int(item) for item in re.split(r"[,;\s]+", text.strip()) if item]
                frames.append(int(scene.get_current_frame()))
                self.anchorsChanged.emit(", ".join(str(frame) for frame in sorted(set(frames))))
            self._guard(execute)

        @Slot()
        def openFolder(self):
            from PySide6.QtGui import QDesktopServices

            path = self.job_data.get("output")
            row = next((item for item in self.rows if item["id"] == self.selected_candidate), {})
            if not row.get("scene") and row.get("output_fbx"):
                path = str(Path(row["output_fbx"]).parent)
            if path and Path(path).is_dir():
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))

        @Slot()
        def play(self):
            from . import common

            def execute():
                if self.active_job:
                    raise ValueError("Дождитесь завершения задачи перед просмотром")
                row = next((item for item in self.rows if item["id"] == self.selected_candidate), {})
                info = _scene_info()
                if not row.get("scene") or Path(row["scene"]).resolve() != Path(info["path"]):
                    raise ValueError("Сначала откройте выбранную анимацию")
                _, _, _, scene = common.current_context()
                _native_call("playback_start", _preview_request(
                    self.job_data.get("run") or {}, scene.data_viewer().get_animation_size()))
                self._message("Нативное воспроизведение запущено; оно автоматически остановится после просмотра.", task=True)
            self._guard(execute)

        @Slot()
        def stopPlayback(self):
            self._guard(lambda: (_native_call("playback_stop", {}), self._message("Остановка воспроизведения запрошена", task=True)))

        def _request(self, method, route, payload=None, callback=None, *, quiet=False):
            key = (method, route)
            if key in self.requests:
                return
            try:
                endpoint, token = _connection()
            except Exception:
                if not quiet:
                    self._message("Сервис ещё не готов; дождитесь подключения", error=True)
                return
            request = QNetworkRequest(QUrl(endpoint + route))
            request.setRawHeader(QByteArray(b"X-Smart-Motion-Token"), QByteArray(token.encode("ascii")))
            request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
            request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                                 QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
            request.setTransferTimeout(12000)
            if method == "GET":
                reply = self.network.get(request)
            else:
                reply = self.network.post(request, QByteArray(json.dumps(payload or {}, allow_nan=False).encode("utf-8")))
            self.requests[key] = reply
            self.changed.emit()

            def finished():
                self.requests.pop(key, None)
                if self.shutting_down:
                    return
                self.changed.emit()
                try:
                    raw = bytes(reply.readAll())
                    if len(raw) > 4 * 1024 * 1024:
                        raise ValueError("Ответ сервиса слишком велик")
                    value = json.loads(raw) if raw else {}
                    if value.get("ok") is not True:
                        error = value.get("error", {})
                        message = error.get("message") if isinstance(error, dict) else str(error)
                        raise ValueError(message or "Сервис недоступен; операция не была повторена")
                    if callback:
                        callback(value["result"])
                except Exception as error:
                    if route == "/status":
                        self.connected = False
                        self.connection_label = "Нет соединения"
                        self.changed.emit()
                    if not quiet:
                        self._message(str(error), error=True, task=route not in {"/jobs", "/status"})
                finally:
                    reply.deleteLater()
            reply.finished.connect(finished)

        def _poll(self):
            if self.shutting_down or not self.isVisible():
                return
            if not self.connected and time.monotonic() - self.start_attempt >= 15:
                self.start_attempt = time.monotonic()
                try:
                    _start_service()
                except Exception as error:
                    self._message(str(error), error=True)
            try:
                _connection()
            except (OSError, ValueError, KeyError):
                return
            self._request("GET", "/status", callback=self._status, quiet=True)
            if self.job_id:
                self._request("GET", "/jobs/" + self.job_id, callback=self._update_job, quiet=True)

        def _status(self, value):
            first = not self.connected
            self.connected = True
            self.connection_label = "Подключено · локальный сервис"
            self.active_job = value.get("active_job")
            if self.active_job and not self.job_id:
                self.job_id = self.active_job["job_id"]
            if not self.settings_loaded:
                self.settings_loaded = True
                self._request("GET", "/settings", callback=self._settings_received)
                self._refresh_jobs()
            if first and not value.get("configured", False):
                self._message("Проверьте подключение Kimodo на вкладке «Сервис»", error=True)
            self.changed.emit()

        def _settings_received(self, settings):
            self._config = settings
            self.configChanged.emit()
            self.changed.emit()

        def _started(self, job):
            self.job_id = job["job_id"]
            if job.get("state") in {"queued", "running", "cancel_requested"}:
                self.active_job = job
            self._update_job(job)
            self._refresh_jobs()
            completed = _completion_notice(job)
            if completed:
                self._message(completed[0], error=completed[1], task=True)
            else:
                self._message("Задача принята. После обработки откроется сохранённая анимация.", task=True)

        def _refresh_jobs(self):
            self._request("GET", "/jobs", callback=self._jobs_received)

        def _jobs_received(self, value):
            self._jobs = []
            for job in value.get("jobs", []):
                self._jobs.append(dict(job, label=_job_label(job)))
            if not self.job_id and self._jobs:
                self.job_id = self._jobs[0]["job_id"]
                self._request("GET", "/jobs/" + self.job_id, callback=self._update_job)
            self.changed.emit()

        def _update_job(self, job):
            if job.get("job_id") != self.job_id:
                return
            previous_state = self.job_data.get("state")
            self.job_data = job
            self.rows = candidate_rows(job)
            if not any(row["id"] == self.selected_candidate for row in self.rows):
                self.selected_candidate = next((row["id"] for row in self.rows if row["selected"]),
                                               self.rows[0]["id"] if self.rows else "")
            self.changed.emit()
            if previous_state and previous_state != job.get("state"):
                notice = _notice_after_update(previous_state, job, task_notice=self._notice_is_task)
                if notice:
                    self._message(notice[0], error=notice[1], task=True)
                self._refresh_jobs()

        def _visibility_changed(self, visible):
            if self.shutting_down:
                return
            if visible:
                self.timer.start()
                self._poll()
            else:
                self.timer.stop()

        def eventFilter(self, watched, event):
            if watched == self._owner and event.type() == QEvent.Type.Close:
                event.ignore()
                self._close_pending = True
                self._shutdown()
                return True
            return False

        def _shutdown(self):
            from shiboken6 import isValid

            if self.shutting_down:
                return
            self.shutting_down = True
            if self._app is not None and isValid(self._app):
                try:
                    self._app.aboutToQuit.disconnect(self._quit_callback)
                except (RuntimeError, TypeError):
                    pass
            self._app, self._quit_callback = None, None
            if self.timer is not None and isValid(self.timer):
                self.timer.blockSignals(True)
                self.timer.stop()
                self.timer.deleteLater()
            for reply in list(self.requests.values()):
                if isValid(reply):
                    reply.blockSignals(True)
                    reply.abort()
            self.requests.clear()
            if self.network is not None and isValid(self.network):
                self.network.blockSignals(True)
                self.network.deleteLater()
            if self.window is not None and isValid(self.window):
                self.window.hide()
            self._qml_cleanup = _DeferredQmlDisposal(
                (self.window, self.component, self.engine), valid=isValid, finished=self._shutdown_finished)
            self._qml_cleanup.start()

        def _shutdown_finished(self):
            from shiboken6 import isValid

            self.window, self.component, self.engine = None, None, None
            self.timer, self.network = None, None
            owner, self._owner = self._owner, None
            if owner is not None and isValid(owner):
                owner.removeEventFilter(self)
                if self._close_pending and not self._close_reissued:
                    self._close_reissued = True
                    # Return from the engine destructor before native close.
                    # This queues C++ work without another Python timer callback.
                    if not QMetaObject.invokeMethod(owner, "close", Qt.ConnectionType.QueuedConnection):
                        self._message("Не удалось повторить закрытие Cascadeur; закройте главное окно ещё раз", error=True)

        def show(self):
            if not self._scene_context.get("characters"):
                self.refreshScene("")
            self.window.show()

        def close(self):
            self.window.hide()

        def isVisible(self):
            return not self.shutting_down and self.window is not None and self.window.isVisible()

        def windowTitle(self):
            return self.window.title() if not self.shutting_down and self.window is not None else "Smart Motion"

        def raise_(self):
            self.window.raise_()

        def activateWindow(self):
            self.window.requestActivate()

    return Panel()


def show_panel():
    """Show one QtQuick window on Cascadeur's existing Qt application thread."""
    global _window
    from PySide6.QtCore import QCoreApplication, QThread

    application = QCoreApplication.instance()
    if application is None or application.thread() != QThread.currentThread():
        raise RuntimeError("Open Smart Motion on Cascadeur's Qt application thread")
    if _window is not None and _window.shutting_down and _window._qml_cleanup and not _window._qml_cleanup.done:
        return dict(panel_state(), closing=True)
    if _window is None or _window.shutting_down:
        _window = _make_window()
    _window.show()
    _window.raise_()
    _window.activateWindow()
    return panel_state()


def panel_state():
    if _window is None:
        return {"visible": False, "title": "Smart Motion", "scene_changes": False}
    return {"visible": _window.isVisible(), "title": _window.windowTitle(), "connected": _window.connected,
            "job_id": _window.job_id, "candidate_count": len(_window.rows), "scene_changes": False}


QML_SOURCE = r"""
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: window
    objectName: "SmartMotionPanel"
    title: "Smart Motion — Cascadeur"
    width: 920; height: 790
    minimumWidth: 760; minimumHeight: 640
    visible: false
    palette.window: "#25272b"; palette.windowText: "#eceef1"; palette.text: "#eceef1"
    palette.base: "#202226"; palette.button: "#393c43"; palette.buttonText: "#eceef1"
    palette.highlight: "#4d86ca"; palette.highlightedText: "white"
    property var settings: motion.config
    property var defaults: settings.defaults || ({})
    property var context: motion.sceneContext
    property var rig: context.rig || ({})
    property int sourceRevision: -1
    function saved(key, fallback) { return defaults[key] === undefined ? fallback : defaults[key] }
    function characterIndex() {
        var rows = context.characters || []
        for (var i = 0; i < rows.length; ++i) if (rows[i].id === context.character_id) return i
        return -1
    }
    function bindingIndex(role) {
        var rows = rig.joints || [], mapping = rig.mapping || ({})
        for (var i = 0; i < rows.length; ++i) if (rows[i].id === mapping[role]) return i
        return -1
    }
    function generationOptions() {
        var data = {prompt: prompt.text.trim(), first_frame: firstFrame.value, last_frame: lastFrame.value,
                    samples: 1, sample_index: 0, no_review: true, no_render: true, no_activate: false,
                    steps: steps.value, text_weight: Number(textWeight.text),
                    constraint_weight: Number(constraintWeight.text), no_cfg: !cfg.checked,
                    timeout: timeout.value, port: port.value, generate_only: rawExport.checked,
                    use_keyposes: useKeyposes.checked, keyframes: keys.text}
        if (seed.text.trim()) data.seed = Number(seed.text)
        if (rawExport.checked && constraints.text.trim()) data.constraints = constraints.text.trim()
        return data
    }
    function generate() { motion.submit(JSON.stringify(generationOptions())) }
    function saveSettings() {
        var data = generationOptions()
        delete data.prompt; delete data.first_frame; delete data.last_frame
        delete data.use_keyposes; delete data.keyframes; delete data.constraints
        motion.saveSettings(JSON.stringify({checkout: checkout.text.trim(), template_motion: templateMotion.text.trim(), defaults: data}))
    }
    function browse(field, folder) {
        if (folder) { folderPicker.targetField = field; folderPicker.open() }
        else { filePicker.targetField = field; filePicker.nameFilters = field === "constraints" ? ["NumPy (*.npz)"] : ["JSON (*.json)"]; filePicker.open() }
    }
    Connections {
        target: motion
        function onSceneContextChanged() {
            if (context.revision === undefined || sourceRevision === context.revision) return
            sourceRevision = context.revision
            firstFrame.value = Math.max(0, context.first || 0)
            lastFrame.value = Math.max(firstFrame.value + 1, context.last || 119)
            keys.text = (context.keys || []).join(", ")
        }
        function onAnchorsChanged(text) { keys.text = text }
        function onResultRequested() { tabs.currentIndex = 1 }
        function onPathSelected(field, path) {
            if (field === "constraints") constraints.text = path
            else if (field === "checkout") checkout.text = path
            else if (field === "template_motion") templateMotion.text = path
        }
    }
    FileDialog { id: filePicker; property string targetField; title: "Выбрать файл"; onAccepted: motion.fileSelected(targetField, selectedFile.toString()) }
    FolderDialog { id: folderPicker; property string targetField; title: targetField === "import_run" ? "Каталог сохранённой анимации" : "Каталог проекта"; onAccepted: motion.fileSelected(targetField, selectedFolder.toString()) }
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 18; spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Label { text: "Smart Motion"; font.pixelSize: 26; font.bold: true; Layout.fillWidth: true }
            Label { objectName: "motionConnectionStatus"; text: motion.connectionStatus; font.pixelSize: 12; color: motion.isConnected ? "#a6c8ac" : "#dfc191" }
        }
        TabBar {
            id: tabs; Layout.fillWidth: true
            TabButton { objectName: "motionTabGeneration"; text: "Создать" }
            TabButton { objectName: "motionTabVariants"; text: "Результаты" }
            TabButton { objectName: "motionTabSettings"; text: "Сервис" }
        }
        StackLayout {
            currentIndex: tabs.currentIndex; Layout.fillWidth: true; Layout.fillHeight: true
            ScrollView {
                id: generateScroll; clip: true; contentWidth: availableWidth
                ColumnLayout {
                    width: generateScroll.availableWidth; spacing: 12
                    Label { text: useKeyposes.checked && !rawExport.checked ? "Описание движения · необязательно" : "Какое движение создать?"; font.bold: true }
                    ScrollView {
                        Layout.fillWidth: true; Layout.preferredHeight: 100
                        TextArea { id: prompt; objectName: "motionPrompt"; wrapMode: TextEdit.Wrap; color: "#eceef1";
                            placeholderTextColor: "#9da5b2"; selectByMouse: true;
                            placeholderText: useKeyposes.checked && !rawExport.checked ? "Можно уточнить характер движения или оставить поле пустым" : "Например: человек медленно идёт вперёд, неся тяжёлый рюкзак";
                            background: Rectangle { color: "#1d1f23"; radius: 4; border.color: prompt.activeFocus ? "#79a5dc" : "#555c68" }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "Персонаж"; Layout.preferredWidth: 95 }
                        ComboBox { id: character; objectName: "motionCharacter"; Layout.fillWidth: true;
                            model: context.characters || []; textRole: "display_name"; valueRole: "id"; currentIndex: characterIndex();
                            displayText: currentIndex < 0 ? "Выберите персонажа в сцене" : currentText;
                            enabled: !motion.busy && !rawExport.checked; onActivated: motion.refreshScene(currentValue)
                        }
                        Button { objectName: "motionRefreshScene"; text: "Обновить из сцены"; enabled: !motion.busy; onClicked: motion.refreshScene("") }
                    }
                    Label { text: context.scene_name || "Откройте сцену с персонажем"; color: "#aeb6c3"; elide: Text.ElideMiddle; Layout.fillWidth: true }
                    Label { visible: !rawExport.checked && !(context.characters || []).length; text: "Откройте персонажа с готовым ригом Cascadeur или сначала настройте его риг."; color: "#dfc191"; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "Диапазон"; Layout.preferredWidth: 95 }
                        SpinBox { id: firstFrame; objectName: "motion_first_frame"; from: 0; to: 2147483646; value: 0; editable: true; Layout.fillWidth: true }
                        Label { text: "—" }
                        SpinBox { id: lastFrame; objectName: "motion_last_frame"; from: firstFrame.value + 1; to: 2147483647; value: 119; editable: true; Layout.fillWidth: true }
                        Button { objectName: "motionUseTimeline"; text: "Взять с таймлайна"; enabled: !motion.busy; onClicked: motion.refreshScene(context.character_id || "") }
                    }
                    Label { text: (lastFrame.value - firstFrame.value + 1) + " кадров · " + ((lastFrame.value - firstFrame.value + 1) / 30).toFixed((lastFrame.value - firstFrame.value + 1) % 30 === 0 ? 0 : 2).replace(".", ",") + " с · 30 FPS · новая сцена с кадра 0"; color: "#aeb6c3"; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    CheckBox { id: useKeyposes; objectName: "motion_use_keyposes"; text: "Движение между ключевыми позами"; enabled: !rawExport.checked }
                    ColumnLayout {
                        visible: useKeyposes.checked && !rawExport.checked; Layout.fillWidth: true; spacing: 8
                        Label { text: "Начало и конец диапазона сохраняются автоматически. Добавьте другие важные позы."; wrapMode: Text.WordWrap; Layout.fillWidth: true; color: "#b8c4d3" }
                        RowLayout {
                            Button { objectName: "motionUseRangeKeys"; text: "Взять ключевые кадры"; enabled: !motion.busy; onClicked: motion.useRangeKeys(firstFrame.value, lastFrame.value) }
                            Button { objectName: "motionAddAnchor"; text: "Добавить текущую позу"; enabled: !motion.busy; onClicked: motion.addAnchor(keys.text, firstFrame.value, lastFrame.value) }
                            Button { text: "Очистить"; onClicked: keys.text = "" }
                        }
                        TextField { id: keys; objectName: "motion_keyframes"; Layout.fillWidth: true; placeholderText: "Начало и конец добавляются автоматически"; placeholderTextColor: "#9da5b2" }
                    }
                    Label { visible: !rawExport.checked && context.character_id && !(rig.ready || false); text: "Сопоставьте кости персонажа в разделе «Настроить скелет»."; color: "#dfc191"; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    CheckBox { id: showMapping; objectName: "motionShowRigMapping"; text: "Настроить скелет"; visible: !rawExport.checked && (rig.roles || []).length > 0 }
                    ColumnLayout {
                        visible: showMapping.checked; Layout.fillWidth: true; spacing: 7
                        Label { text: "Назначьте кость персонажа каждой части тела. Исходный скелет сохраняет свои имена."; wrapMode: Text.WordWrap; Layout.fillWidth: true; color: "#b8c4d3" }
                        Repeater {
                            model: rig.roles || []
                            RowLayout {
                                id: bindingRow; required property var modelData
                                Layout.fillWidth: true
                                Label { text: bindingRow.modelData.label; Layout.preferredWidth: 180 }
                                ComboBox { objectName: "motionBinding_" + bindingRow.modelData.id; Layout.fillWidth: true;
                                    model: rig.joints || []; textRole: "name"; valueRole: "id";
                                    currentIndex: bindingIndex(bindingRow.modelData.id); displayText: currentIndex < 0 ? "Выберите кость" : currentText;
                                    onActivated: motion.setRigBinding(bindingRow.modelData.id, currentValue)
                                }
                            }
                        }
                        Label { text: (rig.issues || []).join("\n"); visible: text.length > 0; color: "#dfc191"; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                    CheckBox { id: advanced; objectName: "motionAdvanced"; text: "Дополнительные параметры" }
                    ColumnLayout {
                        visible: advanced.checked; Layout.fillWidth: true; spacing: 10
                        GridLayout {
                            columns: 4; Layout.fillWidth: true; columnSpacing: 12; rowSpacing: 8
                            Label { text: "Seed" }
                            TextField { id: seed; objectName: "motion_seed"; placeholderText: "Автоматически"; placeholderTextColor: "#9da5b2"; Layout.fillWidth: true; validator: RegularExpressionValidator { regularExpression: /[0-9]{0,10}/ } }
                            Label { text: "Шаги" }
                            SpinBox { id: steps; objectName: "motion_steps"; from: 1; to: 200; value: saved("steps", 50); editable: true; Layout.fillWidth: true }
                            Label { text: "Следование описанию" }
                            TextField { id: textWeight; objectName: "motion_text_weight"; text: String(saved("text_weight", 2)); Layout.fillWidth: true; validator: DoubleValidator { bottom: 0; top: 10; decimals: 2 } }
                            Label { text: "Сила опорных поз" }
                            TextField { id: constraintWeight; objectName: "motion_constraint_weight"; text: String(saved("constraint_weight", 2)); Layout.fillWidth: true; validator: DoubleValidator { bottom: 0; top: 10; decimals: 2 } }
                            CheckBox { id: cfg; objectName: "motion_cfg"; text: "Усилить следование описанию (CFG)"; checked: !saved("no_cfg", false); Layout.columnSpan: 4 }
                        }
                        CheckBox { id: rawExport; objectName: "motionRawExport"; text: "Только файлы NPZ / FBX, без переноса в сцену"; checked: false }
                        RowLayout {
                            visible: rawExport.checked; Layout.fillWidth: true
                            TextField { id: constraints; objectName: "motion_constraints"; Layout.fillWidth: true; placeholderText: "Ограничения NPZ, необязательно"; placeholderTextColor: "#9da5b2" }
                            Button { text: "Обзор…"; onClicked: browse("constraints", false) }
                        }
                    }
                    Item { Layout.preferredHeight: 6 }
                }
            }
            ColumnLayout {
                spacing: 10
                RowLayout {
                    Layout.fillWidth: true
                    ComboBox { id: history; objectName: "motionJobHistory"; Layout.fillWidth: true; model: motion.jobs; textRole: "label"; valueRole: "job_id"; onActivated: motion.selectJob(currentValue) }
                    Button { objectName: "motionImportRun"; text: "Открыть сохранённую…"; onClicked: browse("import_run", true) }
                    Button { text: "Обновить"; onClicked: motion.refresh() }
                }
                ScrollView {
                    id: resultsScroll; objectName: "motionCandidates"; Layout.fillWidth: true; Layout.preferredHeight: Math.min(180, Math.max(68, motion.candidates.length * 58)); clip: true; contentWidth: availableWidth
                    ColumnLayout {
                        width: resultsScroll.availableWidth; spacing: 5
                        Repeater {
                            model: motion.candidates
                            Frame {
                                id: resultRow; required property var modelData
                                Layout.fillWidth: true; Layout.preferredHeight: 54; padding: 9
                                background: Rectangle { color: motion.selectedCandidate === resultRow.modelData.id ? "#3e5876" : "#303339"; radius: 4 }
                                RowLayout {
                                    anchors.fill: parent
                                    Label { text: modelData.state; font.bold: true; Layout.preferredWidth: 130 }
                                    Label { text: modelData.checks; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                                }
                                MouseArea { anchors.fill: parent; onClicked: motion.chooseCandidate(resultRow.modelData.id) }
                            }
                        }
                        Label { visible: motion.candidates.length === 0; text: "Здесь появится сохранённая анимация."; Layout.fillWidth: true; color: "#aeb6c3" }
                    }
                }
                ScrollView { Layout.fillWidth: true; Layout.fillHeight: true; TextArea { objectName: "motionResultDetails"; text: motion.details; readOnly: true; wrapMode: TextEdit.Wrap; selectByMouse: true } }
                Flow {
                    Layout.fillWidth: true; spacing: 8
                    Button { objectName: "motionOpenCandidate"; text: "Открыть анимацию"; enabled: !motion.busy && motion.selectedHasScene; onClicked: motion.openCandidate() }
                    Button { objectName: "motionPlay"; text: "Воспроизвести"; enabled: !motion.busy && motion.selectedHasScene; onClicked: motion.play() }
                    Button { objectName: "motionStopPlayback"; text: "Остановить"; onClicked: motion.stopPlayback() }
                    Button { objectName: "motionResume"; text: "Продолжить после сбоя"; visible: motion.canResume && !motion.busy; onClicked: motion.resume() }
                    Button { text: "Папка результата"; onClicked: motion.openFolder() }
                }
                CheckBox { objectName: "motionTechnicalLog"; text: "Показать журнал"; checked: motion.technicalLogVisible; onToggled: motion.setTechnicalLog(checked) }
            }
            ScrollView {
                id: settingsScroll; clip: true; contentWidth: availableWidth
                ColumnLayout {
                    width: settingsScroll.availableWidth; spacing: 12
                    Label { text: "Подключение Kimodo"; font.bold: true }
                    Label { text: "Генератор запускается автоматически. После обработки GPU освобождается."; wrapMode: Text.WordWrap; Layout.fillWidth: true; color: "#b8c4d3" }
                    GridLayout {
                        columns: 3; Layout.fillWidth: true; columnSpacing: 10; rowSpacing: 10
                        Label { text: "Каталог проекта" }
                        TextField { id: checkout; objectName: "motionPath_checkout"; text: settings.checkout || ""; Layout.fillWidth: true }
                        Button { text: "Обзор…"; onClicked: browse("checkout", true) }
                        Label { text: "Шаблон скелета Kimodo" }
                        TextField { id: templateMotion; objectName: "motionPath_template_motion"; text: settings.template_motion || ""; Layout.fillWidth: true }
                        Button { text: "Обзор…"; onClicked: browse("template_motion", false) }
                        Label { text: "Ожидание генерации, с" }
                        SpinBox { id: timeout; objectName: "motion_timeout"; from: 30; to: 86400; value: saved("timeout", 3600); editable: true; Layout.columnSpan: 2; Layout.fillWidth: true }
                        Label { text: "Порт Kimodo" }
                        SpinBox { id: port; objectName: "motion_port"; from: 1024; to: 65535; value: saved("port", 9552); editable: true; Layout.columnSpan: 2; Layout.fillWidth: true }
                    }
                    Button { objectName: "motionSaveSettings"; text: "Сохранить настройки"; enabled: !motion.busy; Layout.fillWidth: true; onClicked: saveSettings() }
                    Flow {
                        Layout.fillWidth: true; spacing: 8
                        Button { objectName: "motionRuntime_status"; text: "Состояние"; onClicked: motion.runtime("status") }
                        Button { objectName: "motionRuntime_start"; text: "Запустить Kimodo"; enabled: !motion.busy; onClicked: motion.runtime("start") }
                        Button { objectName: "motionRuntime_stop"; text: "Освободить GPU"; enabled: !motion.busy; onClicked: motion.runtime("stop") }
                        Button { objectName: "motionRuntime_logs"; text: "Журнал сервиса"; onClicked: motion.runtime("logs") }
                    }
                    CheckBox { id: maintenance; text: "Обслуживание Docker" }
                    Button { visible: maintenance.checked; objectName: "motionRuntime_build"; text: "Пересобрать Docker-образ"; enabled: !motion.busy; onClicked: motion.runtime("build") }
                }
            }
        }
        Label { objectName: "motionStage"; text: tabs.currentIndex === 0 && !motion.busy ? "Готово к генерации" : motion.stageLabel; wrapMode: Text.WordWrap; Layout.fillWidth: true }
        RowLayout {
            Layout.fillWidth: true
            Item { visible: !motion.busy; Layout.fillWidth: true }
            ProgressBar { objectName: "motionProgress"; visible: motion.busy; Layout.fillWidth: true; from: 0; to: 1;
                indeterminate: motion.busy && motion.progress.fraction === null;
                value: motion.progress.fraction === null ? (motion.job.state === "completed" ? 1 : 0) : motion.progress.fraction }
            Button { objectName: "motionGenerate"; text: "Создать анимацию"; visible: tabs.currentIndex === 0;
                enabled: motion.isConnected && !motion.busy && (prompt.text.trim().length > 0 || (useKeyposes.checked && !rawExport.checked)) && (rawExport.checked || (context.character_id && rig.ready)); highlighted: true; onClicked: generate() }
            Button { objectName: "motionCancel"; text: "Отменить"; visible: motion.busy; onClicked: motion.cancel() }
        }
        Label { objectName: "motionNotice"; text: motion.message; visible: tabs.currentIndex !== 0 || motion.busy || !motion.messageIsTask; wrapMode: Text.WordWrap; Layout.fillWidth: true;
            maximumLineCount: 4; elide: Text.ElideRight; color: motion.messageIsError ? "#e8a3a3" : "#c8cbd0" }
    }
    onClosing: function(close) { close.accepted = false; window.hide() }
}
"""
