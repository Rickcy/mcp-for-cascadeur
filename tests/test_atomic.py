import os
import threading

import pytest

from mcp_for_cascadeur.atomic import replace_file


def test_transient_windows_sharing_failure_preserves_the_new_bytes(tmp_path, monkeypatch):
    temporary, target = tmp_path / "value.tmp", tmp_path / "value.json"
    temporary.write_bytes(b"new")
    target.write_bytes(b"old")
    original = type(temporary).replace
    attempts = []

    def sharing_lock(path, destination):
        attempts.append(path)
        if len(attempts) < 3:
            error = PermissionError("sharing violation")
            error.winerror = 5
            raise error
        return original(path, destination)

    monkeypatch.setattr(type(temporary), "replace", sharing_lock)
    replace_file(temporary, target)
    assert len(attempts) == 3
    assert target.read_bytes() == b"new" and not temporary.exists()


def test_an_unrelated_error_is_not_retried(tmp_path, monkeypatch):
    attempts = []

    def denied(path, destination):
        attempts.append(path)
        raise PermissionError("ordinary access denial")

    monkeypatch.setattr(type(tmp_path), "replace", denied)
    with pytest.raises(PermissionError):
        replace_file(tmp_path / "value.tmp", tmp_path / "value.json")
    assert len(attempts) == 1


def test_permanent_sharing_denial_is_bounded_and_keeps_both_files(tmp_path, monkeypatch):
    temporary, target = tmp_path / "value.tmp", tmp_path / "value.json"
    temporary.write_bytes(b"new")
    target.write_bytes(b"old")

    def denied(path, destination):
        error = PermissionError("still open")
        error.winerror = 32
        raise error

    monkeypatch.setattr(type(tmp_path), "replace", denied)
    with pytest.raises(PermissionError):
        replace_file(temporary, target, timeout=0.01)
    assert temporary.read_bytes() == b"new" and target.read_bytes() == b"old"


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_a_real_windows_reader_can_finish_before_the_atomic_publish(tmp_path):
    temporary, target = tmp_path / "value.tmp", tmp_path / "value.json"
    temporary.write_bytes(b"new")
    target.write_bytes(b"old")
    locked, release = threading.Event(), threading.Event()

    def read():
        with target.open("rb") as stream:
            assert stream.read() == b"old"
            locked.set()
            release.wait(1)

    reader = threading.Thread(target=read)
    reader.start()
    assert locked.wait(1)
    timer = threading.Timer(0.05, release.set)
    timer.start()
    try:
        replace_file(temporary, target)
        assert target.read_bytes() == b"new"
    finally:
        release.set()
        timer.join()
        reader.join()
