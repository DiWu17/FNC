import json
from pathlib import Path

import pytest

from clickagent import action_benchmark


def test_atomic_json_retries_transient_windows_reader_lock(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    target.write_text('{"previous":true}', encoding="utf-8")
    original_replace = Path.replace
    attempts, waits = [], []

    def sharing_violation_then_replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) <= 2:
            raise PermissionError("Windows sharing violation")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", sharing_violation_then_replace)
    monkeypatch.setattr(action_benchmark.time, "sleep", waits.append)
    action_benchmark.write_json(target, {"completed": 164})
    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 164}
    assert len(attempts) == 3 and waits == [.05, .1]
    assert not target.with_suffix(".json.tmp").exists()


def test_persistent_lock_raises_and_retains_complete_temporary_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    target.write_text('{"previous":true}', encoding="utf-8")
    attempts, waits = [], []

    def always_locked(source, destination):
        attempts.append(source)
        raise PermissionError("persistent lock")

    monkeypatch.setattr(Path, "replace", always_locked)
    monkeypatch.setattr(action_benchmark.time, "sleep", waits.append)
    with pytest.raises(PermissionError, match="persistent lock"):
        action_benchmark.write_json(target, {"completed": 164})
    assert len(attempts) == 8 and len(waits) == 7
    assert json.loads(target.read_text(encoding="utf-8")) == {"previous": True}
    assert json.loads(target.with_suffix(".json.tmp").read_text(encoding="utf-8")) == {"completed": 164}
