import json
import threading
import time

from sctag import serve


def test_save_label_merges_and_removes(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "LABELS", tmp_path / "labels.json")
    state = serve.State()
    state.save_label({"id": 1, "genres": ["funk"]})
    state.save_label({"id": 1, "energy": "calm"})
    assert json.loads((tmp_path / "labels.json").read_text()) == {"1": {"genres": ["funk"], "energy": "calm"}}
    state.save_label({"id": 1, "genres": None, "energy": None})
    assert serve.read_labels() == {}


def wait(state):
    for _ in range(200):
        if not state.refreshing:
            return state.job
        time.sleep(0.01)
    raise AssertionError("refresh never finished")


def test_refresh_job(monkeypatch):
    from sctag import analyze, soundcloud

    gate = threading.Event()

    def fake_analyze(kind, load_models=None, progress=None):
        gate.wait(1)
        progress(kind, 1, 2)
        return [({"id": 7}, "err"), ({"id": 99}, "old DRM failure")] if kind == "tracks" else []

    monkeypatch.setattr(soundcloud, "save_likes", lambda username: {7, 8})
    monkeypatch.setattr(analyze, "analyze", fake_analyze)
    state = serve.State()
    rebuilt = []
    monkeypatch.setattr(state, "rebuild", rebuilt.append)
    assert state.start_refresh("someone")
    assert not state.start_refresh("someone")  # one at a time
    gate.set()
    job = wait(state)
    assert job["state"] == "done"
    assert (job["new"], job["failed"]) == (2, 1)  # only failures among the new likes count
    assert job["fetched"] and job["rebuilds"] == {"tracks": 1, "sets": 1}
    assert rebuilt == [["tracks"], ["sets"]]  # each page gets its tags as soon as its kind is analyzed


def test_refresh_rebuilds_during_long_analysis(monkeypatch):
    from sctag import analyze, soundcloud

    def fake_analyze(kind, load_models=None, progress=None):
        progress(kind, 1, 2)
        progress(kind, 2, 2)
        return []

    monkeypatch.setattr(soundcloud, "save_likes", lambda username: set())
    monkeypatch.setattr(analyze, "analyze", fake_analyze)
    monkeypatch.setattr(serve, "REBUILD_EVERY_S", 0)
    state = serve.State()
    monkeypatch.setattr(state, "rebuild", lambda kinds: None)
    assert state.start_refresh("someone")
    assert wait(state)["rebuilds"] == {"tracks": 3, "sets": 3}  # after each like here, plus once at the end


def test_refresh_failure_is_recorded(monkeypatch):
    from sctag import soundcloud

    def boom(username):
        raise RuntimeError("no such user")

    monkeypatch.setattr(soundcloud, "save_likes", boom)
    state = serve.State()
    assert state.start_refresh("nobody")
    job = wait(state)
    assert job["state"] == "failed" and "no such user" in job["error"]
    assert state.start_refresh("nobody")  # can retry after a failure
    wait(state)


def test_settings_round_trip(tmp_path, monkeypatch):
    from sctag import config

    monkeypatch.setattr(config, "SETTINGS", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DATA", tmp_path)
    config.save_settings(username="someone")
    config.save_settings(liked_since="2016-10-27")
    assert config.settings() == {"username": "someone", "liked_since": "2016-10-27"}
