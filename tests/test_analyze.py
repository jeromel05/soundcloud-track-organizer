import pytest

from sctag.analyze import energy_summary, kind_of, window_starts
from sctag.config import ENERGY_COOL, ENERGY_HOT, SET_MAX_WINDOWS, WIN_S


def test_kind_by_duration_with_overrides():
    assert kind_of({"id": 1, "duration_s": 200}, {}) == "tracks"
    assert kind_of({"id": 1, "duration_s": 3600}, {}) == "sets"
    assert kind_of({"id": 1, "duration_s": 200}, {"1": "sets"}) == "sets"


def test_track_windows():
    assert window_starts(200, "tracks") == [50.0, 100.0, 150.0]
    assert window_starts(30, "tracks") == [0.0]


@pytest.mark.parametrize("duration", [16 * 60, 3600, 6 * 3600])
def test_set_windows_cover_the_whole_set(duration):
    starts = window_starts(duration, "sets")
    assert 4 <= len(starts) <= SET_MAX_WINDOWS
    assert starts == sorted(starts) and starts[0] > 0 and starts[-1] <= duration - WIN_S


def test_energy_labels():
    hot, cool, mid = ENERGY_HOT + 0.1, ENERGY_COOL - 0.1, (ENERGY_HOT + ENERGY_COOL) / 2
    assert energy_summary([hot] * 9)["energy"] == "energetic"
    assert energy_summary([cool] * 9)["energy"] == "calm"
    assert energy_summary([hot, mid, cool] * 3)["energy"] == "mixed"


def test_energy_arc_of_mixed_sets():
    hot, cool = ENERGY_HOT + 0.1, ENERGY_COOL - 0.1
    assert energy_summary([cool] * 3 + [hot] * 3)["energy_arc"] == "builds up"
    assert energy_summary([hot] * 3 + [cool] * 3)["energy_arc"] == "winds down"


def test_waiting_lists_unanalyzed_likes_of_one_kind(tmp_path, monkeypatch):
    import json

    from sctag import analyze

    likes = [
        {"id": 1, "duration_s": 200},  # analyzed track
        {"id": 2, "duration_s": 200},  # new track
        {"id": 3, "duration_s": 200},  # failed before (DRM)
        {"id": 4, "duration_s": 3600},  # new set
    ]
    (tmp_path / "likes.json").write_text(json.dumps(likes))
    (tmp_path / "tracks").mkdir()
    (tmp_path / "tracks" / "1.json").write_text("{}")
    (tmp_path / "tracks" / "3.error").write_text("DRM")
    monkeypatch.setattr(analyze, "LIKES", tmp_path / "likes.json")
    monkeypatch.setattr(analyze, "ANALYSIS", tmp_path)
    monkeypatch.setattr(analyze, "KIND_OVERRIDES", tmp_path / "none.json")
    assert [(t["id"], t["pending"]) for t in analyze.waiting("tracks")] == [(2, "queued"), (3, "failed")]
    assert [t["id"] for t in analyze.waiting("sets")] == [4]
