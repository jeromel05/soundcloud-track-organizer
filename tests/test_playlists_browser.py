"""The browser script (sctag/playlists_browser.js), run with Node against a fake SoundCloud API."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sctag.playlists import browser_script

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="needs Node.js")
FAKE = Path(__file__).parent / "js" / "fake_soundcloud.js"
PLAN = {"sctag · a": [1, 2, 3], "sctag · b": [4]}


def run(tmp_path, plan=PLAN, **scenario):
    script, scen = tmp_path / "script.js", tmp_path / "scenario.json"
    script.write_text(browser_script(plan, "CID", pauses={"read": [0, 0], "write": [0, 0]}))
    scen.write_text(json.dumps(scenario))
    out = subprocess.run(["node", FAKE, script, scen], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


def writes(result):
    return [(c["method"], c["path"]) for c in result["calls"] if c["method"] != "GET"]


def test_creates_private_playlists_and_remembers_them(tmp_path):
    other = {"5": {"id": 5, "user_id": 8, "title": "sctag · a", "tracks": []}}  # someone else's, same title
    r = run(tmp_path, playlists=other)
    assert writes(r) == [("POST", "/playlists"), ("POST", "/playlists")]
    created = [c["body"]["playlist"] for c in r["calls"] if c["method"] == "POST"]
    assert [(p["title"], p["sharing"], p["tracks"]) for p in created] == [
        ("sctag · a", "private", [1, 2, 3]),
        ("sctag · b", "private", [4]),
    ]
    assert r["store"] == {"sctag · a": 100, "sctag · b": 101}
    assert all(c["auth"] == "OAuth 2-123-456-abc" and c["client_id"] == "CID" for c in r["calls"])
    assert r["playlists"]["5"]["tracks"] == []


def test_rerun_updates_only_what_changed(tmp_path):
    mine = {
        "9": {"id": 9, "user_id": 7, "title": "sctag · a", "tracks": [{"id": 1}]},
        "10": {"id": 10, "user_id": 7, "title": "sctag · b", "tracks": [{"id": 4}]},
        "11": {"id": 11, "user_id": 7, "title": "sctag · old", "tracks": [{"id": 9}]},
        "12": {"id": 12, "user_id": 7, "title": "summer", "tracks": [{"id": 9}]},
    }
    r = run(tmp_path, playlists=mine)
    assert writes(r) == [("PUT", "/playlists/9")]
    assert [t["id"] for t in r["playlists"]["9"]["tracks"]] == [1, 2, 3]
    assert any("left untouched" in line and "sctag · old" in line for line in r["logs"])


def test_remembered_ids_are_checked_before_editing(tmp_path):
    foreign = {
        "5": {"id": 5, "user_id": 8, "title": "sctag · a", "tracks": []},
        "6": {"id": 6, "user_id": 7, "title": "renamed", "tracks": []},
    }
    r = run(tmp_path, playlists=foreign, store={"sctag · a": 5, "sctag · b": 6}, list_fails=True)
    assert writes(r) == []
    assert sum("not touching it" in line for line in r["logs"]) == 2


def test_stops_at_the_bot_check(tmp_path):
    r = run(tmp_path, block=True)
    assert writes(r) == [("POST", "/playlists")]  # one attempt, then stop
    assert any("bot protection" in line for line in r["logs"]) and r["store"] == {}


def test_stops_when_tracks_are_dropped(tmp_path):
    r = run(tmp_path, keep=1)  # SoundCloud keeps 1 of 3: wrong request format
    assert writes(r) == [("POST", "/playlists")]
    assert any("kept only 1 of 3" in line for line in r["logs"])


def test_nothing_happens_without_confirmation(tmp_path):
    assert writes(run(tmp_path, confirm=False)) == []


def test_never_writes_outside_sctag_titles(tmp_path):
    r = run(tmp_path, plan={"summer": [1]})
    assert writes(r) == [] and any("refusing to create summer" in line for line in r["logs"])
