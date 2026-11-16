import json

from sctag import playlists as pl


def like(i, sure=(), guessed=(), moods=(), energy=None, bpm=None):
    return {
        "id": i,
        "title": str(i),
        "sure": set(sure),
        "guessed": set(guessed),
        "moods": set(moods),
        "energy": energy,
        "bpm": bpm,
    }


def test_tagged_your_labels_replace_tags_and_guesses():
    r = {"id": 1, "text_genres": [("rap", "title")], "guess_genres": [("trap", 1.2)], "clap_moods": [("dark", 1.0)]}
    assert pl.tagged(r, {})["sure"] == {"rap"} and pl.tagged(r, {})["guessed"] == {"trap"}
    mine = pl.tagged(r, {"1": {"genres": ["drill"], "energy": "calm"}})
    assert (mine["sure"], mine["guessed"], mine["energy"]) == ({"drill"}, set(), "calm")


def test_matches_all_filters():
    funk = like(1, sure={"baile funk"}, moods={"dark"}, bpm=135)
    guess = like(2, guessed={"baile funk"}, bpm=90)
    entry = {"kind": "tracks", "genre": "baile funk"}
    assert pl.matches(funk, entry) and pl.matches(guess, entry)
    assert not pl.matches(guess, {**entry, "guesses": False})
    assert pl.matches(funk, {**entry, "mood": "dark", "bpm": [130, 200]})
    assert not pl.matches(guess, {**entry, "bpm": [130, 200]})
    assert not pl.matches(like(3), {"kind": "tracks", "bpm": [0, 300]})  # no BPM known
    assert pl.matches(like(4, energy="calm"), {"kind": "sets", "energy": "calm"})


def test_long_playlists_are_split():
    assert pl.sc_titles("funk", 500) == ["sctag · funk"]
    assert pl.sc_titles("funk", 501) == ["sctag · funk (1)", "sctag · funk (2)"]
    library = {"tracks": [like(i, sure={"funk"}) for i in range(1200)], "sets": []}
    plan = pl.plan([{"make": True, "title": "funk", "kind": "tracks", "genre": "funk"}], library)
    assert [len(v) for v in plan.values()] == [400, 400, 400]
    assert sum(plan.values(), []) == list(range(1200))  # like order kept


def test_picks_file_keeps_your_choices(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "PLAYLISTS", tmp_path / "playlists.json")
    library = {"tracks": [like(i, sure={"funk"}, moods={"dark"}) for i in range(6)], "sets": []}
    entries, n_new = pl.load_picks(library)
    assert n_new == 2 and not any(e["make"] for e in entries)
    entries[0].update(make=True, title="my funk")
    pl.save_picks(entries + [{"make": True, "title": "custom", "kind": "tracks", "genre": "funk", "mood": "dark"}])
    entries, n_new = pl.load_picks(library)
    assert n_new == 0 and [e["title"] for e in entries if e["make"]] == ["my funk", "custom"]
    assert len(json.loads((tmp_path / "playlists.json").read_text())) == 3
