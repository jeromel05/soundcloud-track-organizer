"""SoundCloud playlists made from your tags, to listen by genre / mood / energy in the SoundCloud app.

SoundCloud's bot protection blocks scripts from creating playlists, so this part runs in your own browser:
`sctag playlists --browser` writes data/playlists_browser.js, which you paste into the developer console of
soundcloud.com while logged in. That script is the only part of sctag that writes to your account. It can only:
  - create private playlists titled "sctag · ..."
  - replace the track list of a playlist titled "sctag · ..." that belongs to you
It never deletes playlists, never changes their title or sharing, and never touches likes, follows, reposts or any
other playlist. Every write goes through one function in the script, which refuses anything else.

You pick the playlists: data/playlists.json lists every candidate with "make": false, one per line. Set the ones you
want to true (and rename them if you like). You can also add your own, combining filters:
  {"make": true, "title": "dark fast funk", "kind": "tracks", "genre": "baile funk", "mood": "dark", "bpm": [130, 200]}
Filters: genre, mood, energy (sets: energetic / mixed / calm), bpm [low, high]; a like must match all of them.
"guesses": false leaves out the genres guessed from audio (only your labels and the title / tags count).
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

from .analyze import kind_of, load_likes, load_overrides
from .classify import load_labels
from .config import HOME, KINDS, PLAYLISTS, PLAYLISTS_SCRIPT, library_path
from .soundcloud import scrape_client_id
from .text_tags import text_tags

PREFIX = "sctag · "  # every playlist this tool makes or edits has this title prefix
MAX_TRACKS = 500  # SoundCloud's playlist length limit: longer playlists are split into "(1)", "(2)", ...
MIN_CANDIDATE = 5  # a genre / mood is offered as a playlist once this many likes have it
FILTERS = ("genre", "mood", "energy", "bpm")
SCRIPT_TEMPLATE = Path(__file__).parent / "playlists_browser.js"
# Random pauses (s) before each request in the browser, like someone clicking around rather than a burst of calls
PAUSES = {"read": (2, 5), "write": (10, 25)}


def short(path):
    return path.relative_to(HOME) if path.is_relative_to(HOME) else path


# --- what goes in each playlist -------------------------------------------------------------------------------------


def tagged(r, labels):
    """The tags of one like that playlists filter on, with your labels applied."""
    mine = labels.get(str(r["id"]), {})
    if mine.get("genres") is not None:
        sure, guessed = set(mine["genres"]), set()
    else:
        sure = {g for g, _ in r.get("text_genres") or ()}
        guessed = {g for g, _ in r.get("guess_genres") or ()} - sure
    return {
        "id": r["id"],
        "title": r.get("title"),
        "sure": sure,
        "guessed": guessed,
        "moods": {m for m, _ in r.get("clap_moods") or ()},
        "energy": mine.get("energy") or r.get("energy"),
        "bpm": r.get("bpm"),
    }


def load_library():
    """{kind: [tagged like]} for everything still liked, newest like first.

    Likes that couldn't be analyzed (DRM-protected) are included with their title / tags genres only.
    """
    analyzed = {}
    for kind in KINDS:
        if not (path := library_path(kind)).exists():
            sys.exit("no tags yet: run  sctag report")
        analyzed.update({r["id"]: (kind, r) for r in json.loads(path.read_text())})
    labels, overrides = load_labels(), load_overrides()
    library = {kind: [] for kind in KINDS}
    for t in load_likes():
        if t["id"] in analyzed:
            kind, r = analyzed[t["id"]]
        else:
            kind = kind_of(t, overrides)
            r = {**t, "text_genres": text_tags(t, kind)}
        library[kind].append(tagged(r, labels))
    return library


def matches(like, entry):
    if (g := entry.get("genre")) and not (g in like["sure"] or (entry.get("guesses", True) and g in like["guessed"])):
        return False
    if (m := entry.get("mood")) and m not in like["moods"]:
        return False
    if (e := entry.get("energy")) and e != like["energy"]:
        return False
    return not (b := entry.get("bpm")) or (like["bpm"] is not None and b[0] <= like["bpm"] <= b[1])


def members(entry, library):
    return [like for like in library[entry["kind"]] if matches(like, entry)]


def sc_titles(title, n_tracks):
    """SoundCloud title(s) of one picked playlist: split in parts when it's too long."""
    parts = max(1, math.ceil(n_tracks / MAX_TRACKS))
    return [PREFIX + title] if parts == 1 else [f"{PREFIX}{title} ({i})" for i in range(1, parts + 1)]


# --- data/playlists.json: the candidates, and your picks ------------------------------------------------------------


def key(entry):
    return entry["kind"], tuple((f, json.dumps(entry[f])) for f in FILTERS if f in entry)


def candidates(library):
    """Every genre / mood (and set energy) with enough likes, most common first."""
    out = []
    for kind, likes in library.items():
        pre = "sets · " if kind == "sets" else ""
        energy = Counter(like["energy"] for like in likes if like["energy"])
        out += [
            {"title": f"{pre}{e}", "kind": kind, "energy": e}
            for e in ("energetic", "mixed", "calm")
            if energy[e] >= MIN_CANDIDATE
        ]
        genres = Counter(g for like in likes for g in like["sure"] | like["guessed"])
        out += [{"title": pre + g, "kind": kind, "genre": g} for g, c in genres.most_common() if c >= MIN_CANDIDATE]
        moods = Counter(m for like in likes for m in like["moods"])
        out += [
            {"title": f"{pre}mood · {m}", "kind": kind, "mood": m} for m, c in moods.most_common() if c >= MIN_CANDIDATE
        ]
    return [{"make": False, **c} for c in out]


def save_picks(entries):
    lines = ",\n".join(" " + json.dumps(e, ensure_ascii=False) for e in entries)  # one per line: easy to edit
    PLAYLISTS.write_text(f"[\n{lines}\n]\n")


def load_picks(library):
    """data/playlists.json, with new candidates appended (not picked). Returns (entries, number added)."""
    entries = json.loads(PLAYLISTS.read_text()) if PLAYLISTS.exists() else []
    known = {key(e) for e in entries}
    new = [c for c in candidates(library) if key(c) not in known]
    if new:
        save_picks(entries + new)
    for e in entries:
        if e.get("kind") not in KINDS or not isinstance(e.get("title"), str) or not any(f in e for f in FILTERS):
            sys.exit(f"{PLAYLISTS}: each playlist needs a title, a kind (tracks / sets) and a filter ({e})")
    titles = Counter(e["title"] for e in entries if e.get("make"))
    if dup := [t for t, c in titles.items() if c > 1]:
        sys.exit(f"{PLAYLISTS}: two picked playlists have the same title: {dup}")
    return entries + new, len(new)


def plan(entries, library):
    """{SoundCloud title: [track ids]} of the picked playlists."""
    out = {}
    for e in entries:
        if e.get("make"):
            ids = [like["id"] for like in members(e, library)]
            titles = sc_titles(e["title"], len(ids))
            size = math.ceil(len(ids) / len(titles))  # equal parts: 514 likes = 257 + 257, not 500 + 14
            out |= {title: ids[i * size : (i + 1) * size] for i, title in enumerate(titles)}
    return out


# --- the report ------------------------------------------------------------------------------------------------------


def report(entries, library, n_new):
    picks_file = short(PLAYLISTS)
    if n_new:
        print(f"{n_new} new candidates added to {picks_file} (not picked)\n")
    sizes = ", ".join(f"{len(library[k])} {k}" for k in KINDS)
    print(f"Candidate playlists, from {sizes} still liked. Pick them in {picks_file}.\n")
    for kind in KINDS:
        print(f"{kind.upper():34} {'likes':>6} {'guessed':>8}  {'playlists':>9}  picked")
        for e in (e for e in entries if e["kind"] == kind):
            ms = members(e, library)
            guessed = sum(e["genre"] not in like["sure"] for like in ms) if "genre" in e else None
            n = len(sc_titles(e["title"], len(ms))) if ms else 0
            print(
                f"  {e['title'][:32]:32} {len(ms):6} {'' if guessed is None else guessed:>8}  {n:9}"
                f"  {'yes' if e.get('make') else ''}"
            )
        print()
    print("guessed: likes that are in it only because the genre model guessed the genre from audio")
    print('("guesses": false leaves them out). Moods are always guessed from audio, relative to your library.\n')

    picked = plan(entries, library)
    if not picked:
        print(f'Nothing picked yet: set "make": true on the playlists you want in {picks_file},')
        print("then rerun  sctag playlists  to check, and  sctag playlists --browser  to make them.")
        return picked
    n_tracks = len({i for v in picked.values() for i in v})
    print(f"Picked: {len(picked)} SoundCloud playlists, {n_tracks} different likes")
    for title, v in picked.items():
        print(f"  {title}  ({len(v)} likes)")
    return picked


# --- the browser script ----------------------------------------------------------------------------------------------


def browser_script(picked, client_id, pauses=PAUSES):
    """The JavaScript that makes the picked playlists, run from the console of soundcloud.com."""
    data = {"client_id": client_id, "prefix": PREFIX, "pauses": pauses, "playlists": picked}
    return SCRIPT_TEMPLATE.read_text().replace("/*PLAN*/null", json.dumps(data, ensure_ascii=False))


def write_script(picked):
    PLAYLISTS_SCRIPT.write_text(browser_script(picked, scrape_client_id()))
    minutes = round(len(picked) * sum(PAUSES["write"]) / 2 / 60) or 1
    print(f"""
Wrote {short(PLAYLISTS_SCRIPT)}. To make the playlists (takes about {minutes} min, it pauses between requests):
  1. In Firefox, open soundcloud.com, logged in. Keep this tab open until it's done.
  2. Open the developer tools (Cmd+Option+I) and click the Console tab.
  3. Copy the whole script:  pbcopy < {short(PLAYLISTS_SCRIPT)}
  4. Click in the console, paste (Cmd+V) and press Enter. The first time, Firefox asks you to type
     allow pasting  and press Enter, then paste again.
  5. It lists what it will do and asks you to confirm. Progress shows in the console.
Rerun  sctag playlists --browser  and paste the new script to update the playlists later.""")


def playlists(browser=False):
    library = load_library()
    entries, n_new = load_picks(library)
    picked = report(entries, library, n_new)
    if browser and picked:
        write_script(picked)
