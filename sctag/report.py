"""Build the HTML reports (data/report_tracks.html, data/report_sets.html) from data/analysis/.

Tracks and sets are scored, compared and searched only against their own kind. Genres come from your labels,
the title / tags (text_tags) and the genre model trained on both (classify). Moods are zero-shot CLAP prompts.
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .analyze import energy_summary, load_likes, waiting
from .classify import MIN_POSITIVES, GenreModel, load_labels
from .config import ANALYSIS, ENERGY_COOL, ENERGY_HOT, ENERGY_MAJORITY, KINDS, report_path
from .models import unit
from .search import Index, best, load_arrays
from .text_tags import GENRE_PATTERNS, text_tags

log = logging.getLogger(__name__)
TEMPLATE = Path(__file__).parent / "report_template.html"

# Your own mood vocabulary: mood -> CLAP prompt
MOODS = {
    "light": "light, airy, gentle music",
    "dark": "dark, ominous, brooding music",
    "angelic": "angelic, ethereal music with heavenly vocals or choir",
    "heavy": "heavy, massive, distorted bass music",
    "noisy": "noisy, harsh, abrasive music full of distortion",
    "deconstructed": "deconstructed club music with broken, fragmented rhythms",
    "melancholic": "sad, melancholic, emotional music",
    "euphoric": "euphoric, uplifting, joyful music",
    "aggressive": "aggressive, intense, violent music",
    "hypnotic": "hypnotic, repetitive, trance-inducing music",
    "playful": "playful, bouncy, fun music",
    "sensual": "sensual, sexy, slow groove",
    "eerie": "eerie, creepy, haunted music",
    "dreamy": "dreamy, hazy, washed-out music",
    "minimal": "minimal, sparse, stripped-down music",
    "chaotic": "chaotic, frantic, hectic music",
    "icy": "cold, icy, metallic, futuristic music",
    "warm": "warm, soulful, organic music",
}
MOOD_MIN_Z = 0.5  # a mood is shown when a like is this many standard deviations above your average
# Example searches baked into the report (the served report also has a search box)
VIBE_QUERIES = [
    "dark angelic vocals over broken percussion",
    "fast distorted Brazilian funk",
    "calm ambient pads with no drums",
    "hard pounding techno rave",
    "playful bouncy club track with chopped vocal samples",
    "melancholic rap with a sad piano",
]
COVER_QUERIES = [
    "a pink blurry photo",
    "a black and white photo",
    "a glossy 3D render",
    "an anime drawing",
    "a close-up of a person's face",
    "a cover with only text or typography",
    "a colorful collage",
    "a dark minimal abstract cover",
]
SET_GENRE_MIN_SHARE = 0.2  # a set gets a genre guess when this share of its windows has it


def load(kind):
    """Analyzed likes of this kind that are still liked, in like order, with fresh metadata from likes.json."""
    src, recs, arrays = ANALYSIS / kind, [], []
    for t in load_likes():
        if (f := src / f"{t['id']}.json").exists():
            recs.append({**json.loads(f.read_text()), **t})
            arrays.append(load_arrays(src / f"{t['id']}.npz"))
    return recs, arrays


class Moods:
    """CLAP mood prompts, z-scored per mood over one kind: "dark" means darker than your usual track / set."""

    def __init__(self, enc, clap):
        self.names = list(MOODS)
        self.text = enc.clap_text(list(MOODS.values()))
        sims = clap @ self.text.T
        self.mu, self.sd = sims.mean(0), sims.std(0) + 1e-8

    def z(self, emb):
        return (unit(emb) @ self.text.T - self.mu) / self.sd

    def top(self, z, k=3):
        return [(self.names[i], round(float(z[i]), 2)) for i in np.argsort(-z)[:k] if z[i] > MOOD_MIN_Z]


def tag_track(r, scores, gm, learned):
    """Genre guesses for a track, and how useful labelling it would be."""
    text = {g for g, _ in r["text_genres"]}
    r["guess_genres"] = [
        (gm.genres[k], round(float(scores[k]), 2))
        for k in np.argsort(-scores)[:3]
        if scores[k] >= 1 and gm.genres[k] not in text
    ]
    # ask first about untagged tracks closest to a genre cutoff (the most informative for the model)
    unsure = np.abs(np.log(np.maximum(scores[learned], 1e-6))).min() if learned.any() else 0
    r["label_priority"] = None if r["my_genres"] is not None or text else round(float(unsure), 3)


def tag_set(r, clap_windows, gm, moods):
    """Energy, per-window mood and genre (so you can see how the set moves), set-level genre guesses."""
    r.update(energy_summary([step["energy"] for step in r["timeline"]]))  # with the current cutoffs
    wz, wg = moods.z(clap_windows), gm.windows(clap_windows)
    for k, step in enumerate(r["timeline"]):
        step["mood"] = moods.names[int(np.argmax(wz[k]))]
        step["guess"] = gm.genres[int(np.argmax(wg[k]))] if wg.shape[1] and wg[k].max() >= 1 else None
    text, n = {g for g, _ in r["text_genres"]}, len(r["timeline"])
    mix = Counter(step["guess"] for step in r["timeline"] if step["guess"])
    r["guess_genres"] = [
        (g, round(c / n, 2)) for g, c in mix.most_common(3) if c / n >= SET_GENRE_MIN_SHARE and g not in text
    ]
    # ask first about sets closest to an energy boundary
    margin = min(abs(r["energetic_share"] - ENERGY_MAJORITY), abs(r["calm_share"] - ENERGY_MAJORITY))
    r["label_priority"] = None if r["my_energy"] else round(margin, 3)


def annotate(kind, recs, arrays, ix, enc, gm, labels):
    """Add everything the page shows to each record."""
    moods = Moods(enc, ix.clap)
    mood_z = moods.z(ix.clap)
    similar = ix.clap @ ix.clap.T
    np.fill_diagonal(similar, -1)
    learned = np.isin(gm.genres, gm.learned())
    if kind == "tracks":
        scores = gm.tracks(np.stack([a["clap"] for a in arrays]), np.stack([a["effnet"] for a in arrays]))
    for i, r in enumerate(recs):
        mine = labels.get(str(r["id"]), {})
        r["idx"] = i
        r["my_genres"], r["my_energy"] = mine.get("genres"), mine.get("energy")  # None = not labelled by you
        r["text_genres"] = text_tags(r, kind)
        r["clap_moods"] = moods.top(mood_z[i])
        r["similar"] = [int(j) for j in np.argsort(-similar[i])[:5]]
        if kind == "tracks":
            tag_track(r, scores[i], gm, learned)
        else:
            tag_set(r, arrays[i]["clap_windows"], gm, moods)


def write(kind, recs, ix, enc, gm, labels):
    vibe = {q: best(ix.audio_scores(e), 6) for q, e in zip(VIBE_QUERIES, enc.clap_text(VIBE_QUERIES), strict=True)}
    covers = {q: best(ix.cover_scores(e), 6) for q, e in zip(COVER_QUERIES, enc.clip_text(COVER_QUERIES), strict=True)}
    vocab = list(GENRE_PATTERNS) + ["rap edit"]
    vocab += sorted({g for e in labels.values() for g in e.get("genres", ())} - set(vocab))  # your own tags
    payload = {
        "kind": kind,
        "tracks": recs,
        "vibe": vibe,
        "covers": covers,
        "moods": list(MOODS),
        "genres": vocab,
        "energy_cut": [ENERGY_HOT, ENERGY_COOL],
        "learned": gm.learned(),
        "min_positives": MIN_POSITIVES,
        "pending": waiting(kind),  # liked but not analyzed yet: shown and playable, without tags
    }
    out = report_path(kind)
    out.write_text(TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(payload, ensure_ascii=False)))
    guessed = sum(bool(r["guess_genres"]) and not r["text_genres"] for r in recs)
    log.info("wrote %s: %d %s, %d untagged ones got a genre guess", out, len(recs), kind, guessed)


def build(kinds=KINDS, enc=None):
    """Train the genre model on your tracks, then write the report of each kind."""
    if enc is None:
        from .models import Encoders

        enc = Encoders()
    labels = load_labels()
    data = {kind: load(kind) for kind in KINDS}
    recs, arrays = data["tracks"]  # the genre model learns from tracks, and also tags set windows
    if not recs:
        sys.exit("no tracks analyzed yet: run  sctag analyze tracks")
    gm = GenreModel(recs, np.stack([a["clap"] for a in arrays]), np.stack([a["effnet"] for a in arrays]), labels)
    log.info("genre model learned: %s", ", ".join(gm.learned()) or "nothing yet (label more tracks)")
    for kind in kinds:
        recs, arrays = data[kind]
        if not recs:
            log.info("no %s analyzed yet", kind)
            continue
        ix = Index(kind, [r["id"] for r in recs], arrays)
        annotate(kind, recs, arrays, ix, enc, gm, labels)
        write(kind, recs, ix, enc, gm, labels)
