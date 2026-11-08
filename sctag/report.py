"""Build static HTML reports from data/analysis/ to eyeball the tags, one per kind.

Tracks and sets are scored, compared and searched only against their own kind. Genres come from the title /
tags (text_tags), your labels (data/labels.json) and the genre model trained on both (classify.py).
Usage: python -m sctag.report          -> data/report_tracks.html, data/report_sets.html
       python -m sctag.report sets     -> just one
Open the files directly, or run `python -m sctag.serve` for live search and labelling.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .analyze import DATA, ENERGY_COOL, ENERGY_HOT, ENERGY_MAJORITY, KINDS, OUT, energy_summary
from .classify import MIN_POSITIVES, GenreModel, load_labels, unit
from .text_tags import GENRE_PATTERNS, text_tags

# Your own vocabulary, scored zero-shot with CLAP text prompts.
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


def top(names, scores, k):
    idx = np.argsort(-scores)[:k]
    return [(names[i], round(float(scores[i]), 2)) for i in idx]


def load(kind):
    """Analyzed likes of this kind that are still liked, in like order, with fresh metadata from likes.json."""
    src, recs, arr = OUT / kind, [], []
    for t in json.load(open(DATA / "likes.json")):
        if (f := src / f"{t['id']}.json").exists():
            recs.append({**json.loads(f.read_text()), **t})
            arr.append(np.load(src / f"{t['id']}.npz"))
    return recs, arr


def build(enc, kind, recs, arr, gm, labels):
    if not recs:
        print(f"no {kind} analyzed yet")
        return
    clap = unit(np.stack([a["clap"] for a in arr]))
    has_cover = [("cover_clip" in a) for a in arr]
    cover = np.stack([a["cover_clip"] if "cover_clip" in a else np.zeros(512) for a in arr])

    mood_names = list(MOODS)
    mood_text = enc.clap_text(list(MOODS.values()))
    # moods: relative to this kind (z-score per tag), so "dark" means darker than your usual track / set
    mood_mu, mood_sd = (clap @ mood_text.T).mean(0), (clap @ mood_text.T).std(0) + 1e-8
    mood_z = (clap @ mood_text.T - mood_mu) / mood_sd
    learned = np.isin(gm.genres, gm.learned())
    if kind == "tracks":
        scores = gm.tracks(np.stack([a["clap"] for a in arr]), np.stack([a["effnet"] for a in arr]))

    for i, r in enumerate(recs):
        mine = labels.get(str(r["id"]), {})
        r["my_genres"], r["my_energy"] = mine.get("genres"), mine.get("energy")  # None = not labelled by you
        r["text_genres"] = text_tags(r, kind)  # confident tags from title / genre field / tags
        r["clap_moods"] = [t for t in top(mood_names, mood_z[i], 3) if t[1] > 0.5]
        sims = clap @ clap[i]
        sims[i] = -1
        r["similar"] = [int(j) for j in np.argsort(-sims)[:5]]
        r["idx"] = i
        text = {g for g, _ in r["text_genres"]}
        if kind == "sets":
            r.update(energy_summary([step["energy"] for step in r["timeline"]]))
            w = unit(arr[i]["clap_windows"])
            wz, wg = (w @ mood_text.T - mood_mu) / mood_sd, gm.windows(arr[i]["clap_windows"])
            for k, step in enumerate(r["timeline"]):  # tag each window too, so you can see how the set moves
                step["mood"] = mood_names[int(np.argmax(wz[k]))]
                step["guess"] = gm.genres[int(np.argmax(wg[k]))] if wg.shape[1] and wg[k].max() >= 1 else None
            mix = Counter(step["guess"] for step in r["timeline"] if step["guess"])
            r["guess_genres"] = [(g, round(c / len(r["timeline"]), 2)) for g, c in mix.most_common(3)
                                 if c / len(r["timeline"]) >= 0.2 and g not in text]
            # ask first about sets closest to an energy boundary
            r["label_priority"] = None if r["my_energy"] else round(min(
                abs(r["energetic_share"] - ENERGY_MAJORITY), abs(r["calm_share"] - ENERGY_MAJORITY)), 3)
        else:
            s = scores[i]
            r["guess_genres"] = [(gm.genres[k], round(float(s[k]), 2)) for k in np.argsort(-s)[:3]
                                 if s[k] >= 1 and gm.genres[k] not in text]
            # ask first about untagged tracks closest to a genre cutoff (most informative for the classifier)
            unsure = np.abs(np.log(np.maximum(s[learned], 1e-6))).min() if learned.any() else 0
            r["label_priority"] = None if r["my_genres"] is not None or text else round(float(unsure), 3)

    def vibe_hits(qe):
        # tracks: whole-track embedding; sets: best-matching window, so one fitting passage is enough
        if kind == "sets":
            sims = np.array([np.max(unit(a["clap_windows"]) @ qe) for a in arr])
        else:
            sims = clap @ qe
        return [(int(j), round(float(sims[j]), 3)) for j in np.argsort(-sims)[:6]]

    vibe = {q: vibe_hits(qe) for q, qe in zip(VIBE_QUERIES, enc.clap_text(VIBE_QUERIES))}
    cs = cover @ enc.clip_text(COVER_QUERIES).T
    cs[~np.array(has_cover)] = -1
    covers = {q: [int(j) for j in np.argsort(-cs[:, k])[:6]] for k, q in enumerate(COVER_QUERIES)}

    vocab = list(GENRE_PATTERNS) + ["rap edit"]
    vocab += sorted({g for e in labels.values() for g in e.get("genres", ())} - set(vocab))  # your own tags
    payload = {"kind": kind, "tracks": recs, "vibe": vibe, "covers": covers, "moods": mood_names, "genres": vocab,
               "energy_cut": [ENERGY_HOT, ENERGY_COOL], "learned": gm.learned(),
               "min_positives": MIN_POSITIVES}
    html = (Path(__file__).parent / "report_template.html").read_text()
    out = DATA / f"report_{kind}.html"
    out.write_text(html.replace("/*DATA*/null", json.dumps(payload, ensure_ascii=False)))
    guessed = sum(bool(r["guess_genres"]) and not r["text_genres"] for r in recs)
    print(f"wrote {out}: {len(recs)} {kind}, {guessed} untagged ones got a genre guess")


def main():
    from .analyze import Encoders

    enc, labels = Encoders(), load_labels()
    data = {kind: load(kind) for kind in KINDS}
    recs, arr = data["tracks"]  # the genre model learns from tracks, and also tags set windows
    if not recs:
        sys.exit("no tracks analyzed yet: run  python -m sctag.analyze tracks")
    gm = GenreModel(recs, np.stack([a["clap"] for a in arr]), np.stack([a["effnet"] for a in arr]), labels)
    print(f"genre model: learned {', '.join(gm.learned())}")
    for kind in sys.argv[1:] or KINDS:
        build(enc, kind, *data[kind], gm, labels)


if __name__ == "__main__":
    main()
