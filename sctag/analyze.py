"""Analyze likes: stream a few windows of audio, compute tags + embeddings, save them per like.

Likes are split into two kinds that are analyzed separately:
  tracks  regular tracks (< 15 min): 3 windows at 25/50/75 %
  sets    DJ sets, podcasts, radio shows (>= 15 min): one window every 4 min, kept as a timeline,
          plus a BPM range and a rough energy label (energetic / calm / mixed).
Results go to data/analysis/<kind>/<id>.json (tags) and .npz (embeddings). Resumable: analyzed likes are skipped.
"""

import functools
import itertools
import json
import logging
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .audio import decode, fetch_cover, stream_url
from .config import (
    ANALYSIS,
    ENERGY_COOL,
    ENERGY_HOT,
    ENERGY_MAJORITY,
    KIND_OVERRIDES,
    LIKES,
    SET_MAX_WINDOWS,
    SET_MIN_S,
    SET_MIN_WINDOWS,
    SET_STEP_S,
    SR,
    TRACK_WINDOWS,
    WIN_S,
)
from .models import BINARY_HEADS, HEADS, REGRESSION_HEADS

log = logging.getLogger(__name__)


def load_likes():
    if not LIKES.exists():
        sys.exit("no likes yet: run  sctag fetch <username>")
    return json.loads(LIKES.read_text())


def load_overrides():
    return json.loads(KIND_OVERRIDES.read_text()) if KIND_OVERRIDES.exists() else {}


def kind_of(track, overrides):
    return overrides.get(str(track["id"])) or ("sets" if track["duration_s"] >= SET_MIN_S else "tracks")


def window_starts(duration, kind):
    """Start times (s) of the analyzed windows."""
    if duration <= WIN_S + 10:
        return [0.0]
    if kind == "sets":
        n = int(np.clip(duration // SET_STEP_S, SET_MIN_WINDOWS, SET_MAX_WINDOWS))
        fracs = (np.arange(n) + 0.5) / n  # evenly spread over the whole set
    else:
        fracs = TRACK_WINDOWS
    return [round(min(f * duration, duration - WIN_S), 1) for f in fracs]


def fetch(track, kind):
    """Network part, run in worker threads: ([(start, mono audio)], cover image or None)."""
    try:
        cover = fetch_cover(track.get("artwork_url"))
    except Exception:
        cover = None  # a missing cover never fails the track
    url, duration = stream_url(track["url"])
    windows = [
        (s, a) for s in window_starts(duration or track["duration_s"], kind) if len(a := decode(url, s)) > SR * 5
    ]
    return windows, cover


def prefetch(pool, fn, items, ahead):
    """Like pool.map, but only `ahead` jobs in flight: decoded audio is ~5 MB per window."""
    it = iter(items)
    pending = deque((t, pool.submit(fn, t)) for t in itertools.islice(it, ahead))
    while pending:
        t, fut = pending.popleft()
        if (nxt := next(it, None)) is not None:
            pending.append((nxt, pool.submit(fn, nxt)))
        yield t, fut


def energy_summary(scores):
    """Rough set-level energy: energetic / calm / mixed, plus the arc for mixed sets."""
    s = np.asarray(scores)
    energetic, calm = np.mean(s > ENERGY_HOT), np.mean(s < ENERGY_COOL)
    label = "energetic" if energetic >= ENERGY_MAJORITY else "calm" if calm >= ENERGY_MAJORITY else "mixed"
    arc = None
    if label == "mixed" and len(s) >= 6:
        third = len(s) // 3
        delta = s[-third:].mean() - s[:third].mean()
        arc = "builds up" if delta > 0.1 else "winds down" if delta < -0.1 else None
    return {
        "energy": label,
        "energy_arc": arc,
        "energetic_share": round(float(energetic), 2),
        "calm_share": round(float(calm), 2),
        "energy_score": round(float(s.mean()), 3),
    }


def summarize(labels, track, kind, starts, windows, cover_emb):
    """Aggregate per-window results into one record + embedding arrays. `labels`: class names per head."""
    preds = {k: np.mean([w["preds"][k] for w in windows], axis=0) for k in HEADS}
    genre = sorted(zip(labels["genre"], preds["genre"], strict=True), key=lambda x: -x[1])[:5]
    moods = sorted(zip(labels["moodtheme"], preds["moodtheme"], strict=True), key=lambda x: -x[1])[:6]
    bpms = [w["bpm"] for w in windows]
    rec = {
        **track,
        "kind": kind,
        "n_windows": len(windows),
        "discogs_genres": [(g, round(float(p), 3)) for g, p in genre],
        "moodtheme": [(g, round(float(p), 3)) for g, p in moods],
        # binary heads: probability of the positive class, looked up by name (class order varies per head)
        **{k: round(float(preds[k][labels[k].index(k)]), 3) for k in BINARY_HEADS},
        **{k: round(float(preds[k][0]), 3) for k in REGRESSION_HEADS},
        "bpm": round(float(np.median(bpms)), 1),
        "bpm_conf": round(float(np.median([w["bpm_conf"] for w in windows])), 2),
    }
    if kind == "sets":
        tops = [labels["genre"][int(np.argmax(w["preds"]["genre"]))] for w in windows]
        rec.update(energy_summary([w["energy"] for w in windows]))
        rec["bpm_range"] = [round(float(b)) for b in np.percentile(bpms, [10, 90])]
        rec["genre_mix"] = [(g, round(c / len(tops), 2)) for g, c in Counter(tops).most_common(4)]
        rec["timeline"] = [
            {"t": s, "energy": round(w["energy"], 3), "bpm": round(w["bpm"], 1), "genre": g}
            for s, w, g in zip(starts, windows, tops, strict=True)
        ]
    else:
        rec["bpm_windows"] = [round(b, 1) for b in bpms]
    arrays = {
        "effnet": np.mean([w["effnet"] for w in windows], axis=0),
        "clap": np.mean([w["clap"] for w in windows], axis=0),
        "clap_windows": np.stack([w["clap"] for w in windows]),
        "genre_probs": preds["genre"],
        "moodtheme_probs": preds["moodtheme"],
    }
    if cover_emb is not None:
        arrays["cover_clip"] = cover_emb
    return rec, arrays


@functools.cache
def audio_models():
    """Loaded once per process, shared by tracks and sets."""
    from .models import AudioModels

    return AudioModels()


def _analyze_all(kind, todo, workers):
    """Analyze `todo` likes; returns [(track, error message)] for the failures."""
    out, m, failed, t0 = ANALYSIS / kind, audio_models(), [], time.time()
    with ThreadPoolExecutor(workers) as pool:
        for n, (t, fut) in enumerate(prefetch(pool, lambda t: fetch(t, kind), todo, ahead=workers * 2), 1):
            try:
                audio, cover = fut.result()
                if not audio:
                    raise RuntimeError("no audio")
                windows = [m.analyze_window(a, with_energy=kind == "sets") for _, a in audio]
                cover_emb = m.clip_image(cover) if cover else None
                rec, arrays = summarize(m.labels, t, kind, [s for s, _ in audio], windows, cover_emb)
                np.savez(out / f"{t['id']}.npz", **arrays)
                (out / f"{t['id']}.json").write_text(json.dumps(rec, ensure_ascii=False))
                (out / f"{t['id']}.error").unlink(missing_ok=True)  # earlier failed attempt
                status = f"{rec['bpm']:>5} bpm  {rec.get('energy', ''):9}  {rec['discogs_genres'][0][0]}"
            except Exception as e:
                (out / f"{t['id']}.error").write_text(repr(e))
                failed.append((t, repr(e)))
                status = f"FAILED {e!r}"[:80]
            log.info("[%d/%d %.1fs each] %-40.40s %s", n, len(todo), (time.time() - t0) / n, t["title"], status)
    return failed


def analyze(kind, sample=None, workers=4):
    """Analyze all likes of one kind that aren't analyzed yet; failures are retried once (except DRM)."""
    overrides = load_overrides()
    likes = [t for t in load_likes() if kind_of(t, overrides) == kind]
    if sample:
        likes = [likes[int(i)] for i in np.linspace(0, len(likes) - 1, sample)]
    out = ANALYSIS / kind
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in likes if not (out / f"{t['id']}.json").exists()]
    log.info("%s: %d already analyzed, %d to go", kind, len(likes) - len(todo), len(todo))
    if not todo:
        return
    failed = _analyze_all(kind, todo, workers)
    if retry := [t for t, err in failed if "DRM" not in err]:  # DRM-protected tracks can never be streamed
        log.info("retrying %d failures", len(retry))
        failed = [f for f in failed if "DRM" in f[1]] + _analyze_all(kind, retry, workers)
    n_drm = sum("DRM" in err for _, err in failed)
    log.info("%s: %d analyzed, %d failed (%d DRM-protected)", kind, len(todo) - len(failed), len(failed), n_drm)
