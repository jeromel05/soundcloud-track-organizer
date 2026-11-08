"""Analyze likes: stream a few windows of audio (never saved to disk), compute tags + embeddings.

Likes are split into two kinds that are analyzed separately:
  tracks  regular tracks (< 15 min): 3 windows at 25/50/75 %
  sets    DJ sets, podcasts, radio shows (>= 15 min): one window every 4 min, kept as a timeline.
          Plus a BPM range and a rough energy label (energetic / calm / mixed) to pick the right moment of the day.
Override the duration rule per like in data/kind_overrides.json: {"<track id>": "sets"}.

Usage: python -m sctag.analyze tracks --sample 50   # evenly spread sample
       python -m sctag.analyze sets                 # everything of that kind (resumable)
Results go to data/analysis/<kind>/<id>.json and .npz.
"""
import argparse
import io
import itertools
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"
OUT = DATA / "analysis"
FFMPEG = str(Path(sys.executable).parent / "ffmpeg") if (Path(sys.executable).parent / "ffmpeg").exists() \
    else shutil.which("ffmpeg") or "ffmpeg"  # conda env's ffmpeg, else the one on PATH
SR = 44100
WIN_S = 30
KINDS = ("tracks", "sets")
SET_MIN_S = 15 * 60  # sets / podcasts / radio shows are at least this long
SET_STEP_S = 4 * 60  # sets: one window every 4 minutes...
SET_MAX_WINDOWS = 30  # ...capped for very long shows
# Set energy: per window, CLAP similarity to the closest "energetic" prompt minus the closest "calm" one
ENERGETIC_PROMPTS = ["energetic, intense club music with loud driving drums",
                     "chaotic, frantic, aggressive noisy music",
                     "fast hard dance music for a rave"]
CALM_PROMPTS = ["calm, ambient, contemplative music with soft pads and no drums",
                "quiet, relaxing, meditative music",
                "slow gentle downtempo music"]
# The prompts lean energetic (median window +0.23), so the cutoffs are asymmetric. Calibrated on the full library
# against Essentia's "relaxed" head (r = -0.94). Only used when building the report: change them freely.
ENERGY_HOT = 0.15  # window counts as energetic above this...
ENERGY_COOL = 0.03  # ...and as calm below this
ENERGY_MAJORITY = 2 / 3  # share of windows needed to call a whole set energetic or calm
# note: the HF port of laion/larger_clap_music has a broken text tower (all prompts embed identically)
CLAP_MODEL = "laion/larger_clap_music_and_speech"

# Essentia classifier heads on top of the Discogs-EffNet embedding
HEADS = {
    "genre": ("genre_discogs400-discogs-effnet-1", "PartitionedCall:0", "serving_default_model_Placeholder"),
    "moodtheme": ("mtg_jamendo_moodtheme-discogs-effnet-1", "model/Sigmoid", "model/Placeholder"),
    "aggressive": ("mood_aggressive-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "relaxed": ("mood_relaxed-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "happy": ("mood_happy-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "sad": ("mood_sad-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "party": ("mood_party-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "danceable": ("danceability-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "instrumental": ("voice_instrumental-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "bright": ("nsynth_bright_dark-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "tonal": ("tonal_atonal-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "approachability": ("approachability_regression-discogs-effnet-1", "model/Identity", "model/Placeholder"),
    "engagement": ("engagement_regression-discogs-effnet-1", "model/Identity", "model/Placeholder"),
}

def load_overrides():
    f = DATA / "kind_overrides.json"
    return json.loads(f.read_text()) if f.exists() else {}


def kind_of(track, overrides):
    return overrides.get(str(track["id"])) or ("sets" if track["duration_s"] >= SET_MIN_S else "tracks")


def window_starts(duration, kind):
    if duration <= WIN_S + 10:
        return [0.0]
    if kind == "sets":
        n = int(np.clip(duration // SET_STEP_S, 4, SET_MAX_WINDOWS))
        fracs = (np.arange(n) + 0.5) / n  # evenly spread over the whole set
    else:
        fracs = (0.25, 0.5, 0.75)
    return [round(min(f * duration, duration - WIN_S), 1) for f in fracs]


def stream_url(track_url):
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "bestaudio"}) as y:
        info = y.extract_info(track_url, download=False)
    return info["url"], info.get("duration") or 0


def decode(url, start):
    cmd = [FFMPEG, "-loglevel", "error", "-ss", str(start), "-t", str(WIN_S), "-i", url,
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout, dtype=np.float32)


def fetch_audio(track, kind):
    """Network part, run in worker threads: returns list of (start, mono 44.1k audio)."""
    url, dur = stream_url(track["url"])
    dur = dur or track["duration_s"]
    return [(s, a) for s in window_starts(dur, kind) if len(a := decode(url, s)) > SR * 5]


def fetch_cover(track):
    url = (track.get("artwork_url") or "").replace("-large.", "-t500x500.")
    if not url:
        return None
    from PIL import Image
    with urllib.request.urlopen(url, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def fetch(track, kind):
    try:
        cover = fetch_cover(track)
    except Exception:
        cover = None
    return fetch_audio(track, kind), cover


def prefetch(pool, fn, items, ahead):
    """Like pool.map, but only `ahead` jobs in flight: decoded audio is ~5 MB per window."""
    it = iter(items)
    pending = deque((t, pool.submit(fn, t)) for t in itertools.islice(it, ahead))
    while pending:
        t, fut = pending.popleft()
        if (nxt := next(it, None)) is not None:
            pending.append((nxt, pool.submit(fn, nxt)))
        yield t, fut


class Encoders:
    """CLAP (audio <-> text) and CLIP (cover <-> text): all the report and the search server need."""

    def __init__(self):
        import torch
        from transformers import ClapModel, ClapProcessor, CLIPModel, CLIPProcessor

        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.clap = ClapModel.from_pretrained(CLAP_MODEL).to(self.dev).eval()
        self.clap_proc = ClapProcessor.from_pretrained(CLAP_MODEL)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.dev).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def _emb(out):
        # transformers v5 returns an output object, v4 a tensor
        t = out if hasattr(out, "shape") else out.pooler_output
        return (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def clap_audio(self, audio48):
        chunks = [audio48[i:i + 480000] for i in range(0, len(audio48) - 48000 * 3, 480000)]  # 10 s chunks
        inp = self.clap_proc.feature_extractor(chunks, sampling_rate=48000, return_tensors="pt")
        with self.torch.no_grad():
            return self._emb(self.clap.get_audio_features(**{k: v.to(self.dev) for k, v in inp.items()}))

    def clap_text(self, texts):
        inp = self.clap_proc.tokenizer(texts, padding=True, return_tensors="pt")
        with self.torch.no_grad():
            return self._emb(self.clap.get_text_features(**{k: v.to(self.dev) for k, v in inp.items()}))

    def clip_image(self, img):
        inp = self.clip_proc(images=img, return_tensors="pt")
        with self.torch.no_grad():
            return self._emb(self.clip.get_image_features(pixel_values=inp["pixel_values"].to(self.dev)))[0]

    def clip_text(self, texts):
        inp = self.clip_proc.tokenizer(texts, padding=True, return_tensors="pt")
        with self.torch.no_grad():
            return self._emb(self.clip.get_text_features(**{k: v.to(self.dev) for k, v in inp.items()}))


class Models(Encoders):
    """Everything used to analyze audio: Essentia (EffNet + heads, rhythm) on top of the encoders."""

    def __init__(self):
        if not (MODELS / "discogs-effnet-bs64-1.pb").exists():
            sys.exit("Essentia models missing: run  python -m sctag.download_models")
        import essentia
        essentia.log.warningActive = False
        import essentia.standard as es

        super().__init__()
        self.es = es
        self.effnet = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODELS / "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
        self.heads = {k: es.TensorflowPredict2D(graphFilename=str(MODELS / f"{f}.pb"), output=o, input=i)
                      for k, (f, o, i) in HEADS.items()}
        self.labels = {k: json.load(open(MODELS / f"{f}.json"))["classes"] for k, (f, _, _) in HEADS.items()}
        self.rhythm = es.RhythmExtractor2013(method="multifeature")
        self.resample16 = es.Resample(inputSampleRate=SR, outputSampleRate=16000)
        self.resample48 = es.Resample(inputSampleRate=SR, outputSampleRate=48000)
        self.energetic_text = self.clap_text(ENERGETIC_PROMPTS)
        self.calm_text = self.clap_text(CALM_PROMPTS)

    def energy(self, clap):
        c = clap / np.linalg.norm(clap)
        return float((self.energetic_text @ c).max() - (self.calm_text @ c).max())

    def analyze_window(self, audio, kind):
        a16 = self.resample16(audio)
        emb = self.effnet(a16)  # (patches, 1280)
        preds = {k: np.asarray(h(emb)).mean(axis=0) for k, h in self.heads.items()}
        clap = self.clap_audio(self.resample48(audio)).mean(axis=0)
        bpm, _, conf, _, _ = self.rhythm(audio)
        w = {"effnet": emb.mean(axis=0), "preds": preds, "clap": clap, "bpm": float(bpm), "bpm_conf": float(conf)}
        if kind == "sets":
            w["energy"] = self.energy(clap)
        return w


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
    return {"energy": label, "energy_arc": arc, "energetic_share": round(float(energetic), 2),
            "calm_share": round(float(calm), 2), "energy_score": round(float(s.mean()), 3)}


def summarize(m, track, kind, starts, windows, cover_emb):
    """Aggregate per-window results into one record + embedding arrays."""
    preds = {k: np.mean([w["preds"][k] for w in windows], axis=0) for k in HEADS}
    genre = sorted(zip(m.labels["genre"], preds["genre"]), key=lambda x: -x[1])[:5]
    moods = sorted(zip(m.labels["moodtheme"], preds["moodtheme"]), key=lambda x: -x[1])[:6]
    rec = {
        **track,
        "kind": kind,
        "n_windows": len(windows),
        "discogs_genres": [(g, round(float(p), 3)) for g, p in genre],
        "moodtheme": [(g, round(float(p), 3)) for g, p in moods],
        # binary heads: probability of the positive class, looked up by name (head key == class name; class order varies per head)
        **{k: round(float(preds[k][m.labels[k].index(k)]), 3)
           for k in HEADS if k not in ("genre", "moodtheme", "approachability", "engagement")},
        **{k: round(float(preds[k][0]), 3) for k in ("approachability", "engagement")},
    }
    bpms = [w["bpm"] for w in windows]
    rec["bpm"] = round(float(np.median(bpms)), 1)
    rec["bpm_conf"] = round(float(np.median([w["bpm_conf"] for w in windows])), 2)
    if kind == "sets":
        tops = [m.labels["genre"][int(np.argmax(w["preds"]["genre"]))] for w in windows]
        rec.update(energy_summary([w["energy"] for w in windows]))
        rec["bpm_range"] = [round(float(b)) for b in np.percentile(bpms, [10, 90])]
        rec["genre_mix"] = [(g, round(c / len(tops), 2)) for g, c in Counter(tops).most_common(4)]
        rec["timeline"] = [{"t": s, "energy": round(w["energy"], 3), "bpm": round(w["bpm"], 1), "genre": g}
                           for s, w, g in zip(starts, windows, tops)]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("--sample", type=int, help="evenly spaced sample of N likes of this kind")
    ap.add_argument("--workers", type=int, default=4, help="parallel audio downloads")
    args = ap.parse_args()

    overrides = load_overrides()
    likes = [t for t in json.load(open(DATA / "likes.json")) if kind_of(t, overrides) == args.kind]
    if args.sample:
        likes = [likes[int(i)] for i in np.linspace(0, len(likes) - 1, args.sample)]
    out = OUT / args.kind
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in likes if not (out / f"{t['id']}.json").exists()]
    print(f"{args.kind}: {len(likes) - len(todo)} cached, {len(todo)} to analyze", file=sys.stderr)

    m = Models()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = prefetch(pool, lambda t: fetch(t, args.kind), todo, ahead=args.workers * 2)
        for n, (t, fut) in enumerate(jobs, 1):
            try:
                audio, cover = fut.result()
                if not audio:
                    raise RuntimeError("no audio")
                windows = [m.analyze_window(a, args.kind) for _, a in audio]
                cover_emb = m.clip_image(cover) if cover else None
                rec, arrays = summarize(m, t, args.kind, [s for s, _ in audio], windows, cover_emb)
                np.savez(out / f"{t['id']}.npz", **arrays)
                (out / f"{t['id']}.json").write_text(json.dumps(rec, ensure_ascii=False))
                (out / f"{t['id']}.error").unlink(missing_ok=True)  # earlier failed attempt
                status = f"{rec['bpm']:>5} bpm  {rec.get('energy', ''):9}  {rec['discogs_genres'][0][0]}"
            except Exception as e:
                (out / f"{t['id']}.error").write_text(repr(e))
                status = f"FAILED {e!r}"[:80]
            el = time.time() - t0
            print(f"[{n}/{len(todo)} {el / n:.1f}s/track] {t['title'][:40]:40} {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
