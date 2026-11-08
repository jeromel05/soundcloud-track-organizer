# SoundCloud likes organizer

Tag, browse and search your SoundCloud likes by genre, mood, BPM, energy and cover art, using audio models
that run on your own computer. Made for curating DJ sets out of a big pile of likes.

- **Library**: filter by genre, mood and BPM, search by title or artist, play any track in the page.
- **Vibe search**: describe a sound ("glitchy angelic vocals over a heavy kick") and get the closest likes.
- **Cover search**: describe artwork ("a pink blurry photo") and get the matching covers.
- **Sets & podcasts** are handled separately: each one is labelled energetic / mixed / calm with a timeline, so
  you can pick one for the right moment of the day.
- **Your own genres**: label a few tracks in the page and a small model learns your vocabulary and tags the rest.

Audio is streamed in short 30 s windows and never written to disk; only tags and embeddings are kept.
It only reads public data: no login, and it never likes, unlikes, follows or posts anything.

## Setup

Tested on macOS with Apple Silicon; Linux should work too (models then run on the CPU).

```sh
git clone <this repo> && cd soundcloud-track-organizer
conda create -p ./.env -c conda-forge python=3.12 ffmpeg   # or any Python 3.12 + ffmpeg on your PATH
./.env/bin/pip install -r requirements.txt
./.env/bin/python -m sctag.download_models                # Essentia models, ~28 MB
```

CLAP and CLIP (~1.5 GB) are downloaded from Hugging Face the first time they're used.

## Run

Your likes must be public. Your username is the part after `soundcloud.com/` in your profile URL.

```sh
./run_all.sh <username>            # fetch likes, analyze everything, build the reports
./.env/bin/python -m sctag.serve   # then open http://127.0.0.1:8765
```

The first run takes a while: about 3.5 hours for 1,700 likes on an M-series Mac, mostly streaming audio.
It's resumable, and rerunning it later only analyzes new likes. To run it overnight without the Mac sleeping
(keep it plugged in with the lid open):

```sh
nohup caffeinate -i -s ./run_all.sh <username> > /dev/null 2>&1 &
tail -f data/run.log
```

The reports (`data/report_tracks.html`, `data/report_sets.html`) also open directly as files, but live search
and labelling need `sctag.serve`. It only listens on your own machine; pass a port if 8765 is taken
(`python -m sctag.serve 8766`).

Individual steps, if you prefer:

```sh
./.env/bin/python -m sctag.fetch_likes <username>     # -> data/likes.json
./.env/bin/python -m sctag.analyze tracks --sample 50 # try a sample first; no --sample = everything
./.env/bin/python -m sctag.analyze sets
./.env/bin/python -m sctag.report                     # -> data/report_*.html
```

## How it works

Likes are split by duration into **tracks** (< 15 min, analyzed at 25/50/75 %) and **sets** (DJ sets, podcasts,
radio shows, >= 15 min, one window every 4 min, kept as a timeline). Each kind is scored and searched on its own.
To move a like to the other kind, add it to `data/kind_overrides.json`: `{"<track id>": "sets"}`.

For each window: [Essentia](https://essentia.upf.edu/models.html) (Discogs-EffNet styles, mood classifiers, BPM),
[CLAP](https://huggingface.co/laion/larger_clap_music_and_speech) (audio ↔ text, used for vibe search, moods and
set energy) and [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) for the cover.

**Genres** come from three sources, from most to least trusted:

1. **Your labels** (`data/labels.json`): set them in the served page (open a track → "Your labels", or the
   "To label" tab, which lists the tracks the model is least sure about). You can type your own tags too.
2. **Title & tags** (`sctag/text_tags.py`): genre words and rapper names in the title, genre field and tags
   (a rapper's name on someone else's upload = "rap edit").
3. **Genre model** (`sctag/classify.py`): learns 1 and 2 from the audio and guesses for the rest (shown with
   a "?"). A genre is learned once 8 tracks have it and its guesses are precise enough.
   Press "Retrain" in the page (or run `sctag.report`) after labelling.

**Set energy** compares each window to "energetic" and "calm" text prompts; a set is energetic or calm when
2/3 of its windows are, otherwise mixed.

## Make it yours

The vocabulary is tuned to one person's library (lots of Brazilian funk and club music). Adapt it freely:

- `sctag/text_tags.py`: `GENRE_PATTERNS` (genre → regexes) and `RAPPERS`.
- `sctag/report.py`: `MOODS` (mood → CLAP prompt), and the example `VIBE_QUERIES` / `COVER_QUERIES`.
- `sctag/analyze.py`: `ENERGY_HOT` / `ENERGY_COOL` (set energy cutoffs; only `sctag.report` needs rerunning).

## Project layout

```
run_all.sh                 full pipeline: fetch → analyze tracks & sets → reports
sctag/fetch_likes.py       likes from SoundCloud's public web API
sctag/analyze.py           stream audio windows, run the models, save tags + embeddings
sctag/text_tags.py         genres from titles and tags
sctag/classify.py          genre model trained on text tags + your labels
sctag/report.py            build the HTML reports
sctag/report_template.html the report page
sctag/serve.py             local server: live search, labelling, retrain
sctag/download_models.py   fetch the Essentia models
data/                      everything about your likes (not in git)
models/                    Essentia models (not in git)
```

## Limits

- DRM-protected major-label tracks can't be streamed and are skipped (about 3 % of likes).
- BPM is often off by half or double; treat it as approximate.
- For personal use. The Essentia models are under a non-commercial Creative Commons license
  ([details](https://essentia.upf.edu/models.html)).

## License

MIT for the code in this repo (see `LICENSE`). The models it downloads have their own licenses (see Limits).
