# SoundCloud likes organizer

Tag, browse and search your SoundCloud likes by genre, mood, BPM, energy and cover art, using audio models
that run on your own computer. Made for curating DJ sets out of a big pile of likes.

- **Library**: filter by genre, mood and BPM, search by title or artist, play any track in the page.
- **Vibe search**: describe a sound ("glitchy angelic vocals over a heavy kick") and get the closest likes.
- **Cover search**: describe artwork ("a pink blurry photo") and get the matching covers.
- **Sets & podcasts** are handled separately: each one is labelled energetic / mixed / calm with a timeline, so
  you can pick one for the right moment of the day.
- **Your own genres**: label a few tracks in the page and a small model learns your vocabulary and tags the rest.
- **Your posts and reposts** are fetched too; the Source filter picks likes, posts and/or reposts.
- **Refresh likes** in the served page fetches new likes, analyzes them and rebuilds, without the terminal.

Audio is streamed in short 30 s windows and never written to disk; only tags and embeddings are kept.
It only reads public data: no login, and it never likes, unlikes, follows or posts anything.

## Setup

Tested on macOS with Apple Silicon; Linux should work too (models then run on the CPU, or CUDA if available).
You need Python 3.12 and ffmpeg.

```sh
git clone https://github.com/jeromel05/soundcloud-track-organizer.git
cd soundcloud-track-organizer

# either conda (brings its own ffmpeg)...
conda create -p ./.env -c conda-forge python=3.12 ffmpeg && conda activate ./.env
# ...or a plain virtualenv, with ffmpeg from your package manager (brew install ffmpeg)
python3.12 -m venv .venv && source .venv/bin/activate

pip install -e .
```

`sctag run` downloads the models it needs: Essentia's (~28 MB, into `models/`; or `sctag models`) and CLAP + CLIP
from Hugging Face (~1.5 GB, cached in `~/.cache/huggingface`).

## Run

Run `sctag` from the repo folder: your data goes to `data/` there (or set `SCTAG_HOME` to use another folder).
Your likes must be public; your username is the part after `soundcloud.com/` in your profile URL.

```sh
sctag run <username>      # fetch likes, analyze everything, build the reports
sctag serve               # then open http://127.0.0.1:8765
```

The first run takes a while: about 3.5 hours for 1,700 likes on an M-series Mac, mostly streaming audio.
It's resumable, and rerunning it later only analyzes new likes. To run it overnight on a Mac (plugged in,
lid open), or to start it later with `--at 23:00`:

```sh
nohup caffeinate -i -s sctag run <username> > data/run.log 2>&1 &
tail -f data/run.log
```

The reports (`data/report_tracks.html`, `data/report_sets.html`) also open directly as files, but live search
and labelling need `sctag serve`. It only listens on your own machine; use `--port` if 8765 is taken.
Its "Refresh likes" button does what `sctag run` does, in the background, for the username in your settings.

Your settings live in `data/settings.json`, which you can edit by hand:

```json
{"username": "your-name", "liked_since": "2016-10-27"}
```

`sctag run` / `fetch` save the username (the page asks for it on the first Refresh if there's none).
`liked_since` is optional: likes and reposts older than it are left out of every fetch and refresh, while your
own posts are always kept. Set it by hand or with `--since YYYY-MM-DD`.

All commands (`sctag <command> --help` for details):

| command | what it does |
| --- | --- |
| `sctag run <username>` | everything below, in order |
| `sctag fetch <username>` | download your likes, posts and reposts to `data/likes.json` |
| `sctag analyze tracks\|sets` | analyze the audio; `--sample 20` to try it on a few likes first |
| `sctag report` | train the genre model and build the HTML reports |
| `sctag serve` | open the reports with live search and labelling |
| `sctag models` | download the Essentia models |

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
   Press "Retrain" in the page (or run `sctag report`) after labelling.

**Set energy** compares each window to "energetic" and "calm" text prompts; a set is energetic or calm when
2/3 of its windows are, otherwise mixed.

## Make it yours

The vocabulary is tuned to one person's library (lots of Brazilian funk and club music). Adapt it freely:

- `sctag/text_tags.py`: `GENRE_PATTERNS` (genre → regexes) and `RAPPERS`.
- `sctag/report.py`: `MOODS` (mood → CLAP prompt), and the example `VIBE_QUERIES` / `COVER_QUERIES`.
- `sctag/config.py`: set energy cutoffs and prompts, window lengths, the 15 min tracks / sets split.

Changes to the vocabulary or the energy cutoffs only need `sctag report`; window settings need a new analysis.

## Project layout

```
sctag/
  cli.py                the `sctag` command
  config.py             paths and analysis settings
  soundcloud.py         your likes, posts and reposts, from SoundCloud's public web API
  audio.py              streaming audio windows (yt-dlp + ffmpeg) and covers
  models.py             Essentia, CLAP and CLIP wrappers, model download
  analyze.py            per-like analysis: windows -> tags + embeddings
  text_tags.py          genres from titles and tags
  classify.py           the genre model (your labels + text tags -> audio)
  search.py             text -> audio / cover search
  report.py             builds the HTML reports
  report_template.html  the report page
  serve.py              local server: live search, labelling, retrain, refresh
tests/                  pytest, no models or network needed
data/                   everything about your likes (not in git)
models/                 Essentia models (not in git)
```

## Development

```sh
pip install -e ".[dev]"
pytest
ruff check . && ruff format .
```

## Limits

- DRM-protected major-label tracks can't be streamed and are skipped (about 3 % of likes).
- BPM is often off by half or double; treat it as approximate.
- For personal use. The Essentia models are under a non-commercial Creative Commons license
  ([details](https://essentia.upf.edu/models.html)).

## License

MIT for the code in this repo (see `LICENSE`). The models it downloads have their own licenses (see Limits).
