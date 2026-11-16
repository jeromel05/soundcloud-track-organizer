# SoundCloud likes organizer

Tag, browse and search your SoundCloud likes by genre, mood, BPM, energy and cover art, using audio models
that run on your own computer. Made for curating DJ sets out of a big pile of likes.

- **Library**: filter by genre, mood and BPM, search by title or artist, play any track in the page.
- **Vibe search**: describe a sound ("glitchy angelic vocals over a heavy kick") and get the closest likes.
- **Cover search**: describe artwork ("a pink blurry photo") and get the matching covers.
- **Sets & podcasts** are handled separately: each one is labelled energetic / mixed / calm with a timeline, so
  you can pick one for the right moment of the day.
- **Your own genres**: label a few tracks in the page and a small model learns your vocabulary and tags the rest.
- **On your phone**: turn the genres you pick into private SoundCloud playlists, to listen in the SoundCloud app.

Audio is streamed in short 30 s windows and never written to disk; only tags and embeddings are kept.
It only reads public data: no login, and it never likes, unlikes, follows or posts anything. The one exception is
the playlist script you paste into your browser yourself: it creates private playlists (see [Playlists](#playlists)).

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

All commands (`sctag <command> --help` for details):

| command | what it does |
| --- | --- |
| `sctag run <username>` | everything below, in order |
| `sctag fetch <username>` | download your list of likes to `data/likes.json` |
| `sctag analyze tracks\|sets` | analyze the audio; `--sample 20` to try it on a few likes first |
| `sctag report` | train the genre model and build the HTML reports |
| `sctag serve` | open the reports with live search and labelling |
| `sctag playlists` | report of the SoundCloud playlists you could make; `--browser` writes the script that makes them |
| `sctag models` | download the Essentia models |

## Playlists

To listen by genre on your phone, `sctag playlists` can turn your tags into private SoundCloud playlists. Nothing
is made until you pick it:

1. `sctag playlists` prints every candidate (each genre, mood and set energy with at least 5 likes) with its size,
   and lists them in `data/playlists.json`, all with `"make": false`. It doesn't contact your account.
2. In `data/playlists.json`, set `"make": true` on the ones you want. You can rename them, add
   `"guesses": false` to leave out genres guessed from audio, or add your own playlist combining filters:
   `{"make": true, "title": "dark fast funk", "kind": "tracks", "genre": "baile funk", "mood": "dark", "bpm": [130, 200]}`.
3. Rerun `sctag playlists` to check the picks, then `sctag playlists --browser`. It writes
   `data/playlists_browser.js` and prints how to run it: open soundcloud.com, logged in, open the developer
   console (Cmd+Option+I > Console), paste the script (`pbcopy < data/playlists_browser.js`) and press Enter.
   Firefox first asks you to type `allow pasting`. The script lists the changes and asks before making them.
   To update the playlists after labelling or a new `sctag report`, make and paste a new script.

Why the browser: SoundCloud's bot protection (DataDome) blocks scripts from creating playlists, so this step runs
in your own logged-in browser, with pauses of 10 to 25 s between changes. If SoundCloud shows you a check, solving
it yourself is fine; sctag never tries to get around it, and stops at the first refusal.

The script only creates private playlists titled "sctag · ..." and changes the track list of those playlists. It
never deletes anything, never changes a playlist's title or sharing, and never touches likes, follows or your other
playlists: its one write function refuses anything else. It remembers the playlists it made in your browser,
so a new script updates them instead of making duplicates. Playlists longer than 500 likes are split into
"(1)", "(2)", and so on. A playlist you stop picking is left as it is: delete it in the app if you don't want it.
This uses SoundCloud's website API, which isn't officially documented for this.

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
  soundcloud.py         your likes, from SoundCloud's public web API
  audio.py              streaming audio windows (yt-dlp + ffmpeg) and covers
  models.py             Essentia, CLAP and CLIP wrappers, model download
  analyze.py            per-like analysis: windows -> tags + embeddings
  text_tags.py          genres from titles and tags
  classify.py           the genre model (your labels + text tags -> audio)
  search.py             text -> audio / cover search
  report.py             builds the HTML reports
  report_template.html  the report page
  serve.py              local server: live search, labelling, retrain
  playlists.py          your picked tags -> private SoundCloud playlists
  playlists_browser.js  the script that makes them in your browser (the only part that writes)
tests/                  pytest, no models or network needed (the browser script tests need Node.js)
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
