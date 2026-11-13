"""Paths and analysis settings.

Everything lives under one folder: $SCTAG_HOME if set, else the current directory (run `sctag` from the repo).
"""

import json
import os
from pathlib import Path

HOME = Path(os.environ.get("SCTAG_HOME") or Path.cwd()).resolve()
DATA = HOME / "data"  # everything about your likes: personal, not in git
MODELS = HOME / "models"  # Essentia models, downloaded by `sctag models`
ANALYSIS = DATA / "analysis"  # <kind>/<id>.json (tags), .npz (embeddings), .error (failed attempt)
LIKES = DATA / "likes.json"
LABELS = DATA / "labels.json"  # your own labels, set in the served report
KIND_OVERRIDES = DATA / "kind_overrides.json"  # {"<track id>": "sets"} moves a like to the other kind
CLIENT_ID = DATA / "client_id.txt"  # cached public SoundCloud web client id
# Your settings, editable by hand: {"username": "...", "liked_since": "YYYY-MM-DD"}. `sctag run` / `fetch` save the
# username (the Refresh button uses it) and --since; likes and reposts older than liked_since are skipped.
SETTINGS = DATA / "settings.json"


def settings():
    return json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}


def save_settings(**changes):
    DATA.mkdir(exist_ok=True)
    SETTINGS.write_text(json.dumps({**settings(), **changes}, indent=1, ensure_ascii=False))


def report_path(kind):
    return DATA / f"report_{kind}.html"


# Likes are split by duration into two kinds, analyzed and searched separately
KINDS = ("tracks", "sets")
SET_MIN_S = 15 * 60  # sets / podcasts / radio shows are at least this long
SR = 44100  # decoding sample rate
WIN_S = 30  # length of each analyzed window
TRACK_WINDOWS = (0.25, 0.5, 0.75)  # tracks: windows at these fractions of the track
SET_STEP_S = 4 * 60  # sets: one window every 4 minutes...
SET_MIN_WINDOWS, SET_MAX_WINDOWS = 4, 30  # ...within these bounds

# Set energy: per window, CLAP similarity to the closest "energetic" prompt minus the closest "calm" one.
ENERGETIC_PROMPTS = [
    "energetic, intense club music with loud driving drums",
    "chaotic, frantic, aggressive noisy music",
    "fast hard dance music for a rave",
]
CALM_PROMPTS = [
    "calm, ambient, contemplative music with soft pads and no drums",
    "quiet, relaxing, meditative music",
    "slow gentle downtempo music",
]
# The prompts lean energetic (median window +0.23), so the cutoffs are asymmetric. Calibrated on a 465-set library
# against Essentia's "relaxed" head (r = -0.94). Only `sctag report` needs rerunning after changing them.
ENERGY_HOT = 0.15  # a window counts as energetic above this...
ENERGY_COOL = 0.03  # ...and as calm below this
ENERGY_MAJORITY = 2 / 3  # share of windows needed to call a whole set energetic or calm, else it's mixed
