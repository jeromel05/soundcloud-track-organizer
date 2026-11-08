"""Streaming: audio windows through yt-dlp + ffmpeg, decoded in memory and never saved, and cover art."""

import io
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .config import SR, WIN_S


def _ffmpeg():
    local = Path(sys.executable).parent / "ffmpeg"  # a conda env's own ffmpeg
    return str(local) if local.exists() else shutil.which("ffmpeg") or "ffmpeg"


FFMPEG = _ffmpeg()


def stream_url(track_url):
    """Direct audio URL and duration of a SoundCloud track (raises on DRM-protected tracks)."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "bestaudio"}) as y:
        info = y.extract_info(track_url, download=False)
    return info["url"], info.get("duration") or 0


def decode(url, start):
    """WIN_S seconds of mono float32 audio at SR, starting at `start` seconds."""
    cmd = [FFMPEG, "-loglevel", "error", "-ss", str(start), "-t", str(WIN_S), "-i", url,
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]  # fmt: skip
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout, dtype=np.float32)


def fetch_cover(artwork_url):
    """The 500x500 cover as a PIL image, or None."""
    url = (artwork_url or "").replace("-large.", "-t500x500.")
    if not url:
        return None
    from PIL import Image

    with urllib.request.urlopen(url, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")
