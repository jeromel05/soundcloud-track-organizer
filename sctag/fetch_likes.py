"""Fetch all liked tracks of a public SoundCloud profile, with like dates.

Uses SoundCloud's public web API (api-v2), the same one the website and yt-dlp use.
Usage: python -m sctag.fetch_likes <username>   (the part after soundcloud.com/; likes must be public)
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api-v2.soundcloud.com"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
DATA = Path(__file__).resolve().parent.parent / "data"
CLIENT_ID_CACHE = DATA / "client_id.txt"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def scrape_client_id():
    """Find the public web client_id in soundcloud.com's JS bundles (same trick as yt-dlp)."""
    html = _get("https://soundcloud.com/")
    for src in reversed(re.findall(r'<script[^>]+src="([^"]+)"', html)):
        m = re.search(r'client_id\s*:\s*"([0-9a-zA-Z]{32})"', _get(src))
        if m:
            return m.group(1)
    raise RuntimeError("could not find a SoundCloud client_id")


class Client:
    def __init__(self):
        self.client_id = CLIENT_ID_CACHE.read_text().strip() if CLIENT_ID_CACHE.exists() else None

    def get_json(self, endpoint, **params):
        for attempt in range(2):
            if not self.client_id:
                self.client_id = scrape_client_id()
                CLIENT_ID_CACHE.write_text(self.client_id)
            sep = "&" if "?" in endpoint else "?"
            full = f"{endpoint}{sep}{urllib.parse.urlencode({**params, 'client_id': self.client_id})}"
            try:
                return json.loads(_get(full))
            except urllib.error.HTTPError as e:
                if e.code in (401, 403) and attempt == 0:
                    self.client_id = None  # stale key, rescrape once
                    continue
                raise


def slim_track(t):
    return {
        "id": t["id"],
        "title": t.get("title"),
        "artist": (t.get("user") or {}).get("username"),
        "url": t.get("permalink_url"),
        "artwork_url": t.get("artwork_url") or (t.get("user") or {}).get("avatar_url"),
        "sc_genre": t.get("genre") or "",
        "sc_tags": t.get("tag_list") or "",
        "duration_s": round((t.get("duration") or 0) / 1000),
        "full_duration_s": round((t.get("full_duration") or t.get("duration") or 0) / 1000),
        "policy": t.get("policy"),  # SNIP = Go+ track, only a 30 s preview is streamable
        "uploaded_at": t.get("created_at"),
        "plays": t.get("playback_count"),
        "description": (t.get("description") or "")[:500],
    }


def fetch_likes(username):
    c = Client()
    user = c.get_json(f"{API}/resolve", url=f"https://soundcloud.com/{username}")
    url, params = f"{API}/users/{user['id']}/likes", {"limit": 200, "linked_partitioning": 1}
    out = []
    while url:
        page = c.get_json(url, **params)
        for item in page["collection"]:
            if item.get("track"):  # skip liked playlists
                out.append({**slim_track(item["track"]), "liked_at": item["created_at"]})
        url, params = page.get("next_href"), {}
        print(f"\r{len(out)} likes", end="", file=sys.stderr)
        time.sleep(0.3)
    print(file=sys.stderr)
    return out


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    likes = fetch_likes(sys.argv[1])
    (DATA / "likes.json").write_text(json.dumps(likes, indent=1, ensure_ascii=False))
    print(f"wrote {len(likes)} tracks to {DATA / 'likes.json'}")
