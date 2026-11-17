"""Fetch all liked tracks of a public SoundCloud profile, with like dates, plus the tracks it posted and reposted.

Uses SoundCloud's public web API (api-v2), the same one the website and yt-dlp use. Read-only, no login.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import CLIENT_ID, DATA, LIKES, save_settings, settings

API = "https://api-v2.soundcloud.com"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
log = logging.getLogger(__name__)


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
        self.client_id = CLIENT_ID.read_text().strip() if CLIENT_ID.exists() else None

    def get_json(self, endpoint, **params):
        for attempt in range(2):
            if not self.client_id:
                self.client_id = scrape_client_id()
                CLIENT_ID.write_text(self.client_id)
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
    user = t.get("user") or {}
    return {
        "id": t["id"],
        "title": t.get("title"),
        "artist": user.get("username"),
        "url": t.get("permalink_url"),
        "artwork_url": t.get("artwork_url") or user.get("avatar_url"),
        "sc_genre": t.get("genre") or "",
        "sc_tags": t.get("tag_list") or "",
        "duration_s": round((t.get("duration") or 0) / 1000),
        "full_duration_s": round((t.get("full_duration") or t.get("duration") or 0) / 1000),
        "policy": t.get("policy"),  # SNIP = Go+ track, only a 30 s preview is streamable
        "uploaded_at": t.get("created_at"),
        "release_date": t.get("release_date"),  # official release date, when the artist set one
        "plays": t.get("playback_count"),
        "description": (t.get("description") or "")[:500],
    }


def _paginate(c, url, params, item):
    out = []
    while url:
        page = c.get_json(url, **params)
        out += [x for i in page["collection"] if (x := item(i))]
        url, params = page.get("next_href"), {}
        time.sleep(0.3)
    return out


def fetch_likes(c, user_id):
    """All liked tracks (liked playlists are skipped), newest like first."""
    return _paginate(
        c,
        f"{API}/users/{user_id}/likes",
        {"limit": 200, "linked_partitioning": 1},
        lambda i: i.get("track") and {**slim_track(i["track"]), "liked_at": i["created_at"]},
    )


def fetch_uploads(c, user_id):
    """The public tracks this user posted."""
    return _paginate(c, f"{API}/users/{user_id}/tracks", {"limit": 200, "linked_partitioning": 1}, slim_track)


def fetch_reposts(c, user_id):
    """The tracks this user reposted (reposted playlists are skipped), newest first."""
    return _paginate(
        c,
        f"{API}/stream/users/{user_id}/reposts",
        {"limit": 200, "linked_partitioning": 1},
        lambda i: (
            i.get("type") == "track-repost"
            and i.get("track")
            and {**slim_track(i["track"]), "reposted_at": i["created_at"]}
        ),
    )


def merge(likes, uploads=(), reposts=()):
    """Likes, your posts and your reposts in one list, newest first, each like at most once.

    Posts are marked `posted`, reposts `reposted`; the ones you never liked get `liked: False` and their
    upload / repost date as `liked_at`, so they sort among the likes.
    """
    out = {t["id"]: dict(t) for t in likes}
    for t in uploads:
        out.setdefault(t["id"], {**t, "liked": False, "liked_at": t["uploaded_at"]})["posted"] = True
    for t in reposts:
        out.setdefault(t["id"], {**t, "liked": False, "liked_at": t["reposted_at"]}).update(
            reposted=True, reposted_at=t["reposted_at"]
        )
    return sorted(out.values(), key=lambda t: t["liked_at"] or "", reverse=True)


def skip_before(tracks, since):
    """Drop likes and reposts from before `since` (YYYY-MM-DD); your own posts are always kept."""
    if not since:
        return tracks

    def keep(t):
        liked = t.get("liked") is not False and t["liked_at"] >= since
        reposted = t.get("reposted") and t["reposted_at"] >= since
        return t.get("posted") or liked or reposted

    return [t for t in tracks if keep(t)]


def save_likes(username, since=None):
    """Fetch likes, posts and reposts into likes.json, save the settings; returns the ids that weren't there."""
    DATA.mkdir(exist_ok=True)
    c = Client()
    user = c.get_json(f"{API}/resolve", url=f"https://soundcloud.com/{username}")
    save_settings(username=username, **({"liked_since": since} if since else {}))
    likes = merge(fetch_likes(c, user["id"]), fetch_uploads(c, user["id"]), fetch_reposts(c, user["id"]))
    likes = skip_before(likes, settings().get("liked_since"))
    before = {t["id"] for t in json.loads(LIKES.read_text())} if LIKES.exists() else set()
    LIKES.write_text(json.dumps(likes, indent=1, ensure_ascii=False))
    new = {t["id"] for t in likes} - before
    log.info("wrote %d likes, posts and reposts to %s (%d new)", len(likes), LIKES, len(new))
    return new
