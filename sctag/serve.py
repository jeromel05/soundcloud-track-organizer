"""Local server for the reports: live vibe / cover search, your own labels to train the genre model, and a
refresh button that picks up new likes.

Listens on 127.0.0.1 only. Labels are saved to data/labels.json; "Retrain" in the page rebuilds the reports with
them. SoundCloud is only contacted when you press "Refresh likes" (read-only, like `sctag fetch`).
"""

import contextlib
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import KINDS, LABELS, LIKES, report_path, settings
from .search import Index, best

log = logging.getLogger(__name__)
PAGES = {f"/{report_path(k).name}": k for k in KINDS}
REBUILD_EVERY_S = 180  # during a refresh, finished likes show up with their tags at most this late


class State:
    """Models and search indexes, loaded on first use; labels.json writes are serialized."""

    def __init__(self):
        self.lock = threading.Lock()
        self._enc, self._index = None, {}
        self.job = {"state": "idle"}  # the refresh job, polled by the page

    def encoders(self, audio=False):
        """CLAP + CLIP; with audio=True the full AudioModels (a superset), replacing the plain encoders."""
        from .models import AudioModels, Encoders

        with self.lock:
            if audio and not isinstance(self._enc, AudioModels):
                self._enc = AudioModels()
            elif self._enc is None:
                self._enc = Encoders()
            return self._enc

    def index(self, kind):
        with self.lock:
            if kind not in self._index:
                self._index[kind] = Index.from_disk(kind)
            return self._index[kind]

    def search(self, kind, mode, q, n=36):
        """[(track id, score)] best matching a text query."""
        enc, ix = self.encoders(), self.index(kind)
        scores = ix.cover_scores(enc.clip_text([q])[0]) if mode == "cover" else ix.audio_scores(enc.clap_text([q])[0])
        return [(ix.ids[j], s) for j, s in best(scores, n)]

    def save_label(self, body):
        """Merge {id, genres?, energy?} into labels.json; a null value removes that label."""
        with self.lock:
            labels = read_labels()
            key = str(body["id"])
            entry = {**labels.get(key, {}), **{k: body[k] for k in ("genres", "energy") if k in body}}
            entry = {k: v for k, v in entry.items() if v is not None}
            if entry:
                labels[key] = entry
            else:
                labels.pop(key, None)
            LABELS.write_text(json.dumps(labels, indent=1, ensure_ascii=False))
            return labels

    def rebuild(self, kinds=KINDS):
        from .report import build

        enc = self.encoders()
        with self.lock:
            build(kinds, enc=enc)
            self._index.clear()  # pick up newly analyzed likes

    @property
    def refreshing(self):
        return self.job["state"] == "running"

    def start_refresh(self, username):
        """Fetch likes + uploads, analyze the new ones, rebuild, in a background thread. False if one is running."""
        with self.lock:
            if self.refreshing:
                return False
            self.job = {
                "state": "running", "step": "fetch", "done": 0, "total": 0, "new": 0, "failed": 0,
                "fetched": False, "rebuilds": {},
            }  # fmt: skip
        threading.Thread(target=self._refresh, args=(username,), daemon=True).start()
        return True

    def _refresh(self, username):
        from .analyze import analyze
        from .soundcloud import save_likes

        job = self.job
        try:
            new = save_likes(username)
            job.update(new=len(new), fetched=True)  # the page now shows the new likes, untagged

            def rebuild(kind):
                job["step"] = "rebuild"
                self.rebuild([kind])
                job["rebuilds"][kind] = job["rebuilds"].get(kind, 0) + 1  # the page offers to show the new tags
                job["step"] = kind
                last["at"] = time.time()

            def progress(kind, n, total):
                job.update(step=kind, done=n, total=total)
                if n and time.time() - last["at"] >= REBUILD_EVERY_S:
                    rebuild(kind)  # what's analyzed so far, while the rest keeps going

            for kind in KINDS:
                last = {"at": time.time()}
                job.update(step=kind, done=0, total=0)
                failed = analyze(kind, load_models=lambda: self.encoders(audio=True), progress=progress)
                job["failed"] += sum(t["id"] in new for t, _ in failed)  # not the old ones retried every time
                rebuild(kind)  # this kind's page gets its tags without waiting for the next kind
            job["state"] = "done"
        except (Exception, SystemExit) as e:
            log.exception("refresh failed")
            job.update(state="failed", error=repr(e))


def saved_username():
    return settings().get("username")


def read_labels():
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError):  # the page went away (reload) mid-response
                self.wfile.write(body)

        def json(self, obj):
            self.send(200, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self.send_response(302)
                self.send_header("Location", f"/{report_path('tracks').name}")
                self.end_headers()
            elif u.path in PAGES and report_path(PAGES[u.path]).exists():
                self.send(200, report_path(PAGES[u.path]).read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/labels":
                self.json(read_labels())
            elif u.path == "/api/refresh":
                fetched = LIKES.stat().st_mtime if LIKES.exists() else None
                self.json({**state.job, "fetched_at": fetched, **settings()})
            elif u.path == "/api/pending":
                kind = parse_qs(u.query).get("kind", [""])[0]
                if kind not in KINDS:
                    return self.send(400, b"kind required", "text/plain")
                from .analyze import waiting

                self.json(waiting(kind))
            elif u.path == "/api/search":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                if q.get("kind") not in KINDS or not q.get("q"):
                    return self.send(400, b"kind and q required", "text/plain")
                self.json(state.search(q["kind"], q.get("mode", "vibe"), q["q"]))
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            # only this page can write: other sites can't send a JSON POST to localhost without a CORS preflight
            if self.headers.get("Content-Type") != "application/json":
                return self.send(415, b"json only", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/api/labels" and "id" in body:
                self.json(state.save_label(body))
            elif self.path == "/api/refresh":
                username = (body.get("username") or "").strip() or saved_username()
                if not username:
                    return self.send(400, b"username needed", "text/plain")
                if not state.start_refresh(username):
                    return self.send(409, b"a refresh is already running", "text/plain")
                self.json(state.job)
            elif self.path == "/api/rebuild":
                if state.refreshing:
                    return self.send(409, b"a refresh is running", "text/plain")
                try:
                    state.rebuild()
                    self.json({"ok": True})
                except (Exception, SystemExit) as e:
                    log.exception("rebuild failed")
                    self.json({"ok": False, "log": repr(e)})
            else:
                self.send(404, b"not found", "text/plain")

        def log_message(self, fmt, *args):
            if "/api/" in self.path and not self.path.startswith(("/api/refresh", "/api/pending")):  # polled
                log.info("%s %s", self.command, self.path[:80])

    return Handler


def serve(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(State()))
    log.info("http://127.0.0.1:%d  (ctrl-c to stop)", port)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
