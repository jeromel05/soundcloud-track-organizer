"""Local server for the reports: live vibe / cover search, and your own labels to train the genre model.

Listens on 127.0.0.1 only and never contacts SoundCloud. Labels are saved to data/labels.json;
"Retrain" in the page rebuilds the reports with them.
"""

import contextlib
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import KINDS, LABELS, report_path
from .search import Index, best

log = logging.getLogger(__name__)
PAGES = {f"/{report_path(k).name}": k for k in KINDS}


class State:
    """Models and search indexes, loaded on first use; labels.json writes are serialized."""

    def __init__(self):
        self.lock = threading.Lock()
        self._enc, self._index = None, {}

    def encoders(self):
        with self.lock:
            if self._enc is None:
                from .models import Encoders

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

    def rebuild(self):
        from .report import build

        enc = self.encoders()
        with self.lock:
            build(enc=enc)
            self._index.clear()  # pick up newly analyzed likes


def read_labels():
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
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
            elif self.path == "/api/rebuild":
                try:
                    state.rebuild()
                    self.json({"ok": True})
                except (Exception, SystemExit) as e:
                    log.exception("rebuild failed")
                    self.json({"ok": False, "log": repr(e)})
            else:
                self.send(404, b"not found", "text/plain")

        def log_message(self, fmt, *args):
            if "/api/" in self.path:
                log.info("%s %s", self.command, self.path[:80])

    return Handler


def serve(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(State()))
    log.info("http://127.0.0.1:%d  (ctrl-c to stop)", port)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
