"""Local server for the reports: live vibe / cover search, and your own labels to train the genre model.

Usage: python -m sctag.serve [port]    -> http://127.0.0.1:8765
Listens on this machine only and never contacts SoundCloud. Labels are saved to data/labels.json;
"Retrain" in the page rebuilds the reports with them (same as `python -m sctag.report`).
"""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .analyze import DATA, KINDS, OUT, ROOT
from .classify import unit

LABELS = DATA / "labels.json"
PAGES = {f"/report_{k}.html": DATA / f"report_{k}.html" for k in KINDS}
lock = threading.Lock()
state = {"enc": None, "index": {}}


def read_labels():
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def save_label(body):
    """Merge {id, genres?, energy?} into labels.json; a null value removes that label."""
    with lock:
        labels = read_labels()
        entry = {**labels.get(str(body["id"]), {}), **{k: body[k] for k in ("genres", "energy") if k in body}}
        entry = {k: v for k, v in entry.items() if v is not None}
        if entry:
            labels[str(body["id"])] = entry
        else:
            labels.pop(str(body["id"]), None)
        LABELS.write_text(json.dumps(labels, indent=1, ensure_ascii=False))
    return labels


def index(kind):
    """Embeddings of one kind for live search (loaded once, dropped after a rebuild)."""
    with lock:
        if kind not in state["index"]:
            files = sorted((OUT / kind).glob("*.npz"))
            arr = [np.load(f) for f in files]
            state["index"][kind] = {
                "ids": [int(f.stem) for f in files],
                "clap": unit(np.stack([a["clap"] for a in arr])),
                "windows": [unit(a["clap_windows"]) for a in arr],
                "cover": np.stack([a["cover_clip"] if "cover_clip" in a else np.zeros(512) for a in arr]),
            }
        if state["enc"] is None:
            from .analyze import Encoders
            state["enc"] = Encoders()
        return state["index"][kind], state["enc"]


def search(kind, mode, q, n=36):
    ix, enc = index(kind)
    if mode == "cover":
        sims = ix["cover"] @ enc.clip_text([q])[0]
    elif kind == "sets":  # best-matching window: one fitting passage is enough
        qe = enc.clap_text([q])[0]
        sims = np.array([np.max(w @ qe) for w in ix["windows"]])
    else:
        sims = ix["clap"] @ enc.clap_text([q])[0]
    return [(ix["ids"][j], round(float(sims[j]), 3)) for j in np.argsort(-sims)[:n]]


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
            self.send_header("Location", "/report_tracks.html")
            self.end_headers()
        elif u.path in PAGES and PAGES[u.path].exists():
            self.send(200, PAGES[u.path].read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/labels":
            self.json(read_labels())
        elif u.path == "/api/search":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            self.json(search(q["kind"], q.get("mode", "vibe"), q["q"]))
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        # only this page may write: other sites can't send a JSON POST to localhost without a CORS preflight
        if self.headers.get("Content-Type") != "application/json":
            return self.send(415, b"json only", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/api/labels":
            self.json(save_label(body))
        elif self.path == "/api/rebuild":
            r = subprocess.run([sys.executable, "-m", "sctag.report"], cwd=ROOT, capture_output=True, text=True)
            state["index"].clear()  # pick up newly analyzed likes
            self.json({"ok": r.returncode == 0, "log": (r.stdout + r.stderr)[-3000:]})
        else:
            self.send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):
        if "/api/" in self.path:
            sys.stderr.write(f"{self.command} {self.path[:80]}\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://127.0.0.1:{port}  (ctrl-c to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
