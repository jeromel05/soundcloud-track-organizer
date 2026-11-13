"""The `sctag` command. Run it from the folder that holds (or should hold) data/ and models/, or set SCTAG_HOME.

Heavy modules are imported inside each command, so `sctag --help` stays instant.
"""

import argparse
import datetime
import logging
import time

from .config import KINDS, LIKES

log = logging.getLogger("sctag")


def wait_until(hhmm):
    start = datetime.datetime.combine(datetime.date.today(), datetime.time.fromisoformat(hhmm))
    if (delay := (start - datetime.datetime.now()).total_seconds()) > 0:
        log.info("waiting until %s", hhmm)
        time.sleep(delay)


def cmd_run(args):
    from .analyze import analyze
    from .models import download
    from .report import build
    from .soundcloud import save_likes

    if args.at:
        wait_until(args.at)
    download()
    try:
        save_likes(args.username, args.since)
    except Exception as e:
        if not LIKES.exists():
            raise
        log.warning("could not refresh likes (%r), using the previous list", e)
    for kind in KINDS:
        analyze(kind, workers=args.workers)
    build()


def cmd_fetch(args):
    from .soundcloud import save_likes

    save_likes(args.username, args.since)


def cmd_analyze(args):
    from .analyze import analyze

    analyze(args.kind, args.sample, args.workers)


def cmd_report(args):
    from .report import build

    build([args.kind] if args.kind else KINDS)


def cmd_serve(args):
    from .serve import serve

    serve(args.port)


def cmd_models(args):
    from .models import download

    download()


def since(value):
    datetime.date.fromisoformat(value)  # raises on anything but YYYY-MM-DD
    return value


SINCE_HELP = "skip likes and reposts older than this date (YYYY-MM-DD); saved to data/settings.json"


def parser():
    p = argparse.ArgumentParser(prog="sctag", description="Tag, browse and search your SoundCloud likes.")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    c = sub.add_parser("run", help="everything: fetch likes, analyze tracks and sets, build the reports")
    c.add_argument("username", help="your SoundCloud username (the part after soundcloud.com/)")
    c.add_argument("--at", metavar="HH:MM", help="start at this time today, e.g. 23:00 for an overnight run")
    c.add_argument("--workers", type=int, default=6, help="parallel audio downloads (default: 6)")
    c.add_argument("--since", type=since, metavar="YYYY-MM-DD", help=SINCE_HELP)
    c.set_defaults(func=cmd_run)

    c = sub.add_parser("fetch", help="download your list of likes to data/likes.json")
    c.add_argument("username", help="your SoundCloud username (the part after soundcloud.com/)")
    c.add_argument("--since", type=since, metavar="YYYY-MM-DD", help=SINCE_HELP)
    c.set_defaults(func=cmd_fetch)

    c = sub.add_parser("analyze", help="analyze the audio of one kind of likes (resumable)")
    c.add_argument("kind", choices=KINDS)
    c.add_argument("--sample", type=int, metavar="N", help="only an evenly spread sample of N likes, to try it out")
    c.add_argument("--workers", type=int, default=4, help="parallel audio downloads (default: 4)")
    c.set_defaults(func=cmd_analyze)

    c = sub.add_parser("report", help="train the genre model and build the HTML reports")
    c.add_argument("kind", nargs="?", choices=KINDS, help="only this kind (default: both)")
    c.set_defaults(func=cmd_report)

    c = sub.add_parser("serve", help="open the reports with live search and labelling")
    c.add_argument("--port", type=int, default=8765, help="default: 8765")
    c.set_defaults(func=cmd_serve)

    c = sub.add_parser("models", help="download the Essentia models")
    c.set_defaults(func=cmd_models)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sctag").setLevel(logging.INFO)  # progress from this package only, not from HTTP / ML libs
    args.func(args)
