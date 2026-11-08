"""Personal genre model: learns your genre vocabulary from tagged tracks, then guesses it for the rest.

Labels come from the title / genre field / tags (text_tags: noisy but plentiful) and from your own labels
(data/labels.json, set in the served report), which replace the text tags of that track. One logistic regression
per genre on the audio embeddings, trained on labelled tracks only. Tracks use CLAP + EffNet; sets only keep
per-window CLAP embeddings, so a CLAP-only twin tags each window of a set.
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import LABELS
from .models import unit
from .text_tags import text_tags

MIN_POSITIVES = 8  # a genre is learned once this many labelled tracks have it (and this many don't)
TARGET_PRECISION = 0.7  # per-genre cutoff: as many guesses as possible while cross-validated precision stays above
# (measured against incomplete text tags, so real precision is higher: an untagged baile funk track counts as an error)
AUDIO_ALIASES = {"rap edit": "rap"}  # an edit is a title thing: the audio model learns them as plain rap


def load_labels():
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def track_genres(r, labels):
    """Your labels if you set any for this track, else its text tags; None = unlabelled."""
    mine = labels.get(str(r["id"]), {}).get("genres")
    tags = set(mine) if mine is not None else {g for g, _ in text_tags(r)}
    if mine is None and not tags:
        return None
    return {AUDIO_ALIASES.get(g, g) for g in tags}


def cutoff(y, p):
    """Lowest score whose top-ranked guesses keep precision >= TARGET_PRECISION; None if never reached."""
    order = np.argsort(-p)
    precision = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.where(precision[2:] >= TARGET_PRECISION)[0]  # ignore the first two: too noisy
    return float(p[order[ok[-1] + 2]]) if len(ok) else None


class GenreModel:
    """Trained on construction; scores are probability / cutoff, so >= 1 means "tag it"."""

    def __init__(self, recs, clap, effnet, labels):
        y_sets = [track_genres(r, labels) for r in recs]
        lab = np.array([s is not None for s in y_sets])
        counts = {}
        for s in y_sets:
            for g in s or ():
                counts[g] = counts.get(g, 0) + 1
        self.genres = sorted(
            (g for g, c in counts.items() if c >= MIN_POSITIVES and lab.sum() - c >= MIN_POSITIVES),
            key=lambda g: -counts[g],
        )
        Y = np.array([[g in (s or ()) for g in self.genres] for s in y_sets], dtype=bool).reshape(len(recs), -1)[lab]
        self.full = self._fit(self.features(clap, effnet)[lab], Y)
        self.clap_only = self._fit(unit(clap)[lab], Y)

    @staticmethod
    def features(clap, effnet):
        return np.hstack([unit(clap), unit(effnet)])

    @staticmethod
    def _fit(X, Y):
        models, cuts = [], []
        folds = KFold(5, shuffle=True, random_state=0)
        for y in Y.T:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
            cv = cross_val_predict(clf, X, y, cv=folds, method="predict_proba", n_jobs=-1)[:, 1]
            cuts.append(cutoff(y, cv))
            models.append(clf.fit(X, y))
        return models, np.array([np.inf if c is None else c for c in cuts])

    @staticmethod
    def _predict(fitted, X):
        models, cuts = fitted
        if not models:  # no genre has enough labelled tracks yet
            return np.zeros((len(X), 0))
        p = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
        return p / cuts

    def tracks(self, clap, effnet):
        """(n tracks, n genres) scores from whole-track embeddings."""
        return self._predict(self.full, self.features(clap, effnet))

    def windows(self, clap_windows):
        """(n windows, n genres) scores from per-window CLAP embeddings (sets)."""
        return self._predict(self.clap_only, unit(clap_windows))

    def learned(self):
        """Genres with a usable cutoff (the rest are too hard to guess from audio so far)."""
        return [g for g, c in zip(self.genres, self.full[1], strict=True) if np.isfinite(c)]
