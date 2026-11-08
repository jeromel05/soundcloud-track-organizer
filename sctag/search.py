"""Text -> audio and text -> cover search over the saved embeddings of one kind."""

import numpy as np

from .config import ANALYSIS
from .models import unit

CLIP_DIM = 512


def load_arrays(path):
    """An .npz as a plain dict (np.load keeps the file open otherwise)."""
    with np.load(path) as z:
        return dict(z)


def best(scores, n):
    """[(index, score)] of the n highest scores."""
    return [(int(j), round(float(scores[j]), 3)) for j in np.argsort(-scores)[:n]]


class Index:
    def __init__(self, kind, ids, arrays):
        self.kind, self.ids = kind, ids
        self.clap = unit(np.stack([a["clap"] for a in arrays]))
        self.windows = [unit(a["clap_windows"]) for a in arrays]
        self.has_cover = np.array(["cover_clip" in a for a in arrays])
        self.covers = np.stack([a.get("cover_clip", np.zeros(CLIP_DIM)) for a in arrays])

    @classmethod
    def from_disk(cls, kind):
        files = sorted((ANALYSIS / kind).glob("*.npz"))
        return cls(kind, [int(f.stem) for f in files], [load_arrays(f) for f in files])

    def audio_scores(self, text_emb):
        """Similarity of every like to a CLAP text embedding. Sets match on their best window: one passage is enough."""
        if self.kind == "sets":
            return np.array([np.max(w @ text_emb) for w in self.windows])
        return self.clap @ text_emb

    def cover_scores(self, text_emb):
        """Similarity of every cover to a CLIP text embedding; -1 for likes without a cover."""
        return np.where(self.has_cover, self.covers @ text_emb, -1.0)
