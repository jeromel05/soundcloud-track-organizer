import numpy as np

from sctag.classify import GenreModel, cutoff


def test_cutoff_keeps_precision_above_target():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    p = np.linspace(1, 0.1, len(y))
    assert cutoff(y, p) == p[3]  # top 4: 3/4 = 0.75 >= 0.7, top 5: 0.6
    assert cutoff(np.zeros(8), p) is None


def synthetic(n=120, dim=16, seed=0):
    """Two genres that live in different directions of the embedding space."""
    rng = np.random.default_rng(seed)
    is_funk = np.arange(n) % 2 == 0
    clap = rng.normal(size=(n, dim)) + np.where(is_funk[:, None], 3.0, -3.0) * np.eye(dim)[0]
    effnet = rng.normal(size=(n, dim))
    recs = [{"id": i} for i in range(n)]
    labels = {str(i): {"genres": ["funk" if f else "ambient"]} for i, f in enumerate(is_funk)}
    return recs, clap, effnet, labels, is_funk


def test_learns_genres_from_labels():
    recs, clap, effnet, labels, is_funk = synthetic()
    gm = GenreModel(recs, clap, effnet, labels)
    assert set(gm.learned()) == {"funk", "ambient"}
    scores = gm.tracks(clap, effnet)
    assert (scores[:, gm.genres.index("funk")] >= 1)[is_funk].mean() > 0.9
    assert gm.windows(clap[:5]).shape == (5, 2)


def test_nothing_to_learn():
    recs, clap, effnet, _, _ = synthetic()
    gm = GenreModel(recs, clap, effnet, {str(r["id"]): {"genres": ["funk"]} for r in recs})  # no negatives
    assert gm.genres == [] and gm.learned() == []
    assert gm.tracks(clap, effnet).shape == (len(recs), 0)
