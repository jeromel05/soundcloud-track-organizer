import json

from sctag import serve


def test_save_label_merges_and_removes(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "LABELS", tmp_path / "labels.json")
    state = serve.State()
    state.save_label({"id": 1, "genres": ["funk"]})
    state.save_label({"id": 1, "energy": "calm"})
    assert json.loads((tmp_path / "labels.json").read_text()) == {"1": {"genres": ["funk"], "energy": "calm"}}
    state.save_label({"id": 1, "genres": None, "energy": None})
    assert serve.read_labels() == {}
