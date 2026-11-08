"""Download the Essentia models (Discogs-EffNet + classifier heads, ~28 MB) into ./models.

Usage: python -m sctag.download_models
CLAP and CLIP are fetched from Hugging Face automatically on first use (~1.5 GB, cached in ~/.cache/huggingface).
"""
import urllib.request

from .analyze import HEADS, MODELS

BASE = "https://essentia.upf.edu/models"


def urls():
    yield f"{BASE}/feature-extractors/discogs-effnet", "discogs-effnet-bs64-1"
    for name, _, _ in HEADS.values():
        folder = name.split("-discogs-effnet")[0].removesuffix("_regression")  # e.g. approachability_regression -> approachability
        yield f"{BASE}/classification-heads/{folder}", name


def main():
    MODELS.mkdir(exist_ok=True)
    for base, name in urls():
        for ext in ("pb", "json"):
            dest = MODELS / f"{name}.{ext}"
            if not dest.exists():
                print(f"downloading {dest.name}")
                urllib.request.urlretrieve(f"{base}/{name}.{ext}", dest)
    print(f"models ready in {MODELS}")


if __name__ == "__main__":
    main()
