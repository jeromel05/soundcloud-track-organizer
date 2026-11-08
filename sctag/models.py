"""The audio models: Essentia (Discogs-EffNet styles, mood heads, BPM), CLAP (audio <-> text), CLIP (cover <-> text).

Heavy libraries are imported when a model class is created, so importing this module stays cheap.
"""

import json
import logging
import os
import sys
import urllib.request

import numpy as np

from .config import CALM_PROMPTS, ENERGETIC_PROMPTS, MODELS, SR

log = logging.getLogger(__name__)

# note: the HF port of laion/larger_clap_music has a broken text tower (all prompts embed identically)
CLAP_MODEL = "laion/larger_clap_music_and_speech"
CLIP_MODEL = "openai/clip-vit-base-patch32"
ESSENTIA_URL = "https://essentia.upf.edu/models"
EFFNET = "discogs-effnet-bs64-1"
# Essentia classifier heads on top of the Discogs-EffNet embedding: key -> (file, output node, input node)
HEADS = {
    "genre": ("genre_discogs400-discogs-effnet-1", "PartitionedCall:0", "serving_default_model_Placeholder"),
    "moodtheme": ("mtg_jamendo_moodtheme-discogs-effnet-1", "model/Sigmoid", "model/Placeholder"),
    "aggressive": ("mood_aggressive-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "relaxed": ("mood_relaxed-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "happy": ("mood_happy-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "sad": ("mood_sad-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "party": ("mood_party-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "danceable": ("danceability-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "instrumental": ("voice_instrumental-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "bright": ("nsynth_bright_dark-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "tonal": ("tonal_atonal-discogs-effnet-1", "model/Softmax", "model/Placeholder"),
    "approachability": ("approachability_regression-discogs-effnet-1", "model/Identity", "model/Placeholder"),
    "engagement": ("engagement_regression-discogs-effnet-1", "model/Identity", "model/Placeholder"),
}
REGRESSION_HEADS = ("approachability", "engagement")  # one value; the other binary heads give class probabilities
BINARY_HEADS = [k for k in HEADS if k not in ("genre", "moodtheme", *REGRESSION_HEADS)]


def unit(x):
    """L2-normalize along the last axis."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def model_urls():
    """(url without extension, file name) of every Essentia model used."""
    yield f"{ESSENTIA_URL}/feature-extractors/discogs-effnet", EFFNET
    for name, _, _ in HEADS.values():
        folder = name.split("-discogs-effnet")[0].removesuffix(
            "_regression"
        )  # approachability_regression -> approachability
        yield f"{ESSENTIA_URL}/classification-heads/{folder}", name


def download():
    """Fetch the Essentia models (~28 MB) into MODELS. CLAP and CLIP come from Hugging Face on first use."""
    MODELS.mkdir(exist_ok=True)
    for base, name in model_urls():
        for ext in ("pb", "json"):
            if not (dest := MODELS / f"{name}.{ext}").exists():
                log.info("downloading %s", dest.name)
                urllib.request.urlretrieve(f"{base}/{name}.{ext}", dest)
    log.info("Essentia models ready in %s", MODELS)


class Encoders:
    """CLAP (audio <-> text) and CLIP (cover <-> text): all the report and the search server need."""

    def __init__(self):
        import torch
        import transformers
        from transformers import ClapModel, ClapProcessor, CLIPModel, CLIPProcessor

        transformers.logging.disable_progress_bar()
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.clap = ClapModel.from_pretrained(CLAP_MODEL).to(self.dev).eval()
        self.clap_proc = ClapProcessor.from_pretrained(CLAP_MODEL)
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.dev).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    @staticmethod
    def _emb(out):
        # transformers v5 returns an output object, v4 a tensor
        t = out if hasattr(out, "shape") else out.pooler_output
        return (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def _run(self, fn, inputs):
        with self.torch.no_grad():
            return self._emb(fn(**{k: v.to(self.dev) for k, v in inputs.items()}))

    def clap_audio(self, audio48):
        """One embedding per 10 s chunk of 48 kHz audio."""
        chunks = [audio48[i : i + 480000] for i in range(0, len(audio48) - 48000 * 3, 480000)]
        inputs = self.clap_proc.feature_extractor(chunks, sampling_rate=48000, return_tensors="pt")
        return self._run(self.clap.get_audio_features, inputs)

    def clap_text(self, texts):
        return self._run(
            self.clap.get_text_features, self.clap_proc.tokenizer(texts, padding=True, return_tensors="pt")
        )

    def clip_image(self, img):
        inputs = self.clip_proc(images=img, return_tensors="pt")
        return self._run(self.clip.get_image_features, {"pixel_values": inputs["pixel_values"]})[0]

    def clip_text(self, texts):
        return self._run(
            self.clip.get_text_features, self.clip_proc.tokenizer(texts, padding=True, return_tensors="pt")
        )


class AudioModels(Encoders):
    """Everything used to analyze audio windows: Essentia (EffNet + heads, rhythm) on top of the encoders."""

    def __init__(self):
        if not (MODELS / f"{EFFNET}.pb").exists():
            sys.exit("Essentia models missing: run  sctag models")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import essentia

        essentia.log.infoActive = essentia.log.warningActive = False
        import essentia.standard as es

        super().__init__()
        self.effnet = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODELS / f"{EFFNET}.pb"), output="PartitionedCall:1"
        )
        self.heads = {
            k: es.TensorflowPredict2D(graphFilename=str(MODELS / f"{f}.pb"), output=o, input=i)
            for k, (f, o, i) in HEADS.items()
        }
        self.labels = {k: json.loads((MODELS / f"{f}.json").read_text())["classes"] for k, (f, _, _) in HEADS.items()}
        self.rhythm = es.RhythmExtractor2013(method="multifeature")
        self.resample16 = es.Resample(inputSampleRate=SR, outputSampleRate=16000)
        self.resample48 = es.Resample(inputSampleRate=SR, outputSampleRate=48000)
        self.energetic_text = self.clap_text(ENERGETIC_PROMPTS)
        self.calm_text = self.clap_text(CALM_PROMPTS)

    def energy(self, clap):
        """> 0 = closer to the energetic prompts than to the calm ones."""
        c = clap / np.linalg.norm(clap)
        return float((self.energetic_text @ c).max() - (self.calm_text @ c).max())

    def analyze_window(self, audio, with_energy=False):
        """Embeddings, classifier outputs and BPM of one window of mono audio at SR."""
        emb = self.effnet(self.resample16(audio))  # (patches, 1280)
        preds = {k: np.asarray(h(emb)).mean(axis=0) for k, h in self.heads.items()}
        clap = self.clap_audio(self.resample48(audio)).mean(axis=0)
        bpm, _, conf, _, _ = self.rhythm(audio)
        w = {"effnet": emb.mean(axis=0), "preds": preds, "clap": clap, "bpm": float(bpm), "bpm_conf": float(conf)}
        if with_energy:
            w["energy"] = self.energy(clap)
        return w
