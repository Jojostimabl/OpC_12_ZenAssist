"""Chargement d'un artefact approuvé et inférence, sans apprentissage."""
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
import platform
import re
import unicodedata

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = json.loads((ROOT / "trained/manifest.json").read_text(encoding="utf-8"))
LABELS = SOURCE_MANIFEST["labels"]
EMPTY_MARKERS = {"nan", "none", "null", "n/a", "na", "not provided", "no narrative"}


def normalize_claim(text: str) -> str:
    """Même règle NFC/espaces que la préparation de mission 1."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def has_content(text: str) -> bool:
    without_masks = re.sub(r"\b[xX]{2,}\b", "", text)
    return text.casefold() not in EMPTY_MARKERS and bool(re.search(r"[^\W\d_]", without_masks))


def check_environment(expected: dict) -> None:
    if platform.python_version_tuple()[:2] != tuple(expected["python"].split(".")[:2]):
        raise RuntimeError("Python version incompatible with the trained model.")
    for name, version in expected["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Incompatible model dependency: {name}. Use uv sync --locked.")


def check_pipeline(pipeline) -> None:
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["tfidf", "classifier"]:
        raise RuntimeError("Expected the trained TF-IDF/classifier pipeline.")
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression) or classifier.class_weight != "balanced":
        raise RuntimeError("Expected the selected balanced logistic regression.")
    if sorted(pipeline.classes_.tolist()) != sorted(LABELS) or len(LABELS) != 18:
        raise RuntimeError("Model categories do not match the 18 original labels.")


class ModelBundle:
    def __init__(self, pipeline, metadata):
        self.pipeline = pipeline
        self.metadata = metadata

    @classmethod
    def load(cls, directory: Path):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["sourceSha256"] != SOURCE_MANIFEST["sourceSha256"] or metadata["labels"] != LABELS:
            raise RuntimeError("Model provenance or categories do not match this release.")
        check_environment(metadata["environment"])
        # Pickle is executable: accept only our own verified release files.
        data = (directory / "model.pkl").read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata["modelSha256"]:
            raise RuntimeError("Model checksum mismatch; refusing to load it.")
        pipeline = pickle.loads(data)
        check_pipeline(pipeline)
        pipeline.predict(["An incorrect account appears on my credit report."])
        return cls(pipeline, metadata)

    def predict(self, text: str) -> str:
        tag = str(self.pipeline.predict([text])[0])
        if tag not in LABELS:
            raise RuntimeError("Model returned an unsupported tag.")
        return tag
