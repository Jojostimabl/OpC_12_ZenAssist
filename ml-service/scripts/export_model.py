"""Exporter l'artefact entraîné de mission 1 en pickle pour une release."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import joblib
from threadpoolctl import threadpool_limits

from app.model import ROOT, SOURCE_MANIFEST, check_environment, check_pipeline, normalize_claim

PROBES = [
    "There is an incorrect account on my credit report.",
    "My bank charged an unexpected credit card fee.",
    "My mortgage servicer has not applied my payment.",
    "A debt collector is calling me about a debt I do not owe.",
    "I cannot withdraw money from my checking account.",
    "My student loan repayment amount is incorrect.",
]


def export_model(output: Path = ROOT / "models") -> dict:
    source = ROOT / "trained/selected_pipeline.joblib"
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_MANIFEST["sourceSha256"]:
        raise RuntimeError("Training artefact checksum mismatch.")
    check_environment(SOURCE_MANIFEST["environment"])
    pipeline = joblib.load(source)
    check_pipeline(pipeline)
    dumped = pickle.dumps(pipeline, protocol=5)
    restored = pickle.loads(dumped)
    check_pipeline(restored)
    with threadpool_limits(limits=2):
        claims = [normalize_claim(text) for text in PROBES]
        if pipeline.predict(claims).tolist() != restored.predict(claims).tolist():
            raise RuntimeError("Export changed model predictions.")
    metadata = {
        "modelVersion": SOURCE_MANIFEST["modelVersion"],
        "sourceSha256": SOURCE_MANIFEST["sourceSha256"],
        "modelSha256": hashlib.sha256(dumped).hexdigest(),
        "labels": SOURCE_MANIFEST["labels"],
        "environment": SOURCE_MANIFEST["environment"],
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "preprocessing": "Unicode NFC; collapse whitespace; preserve case, numbers and punctuation.",
        "training": SOURCE_MANIFEST["training"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.pkl").write_bytes(dumped)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = json.loads((ROOT / "trained/metrics.json").read_text(encoding="utf-8"))
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    result = export_model(args.output)
    print(json.dumps({"modelVersion": result["modelVersion"], "modelSha256": result["modelSha256"], "labels": len(result["labels"])}))
