import hashlib
import json
from pathlib import Path
import pickle
import shutil

from fastapi.testclient import TestClient
import joblib
import pytest

from app.main import create_app
from app.model import LABELS, ROOT, ModelBundle, normalize_claim
from scripts.export_model import PROBES, export_model


@pytest.fixture(scope="session")
def exported(tmp_path_factory):
    target = tmp_path_factory.mktemp("model")
    export_model(target)
    return target


@pytest.fixture(scope="module")
def client(exported):
    with TestClient(create_app(exported)) as test_client:
        yield test_client


def test_health_and_real_prediction(client):
    health = client.get("/health").json()
    assert health["status"] == "ready" and health["categories"] == 18
    answer = client.post("/tags", json={"user_claim": "An incorrect account appears on my credit report."})
    assert answer.status_code == 200
    result = answer.json()
    assert result["tag"] in LABELS and result["model"] == "logistic_regression"
    assert result["modelVersion"] == health["modelVersion"]
    assert result["elapsedMs"] >= 0


@pytest.mark.parametrize("body", [
    {}, {"user_claim": None}, {"user_claim": 123}, {"user_claim": ""},
    {"user_claim": " \n\t"}, {"user_claim": "XXXX 1234"}, {"user_claim": "not provided"},
    {"user_claim": "x" * 100_001}, {"user_claim": "Complaint", "extra": "private information"},
])
def test_invalid_inputs_are_rejected_without_echoing_content(client, body):
    response = client.post("/tags", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_CLAIM"
    assert "private information" not in response.text


def test_invalid_json(client):
    response = client.post("/tags", content="not-json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_text_normalization_matches_mission1(client):
    plain = "My credit report contains an incorrect account."
    messy = "  My  credit\nreport contains an incorrect account.\t"
    assert normalize_claim(messy) == plain
    assert client.post("/tags", json={"user_claim": plain}).json()["tag"] == client.post("/tags", json={"user_claim": messy}).json()["tag"]


def test_export_preserves_source_predictions_and_metrics(exported):
    source = joblib.load(ROOT / "trained/selected_pipeline.joblib")
    bundle = ModelBundle.load(exported)
    assert source.predict(PROBES).tolist() == [bundle.predict(text) for text in PROBES]
    metadata = json.loads((exported / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["modelSha256"] == hashlib.sha256((exported / "model.pkl").read_bytes()).hexdigest()
    assert metadata["training"]["fitPartition"] == "train"
    assert metadata["training"]["selectionPartition"] == "validation"
    assert json.loads((exported / "metrics.json").read_text(encoding="utf-8"))["evaluations"][0]["n"] == 1000
    # The published object is a genuine standard pickle, not a renamed joblib.
    assert pickle.loads((exported / "model.pkl").read_bytes()).predict(PROBES).tolist() == source.predict(PROBES).tolist()


def test_corrupt_model_refused_before_unpickling(exported, tmp_path, monkeypatch):
    shutil.copytree(exported, tmp_path / "bad")
    (tmp_path / "bad/model.pkl").write_bytes(b"corrupt")
    monkeypatch.setattr(pickle, "loads", lambda _data: pytest.fail("Unpickling must not occur"))
    with pytest.raises(RuntimeError, match="checksum"):
        ModelBundle.load(tmp_path / "bad")


def test_incompatible_environment_refused(exported, tmp_path):
    shutil.copytree(exported, tmp_path / "bad")
    target = tmp_path / "bad/metadata.json"
    metadata = json.loads(target.read_text(encoding="utf-8"))
    metadata["environment"]["packages"]["scikit-learn"] = "0.0.0"
    target.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incompatible"):
        ModelBundle.load(tmp_path / "bad")


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path)):
            pytest.fail("An unloaded model must not accept traffic")


def test_model_loaded_once_per_process(client, monkeypatch):
    monkeypatch.setattr(ModelBundle, "load", lambda _directory: pytest.fail("Do not reload per request"))
    for _ in range(2):
        assert client.post("/tags", json={"user_claim": "My mortgage payment is missing."}).status_code == 200


def test_service_busy_returns_503_without_predicting(client, monkeypatch):
    capacity = client.app.state.capacity
    capacity.acquire()
    capacity.acquire()
    try:
        monkeypatch.setattr(client.app.state.bundle, "predict", lambda _text: pytest.fail("Capacity exceeded"))
        assert client.post("/tags", json={"user_claim": "My loan payment is missing."}).status_code == 503
    finally:
        capacity.release()
        capacity.release()


def test_inference_failure_is_safe_and_releases_capacity(client, monkeypatch):
    def fail(_text):
        raise RuntimeError("private implementation detail")
    monkeypatch.setattr(client.app.state.bundle, "predict", fail)
    response = client.post("/tags", json={"user_claim": "My loan payment is missing."})
    assert response.status_code == 500 and "private" not in response.text
    assert client.app.state.capacity.acquire(blocking=False)
    assert client.app.state.capacity.acquire(blocking=False)
    client.app.state.capacity.release()
    client.app.state.capacity.release()
