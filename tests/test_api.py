import pytest
from fastapi.testclient import TestClient

from provisionnement.api import create_app
from provisionnement.llm import TemplateExplainer


@pytest.fixture()
def client():
    return TestClient(create_app(TemplateExplainer()))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_run_waits_for_validation(client):
    response = client.post("/runs", json={"branch": "wkcomp"})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "awaiting_validation"
    assert body["result"]["branch"] == "wkcomp"
    assert body["result"]["total_reserve"] > 0
    assert body["explanation"]


def test_review_approved(client):
    run_id = client.post("/runs", json={"branch": "ppauto"}).json()["run_id"]
    response = client.post(f"/runs/{run_id}/review", json={"approved": True})
    assert response.status_code == 200
    assert response.json()["status"] == "validated"


def test_review_rejected(client):
    run_id = client.post("/runs", json={"branch": "ppauto"}).json()["run_id"]
    response = client.post(f"/runs/{run_id}/review", json={"approved": False, "comment": "non"})
    assert response.json()["status"] == "rejected"


def test_cannot_review_twice(client):
    run_id = client.post("/runs", json={"branch": "ppauto"}).json()["run_id"]
    client.post(f"/runs/{run_id}/review", json={"approved": True})
    again = client.post(f"/runs/{run_id}/review", json={"approved": False})
    assert again.status_code == 409


def test_unknown_branch_is_rejected_by_validation(client):
    assert client.post("/runs", json={"branch": "inconnue"}).status_code == 422


def test_unknown_run_is_404(client):
    assert client.get("/runs/n-existe-pas").status_code == 404
    assert client.post("/runs/n-existe-pas/review", json={"approved": True}).status_code == 404
