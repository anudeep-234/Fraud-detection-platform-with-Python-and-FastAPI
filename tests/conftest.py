import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def viewer_headers(client):
    token = client.post("/auth/login", json={"username": "viewer", "password": "viewer-pass"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session")
def analyst_headers(client):
    token = client.post("/auth/login", json={"username": "analyst", "password": "analyst-pass"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def make_tx(amount=50.0, user_id=1, time=1000.0):
    return {
        "user_id": user_id,
        "amount": amount,
        "time": time,
        "v_features": {f"V{i}": 0.0 for i in range(1, 29)},
    }
