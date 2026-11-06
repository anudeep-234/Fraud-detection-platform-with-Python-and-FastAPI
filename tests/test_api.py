from tests.conftest import make_tx


def test_predict_requires_login(client):
    r = client.post("/predict", json=make_tx())
    assert r.status_code == 401


def test_predict_returns_valid_decision(client, viewer_headers):
    r = client.post("/predict", json=make_tx(amount=50.0), headers=viewer_headers)
    assert r.status_code == 200
    assert r.json()["decision"] in {"approve", "review", "block"}
    assert r.json()["model_version"] == "v1"


def test_amount_rule_blocks_and_creates_alert(client, viewer_headers, analyst_headers):
    r = client.post("/predict", json=make_tx(amount=15000.0), headers=viewer_headers)
    assert r.json()["decision"] == "block"
    assert any(x["feature"] == "amount_over_limit" for x in r.json()["reasons"])


def test_viewer_cannot_review(client, viewer_headers):
    r = client.post("/alerts/1/review", json={"label": "fraud"}, headers=viewer_headers)
    assert r.status_code == 403


def test_review_twice_is_rejected(client, viewer_headers, analyst_headers):
    client.post("/predict", json=make_tx(amount=15000.0), headers=viewer_headers)
    open_alerts = client.get("/alerts", headers=analyst_headers).json()
    aid = open_alerts[0]["alert_id"]
    first = client.post(f"/alerts/{aid}/review", json={"label": "legitimate"}, headers=analyst_headers)
    second = client.post(f"/alerts/{aid}/review", json={"label": "legitimate"}, headers=analyst_headers)
    assert first.status_code == 200
    assert second.status_code == 409
