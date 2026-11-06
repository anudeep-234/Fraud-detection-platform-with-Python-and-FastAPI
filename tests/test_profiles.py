import time

from tests.conftest import make_tx


def test_new_user_profile_returns_population_defaults(client, analyst_headers):
    user_id = 900000 + int(time.time_ns() % 9999)
    response = client.get(f"/users/{user_id}/profile", headers=analyst_headers)

    assert response.status_code == 200
    assert response.json() == {
        "user_id": user_id,
        "transaction_count": 0,
        "average_amount": 0.0,
        "amount_stddev": 0.0,
        "usual_countries": [],
        "typical_hour": 12.0,
        "past_alert_count": 0,
    }


def test_profile_updates_after_transaction_history(client, viewer_headers, analyst_headers):
    user_id = 900000 + int(time.time_ns() % 9999)
    first = client.post(
        "/predict",
        json=make_tx(amount=100.0, user_id=user_id, time=3600.0),
        headers=viewer_headers,
    )
    second = client.post(
        "/predict",
        json=make_tx(amount=200.0, user_id=user_id, time=7200.0),
        headers=viewer_headers,
    )
    profile = client.get(f"/users/{user_id}/profile", headers=analyst_headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert profile.status_code == 200
    assert profile.json()["transaction_count"] == 2
    assert profile.json()["average_amount"] == 150.0
    assert profile.json()["usual_countries"] == []
    assert profile.json()["typical_hour"] == 1.5
