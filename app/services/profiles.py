import json
import math
from collections.abc import Mapping
from datetime import datetime

from sqlalchemy.orm import Session

from app import models_db as m

DEFAULT_PROFILE = {
    "transaction_count": 0,
    "average_amount": 0.0,
    "amount_stddev": 0.0,
    "usual_countries": [],
    "typical_hour": 12.0,
    "past_alert_count": 0,
}


def _loads(value: str, fallback: Mapping) -> dict:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return dict(fallback)
    return parsed if isinstance(parsed, dict) else dict(fallback)


def profile_values(profile: m.CustomerBehaviorProfile | None) -> dict:
    if profile is None:
        return dict(DEFAULT_PROFILE)
    return {
        "transaction_count": profile.transaction_count,
        "average_amount": profile.average_amount,
        "amount_stddev": profile.amount_stddev,
        "usual_countries": json.loads(profile.usual_countries),
        "typical_hour": profile.typical_hour,
        "past_alert_count": profile.past_alert_count,
    }


def get_profile(db: Session, user_id: int) -> dict:
    return profile_values(
        db.get(m.CustomerBehaviorProfile, user_id)
    )


def update_profile(
    db: Session,
    user_id: int,
    amount: float,
    country: str | None,
    transaction_time: float,
    alerted: bool,
) -> dict:
    profile = db.get(m.CustomerBehaviorProfile, user_id)
    if profile is None:
        profile = m.CustomerBehaviorProfile(user_id=user_id)
        db.add(profile)
        db.flush()

    amount_sum = profile.amount_sum + amount
    amount_sum_squares = profile.amount_sum_squares + amount * amount
    count = profile.transaction_count + 1
    mean = amount_sum / count
    variance = max(0.0, (amount_sum_squares / count) - (mean * mean))

    country_counts = _loads(profile.country_counts, {})
    if country:
        country_counts[country.upper()] = country_counts.get(country.upper(), 0) + 1
    usual_countries = [
        key for key, _ in sorted(
            country_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ][:5]

    hour = int((transaction_time // 3600) % 24)
    hour_counts = _loads(profile.hour_counts, {})
    hour_counts[str(hour)] = hour_counts.get(str(hour), 0) + 1
    typical_hour = sum(int(key) * value for key, value in hour_counts.items()) / sum(
        hour_counts.values()
    )

    profile.transaction_count = count
    profile.average_amount = mean
    profile.amount_stddev = math.sqrt(variance)
    profile.usual_countries = json.dumps(usual_countries)
    profile.typical_hour = typical_hour
    profile.past_alert_count += int(alerted)
    profile.amount_sum = amount_sum
    profile.amount_sum_squares = amount_sum_squares
    profile.hour_counts = json.dumps(hour_counts)
    profile.country_counts = json.dumps(country_counts)
    profile.updated_at = datetime.utcnow()
    return profile_values(profile)


def scoring_profile(profile: dict, country: str | None) -> dict[str, float]:
    usual_countries = profile.get("usual_countries", [])
    return {
        "profile_average_amount": float(profile["average_amount"]),
        "profile_amount_stddev": float(profile["amount_stddev"]),
        "profile_typical_hour": float(profile["typical_hour"]),
        "profile_past_alert_count": float(profile["past_alert_count"]),
        "profile_country_match": float(bool(country and country.upper() in usual_countries)),
    }
