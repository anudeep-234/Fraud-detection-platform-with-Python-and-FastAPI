import time

import redis

from app.config import settings

r = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True,
)

WINDOW_SECONDS = 3600          # one hour
MAX_HISTORY = 100              # keep the most recent 100 transactions per user
KEY_TTL_SECONDS = 7 * 24 * 3600


def _key(user_id: int) -> str:
    return f"tx:{user_id}"


def velocity_features(user_id: int, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    key = _key(user_id)

    latest = r.zrevrange(key, 0, 0, withscores=True)
    time_since_last = (now - latest[0][1]) if latest else 0.0

    tx_count_last_hour = r.zcount(key, now - WINDOW_SECONDS, now)

    return {
        "time_since_last": float(time_since_last),
        "tx_count_last_hour": int(tx_count_last_hour),
    }


def record_transaction(user_id: int, tx_id: str, now: float | None = None) -> None:
    now = now if now is not None else time.time()
    key = _key(user_id)

    pipe = r.pipeline()
    pipe.zadd(key, {str(tx_id): now})
    # Trim by count, not by age, so time_since_last still works after an hour of inactivity
    pipe.zremrangebyrank(key, 0, -(MAX_HISTORY + 1))
    pipe.expire(key, KEY_TTL_SECONDS)
    pipe.execute()
