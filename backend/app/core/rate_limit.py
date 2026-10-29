import time
import redis.asyncio as redis
from fastapi import Depends, HTTPException, status

from app.core.config import settings
from app.core.security import get_current_user
from app.db.models import User

# Initialize an async Redis connection pool
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


async def check_rate_limit(current_user: User = Depends(get_current_user)) -> User:
    """
    Per-user rate limiting dependency using Redis fixed-window counters.
    Limits each authenticated user to RATE_LIMIT_REQUESTS_PER_MINUTE requests per minute.
    """
    current_minuteWindow = int(time.time() // 60)
    redis_key = f"rate_limit:{current_user.id}:{current_minuteWindow}"

    # Atomically increment the user's request count for the current minute
    request_count = await redis_client.incr(redis_key)

    # If this is the first request in the window, set a 60-second expiration on the key
    if request_count == 1:
        await redis_client.expire(redis_key, 60)

    if request_count > settings.RATE_LIMIT_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute allowed."
        )

    return current_user