from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, get_db

SECRET_KEY = settings.jwt_secret
ALGORITHM = "HS256"
TOKEN_HOURS = 8
ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}
bearer = HTTPBearer()


class AuthUser(Base):
    __tablename__ = "auth_users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user: AuthUser) -> str:
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> AuthUser:
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    user = db.get(AuthUser, int(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="user not found")
    return user


def require_role(minimum: str):
    def checker(user: AuthUser = Depends(current_user)) -> AuthUser:
        if ROLE_RANK[user.role] < ROLE_RANK[minimum]:
            raise HTTPException(status_code=403, detail=f"requires role '{minimum}' or higher")
        return user
    return checker
