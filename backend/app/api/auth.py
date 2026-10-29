from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import User, RefreshToken
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
    get_current_user,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ==========================================
# Pydantic Request & Response Schemas
# ==========================================
class AuthRequest(BaseModel):
    email: str = Field(..., examples=["ibrahim@example.com"])
    password: str = Field(..., min_length=6, examples=["strongpassword123"])


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str


# ==========================================
# Endpoints
# ==========================================
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(payload: AuthRequest, db: AsyncSession = Depends(get_db)):
    """Registers a new tenant user and issues initial JWT + refresh tokens."""
    # Check if a user with this email already exists
    existing = await db.execute(select(User).where(User.email == payload.email.lower()))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered"
        )

    # Create new user with Argon2 hashed password
    new_user = User(
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password)
    )
    db.add(new_user)
    await db.flush()  # Flush to generate new_user.id before committing

    # Issue access token and store hashed refresh token
    access_token = create_access_token(str(new_user.id))
    raw_refresh, refresh_hash, expires_at = create_refresh_token()

    db_refresh = RefreshToken(
        user_id=new_user.id,
        token_hash=refresh_hash,
        expires_at=expires_at
    )
    db.add(db_refresh)
    await db.commit()

    return TokenResponse(access_token=access_token, refresh_token=raw_refresh)


@router.post("/login", response_model=TokenResponse)
async def login_user(payload: AuthRequest, db: AsyncSession = Depends(get_db)):
    """Authenticates a user and issues a new JWT access token and refresh token."""
    result = await db.execute(select(User).where(User.email == payload.email.lower()))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    access_token = create_access_token(str(user.id))
    raw_refresh, refresh_hash, expires_at = create_refresh_token()

    db_refresh = RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        expires_at=expires_at
    )
    db.add(db_refresh)
    await db.commit()

    return TokenResponse(access_token=access_token, refresh_token=raw_refresh)


@router.post("/refresh", response_model=TokenResponse)
async def rotate_refresh_token(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """
    Implements Refresh Token Rotation:
    1. Verifies the incoming refresh token is valid, not revoked, and not expired.
    2. Revokes the used refresh token immediately.
    3. Issues a brand-new access token and refresh token pair.
    """
    incoming_hash = hash_refresh_token(payload.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == incoming_hash)
    )
    stored_token = result.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if not stored_token or stored_token.revoked or stored_token.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or already revoked refresh token"
        )

    # Revoke the old token (Rotation)
    stored_token.revoked = True

    # Issue new token pair
    access_token = create_access_token(str(stored_token.user_id))
    raw_refresh, new_refresh_hash, expires_at = create_refresh_token()

    new_db_refresh = RefreshToken(
        user_id=stored_token.user_id,
        token_hash=new_refresh_hash,
        expires_at=expires_at
    )
    db.add(new_db_refresh)
    await db.commit()

    return TokenResponse(access_token=access_token, refresh_token=raw_refresh)


@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user: User = Depends(get_current_user)):
    """Returns the currently authenticated tenant user's profile."""
    return UserResponse(id=str(current_user.id), email=current_user.email)