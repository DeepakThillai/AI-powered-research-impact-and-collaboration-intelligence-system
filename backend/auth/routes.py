"""
backend/auth/routes.py - Authentication API endpoints
"""
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from backend.auth.auth_handler import (
    UserCreate, UserLogin, Token, UserOut,
    create_user, authenticate_user, create_access_token,
    get_current_user
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(user_data: UserCreate):
    """Register a new user (student, faculty, or research_head)."""
    user = create_user(user_data)
    return UserOut(
        user_id=user["user_id"],
        name=user["name"],
        email=user["email"],
        department=user["department"],
        role=user["role"],
        created_at=user["created_at"],
    )


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with email and password. Returns JWT access token."""
    user = authenticate_user(form_data.username, form_data.password)
    token = create_access_token({"sub": user["user_id"], "role": user["role"]})
    return Token(
        access_token=token,
        token_type="bearer",
        user=UserOut(
            user_id=user["user_id"],
            name=user["name"],
            email=user["email"],
            department=user["department"],
            role=user["role"],
            created_at=user["created_at"],
        )
    )


@router.get("/me", response_model=UserOut)
def get_me(current_user: dict = Depends(get_current_user)):
    """Get current logged-in user's profile."""
    return UserOut(
        user_id=current_user["user_id"],
        name=current_user["name"],
        email=current_user["email"],
        department=current_user["department"],
        role=current_user["role"],
        created_at=current_user["created_at"],
    )
