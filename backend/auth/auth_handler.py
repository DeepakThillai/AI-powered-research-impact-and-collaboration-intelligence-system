"""
backend/auth/auth_handler.py - JWT Authentication & User Management
"""
from datetime import datetime, timedelta
from typing import Optional
import uuid

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from loguru import logger

from backend.config import settings
from backend.database import get_users_collection


# ── Password hashing ──────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Pydantic Schemas ──────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    department: str
    role: str  # student | faculty | research_head


class UserLogin(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    user_id: str
    name: str
    email: str
    department: str
    role: str
    created_at: datetime

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


# ── Core Auth Functions ───────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── User CRUD ─────────────────────────────────────────────────
def create_user(user_data: UserCreate) -> dict:
    col = get_users_collection()

    # Validate role
    valid_roles = {"student", "faculty", "research_head"}
    if user_data.role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Role must be one of: {sorted(valid_roles)}"
        )

    # Check duplicate email (case-insensitive)
    email_normalized = user_data.email.lower().strip()
    if col.find_one({"email": email_normalized}):
        raise HTTPException(status_code=409, detail="Email already registered")

    user_doc = {
        "user_id": str(uuid.uuid4()),
        "name": user_data.name.strip(),
        "email": email_normalized,
        "password_hash": hash_password(user_data.password),
        "department": user_data.department.strip(),
        "role": user_data.role,
        "papers_uploaded": [],
        "created_at": datetime.utcnow(),
        "last_login": None,
        "is_active": True,
    }
    col.insert_one(user_doc)
    logger.info(f"New user created: {email_normalized} [{user_data.role}]")
    return user_doc


def authenticate_user(email: str, password: str) -> dict:
    col = get_users_collection()
    email_normalized = email.lower().strip()
    user = col.find_one({"email": email_normalized})

    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated")

    col.update_one(
        {"email": email_normalized},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    return user


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    col = get_users_collection()
    user = col.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def require_role(*roles: str):
    """Dependency factory: restrict endpoint to specific roles."""
    def _check(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access restricted. Required role(s): {list(roles)}"
            )
        return current_user
    return _check


# Role-specific dependencies
require_faculty_or_above = require_role("faculty", "research_head")
require_research_head = require_role("research_head")
