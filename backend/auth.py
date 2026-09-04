"""
Authentication and authorization.

- Passwords hashed with bcrypt (never stored or logged in plain text).
- Sessions are stateless JWTs signed with a server secret.
- Roles: Reporter (default for every signup), Supervisor, HSE, Admin.
  Admin is never a self-registration choice -- see database.DEFAULT_ADMINS
  and the /admin/users/{id}/role endpoint for how role elevation works.
"""

import os, bcrypt, jwt
from datetime import datetime, timedelta, timezone
from fastapi import Header, HTTPException, Depends

# In a real deployment this MUST come from a secret manager / env var,
# never a hardcoded literal. Read from env with a dev fallback so the
# app still boots locally.
JWT_SECRET = os.environ.get("SIF_JWT_SECRET", "dev-only-secret-change-in-production")
JWT_ALGO = "HS256"
TOKEN_TTL_HOURS = 12

ROLES = ["Reporter", "Supervisor", "HSE", "Admin"]

# ── Password hashing ────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

# ── JWT ──────────────────────────────────────────────────────────────
def create_token(user: dict) -> str:
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid session")

# ── FastAPI dependencies ────────────────────────────────────────────
def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    return {
        "id": int(payload["sub"]),
        "email": payload["email"],
        "name": payload["name"],
        "role": payload["role"],
    }

def require_roles(*allowed_roles):
    """Dependency factory: Depends(require_roles("HSE", "Admin")) gates an
    endpoint to only those roles. Keeps authorization declarative at the
    route instead of scattered inside handler bodies."""
    def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(403, f"Requires role: {' or '.join(allowed_roles)}")
        return user
    return checker
