import hashlib
import hmac
import secrets
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.store.models import ApiKey

SCOPES = frozenset({"ingest", "read", "admin"})
_DUMMY_SALT = "0" * 32
_DUMMY_HASH = hashlib.sha256(bytes.fromhex(_DUMMY_SALT) + b"dummy").hexdigest()


def create_key(session: Session, name: str, scopes: set[str]) -> str:
    if not scopes or not scopes <= SCOPES:
        raise ValueError("invalid scopes")
    prefix = secrets.token_hex(4)
    secret = secrets.token_urlsafe(32)
    token = f"sk_{prefix}_{secret}"
    salt = secrets.token_hex(16)
    session.add(
        ApiKey(
            prefix=prefix,
            salt=salt,
            key_hash=hashlib.sha256(bytes.fromhex(salt) + token.encode()).hexdigest(),
            name=name,
            scopes=",".join(sorted(scopes)),
            revoked=False,
            created_at=datetime.now(UTC),
        )
    )
    session.commit()
    return token


def authenticate(session: Session, token: str | None) -> tuple[ApiKey | None, str | None]:
    parts = token.split("_", 2) if token else []
    prefix = parts[1] if len(parts) == 3 and parts[0] == "sk" and len(parts[1]) == 8 else None
    key = session.scalar(select(ApiKey).where(ApiKey.prefix == prefix)) if prefix else None
    salt = key.salt if key else _DUMMY_SALT
    expected = key.key_hash if key else _DUMMY_HASH
    actual = hashlib.sha256(bytes.fromhex(salt) + (token or "").encode()).hexdigest()
    valid = hmac.compare_digest(actual, expected)
    return (key if valid and key and not key.revoked else None), prefix
