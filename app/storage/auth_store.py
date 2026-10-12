import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
import uuid

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.storage.database import get_database_path

bearer_scheme = HTTPBearer(auto_error=False)
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7


def initialize_auth_store() -> None:
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, ?)",
            ("auth_token_secret", secrets.token_urlsafe(48)),
        )


def _token_secret(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT value FROM app_settings WHERE key = 'auth_token_secret'"
    ).fetchone()
    if row is None:
        raise RuntimeError("Authentication store is not initialized")
    return str(row[0])


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return f"{salt.hex()}${password_hash.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, expected_hex = stored_hash.split("$", 1)
        actual_hash = _hash_password(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual_hash, expected_hex)


def _encode_token(payload: dict[str, str | int], secret: str) -> str:
    encoded_payload = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).rstrip(b"=")
    signature = hmac.new(
        secret.encode("utf-8"), encoded_payload, hashlib.sha256
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return f"{encoded_payload.decode('ascii')}.{encoded_signature.decode('ascii')}"


def _decode_token(token: str, secret: str) -> dict[str, str | int]:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload_bytes = encoded_payload.encode("ascii")
        supplied_signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        expected_signature = hmac.new(
            secret.encode("utf-8"), payload_bytes, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError("Invalid token signature")
        payload = json.loads(
            base64.urlsafe_b64decode(
                encoded_payload + "=" * (-len(encoded_payload) % 4)
            )
        )
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("sub"), str)
            or not isinstance(payload.get("exp"), int)
            or payload["exp"] <= int(time.time())
        ):
            raise ValueError("Invalid or expired token")
        return payload
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as error:
        raise HTTPException(status_code=401, detail="Invalid or expired access token") from error


def register_user(email: str, password: str) -> tuple[dict[str, str], str]:
    normalized_email = email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized_email):
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    if len(password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")
    if len(password) > 128:
        raise HTTPException(status_code=422, detail="Password cannot exceed 128 characters")

    database_path = get_database_path()
    user = {
        "id": str(uuid.uuid4()),
        "email": normalized_email,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM users WHERE email = ?", (normalized_email,)
        ).fetchone():
            raise HTTPException(status_code=409, detail="An account with this email already exists")
        is_first_user = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        connection.execute(
            """
            INSERT INTO users (id, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user["id"], user["email"], _hash_password(password), user["created_at"]),
        )
        if is_first_user:
            from app.storage.session_store import claim_unowned_data

            claim_unowned_data(connection, user["id"])
        secret = _token_secret(connection)

    token = _encode_token(
        {"sub": user["id"], "exp": int(time.time()) + TOKEN_TTL_SECONDS},
        secret,
    )
    return user, token


def login_user(email: str, password: str) -> tuple[dict[str, str], str]:
    normalized_email = email.strip().lower()
    with sqlite3.connect(get_database_path()) as connection:
        connection.row_factory = sqlite3.Row
        user_row = connection.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
        if user_row is None or not _verify_password(password, user_row["password_hash"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        user = {
            "id": user_row["id"],
            "email": user_row["email"],
            "created_at": user_row["created_at"],
        }
        secret = _token_secret(connection)

    token = _encode_token(
        {"sub": user["id"], "exp": int(time.time()) + TOKEN_TTL_SECONDS},
        secret,
    )
    return user, token


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, str]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication required")
    with sqlite3.connect(get_database_path()) as connection:
        payload = _decode_token(credentials.credentials, _token_secret(connection))
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, email FROM users WHERE id = ?",
            (payload["sub"],),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Account no longer exists")
    return {"id": row["id"], "email": row["email"]}
