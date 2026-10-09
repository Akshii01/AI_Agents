import sqlite3
import uuid
from datetime import datetime, timezone

from app.prompt_store import get_database_path


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_session_store() -> None:
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS story_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES story_sessions(id) ON DELETE CASCADE,
                story_name TEXT NOT NULL,
                story_key TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_session_summaries_session
                ON session_summaries(session_id, created_at);
            """
        )


def create_session(title: str) -> dict[str, str]:
    session = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
    }
    with sqlite3.connect(get_database_path()) as connection:
        connection.execute(
            """
            INSERT INTO story_sessions (id, title, created_at, updated_at)
            VALUES (:id, :title, :created_at, :updated_at)
            """,
            session,
        )
    return session


def list_sessions() -> list[dict[str, str]]:
    with sqlite3.connect(get_database_path()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM story_sessions
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def session_exists(session_id: str) -> bool:
    with sqlite3.connect(get_database_path()) as connection:
        row = connection.execute(
            "SELECT 1 FROM story_sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return row is not None


def get_session(session_id: str) -> dict | None:
    with sqlite3.connect(get_database_path()) as connection:
        connection.row_factory = sqlite3.Row
        session = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM story_sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if session is None:
            return None

        summaries = connection.execute(
            """
            SELECT id, story_name, story_key AS key, content, created_at
            FROM session_summaries
            WHERE session_id = ?
            ORDER BY created_at
            """,
            (session_id,),
        ).fetchall()
    return {**dict(session), "summaries": [dict(row) for row in summaries]}


def add_summary(
    session_id: str,
    story_name: str,
    story_key: str,
    content: str,
) -> dict[str, str]:
    summary = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "story_name": story_name,
        "key": story_key,
        "content": content,
        "created_at": _timestamp(),
    }
    with sqlite3.connect(get_database_path()) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO session_summaries
                (id, session_id, story_name, story_key, content, created_at)
            VALUES
                (:id, :session_id, :story_name, :key, :content, :created_at)
            """,
            summary,
        )
        connection.execute(
            "UPDATE story_sessions SET updated_at = ? WHERE id = ?",
            (summary["created_at"], session_id),
        )
    return {key: value for key, value in summary.items() if key != "session_id"}
