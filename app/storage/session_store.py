import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.storage.database import get_database_path

SHORT_TERM_EXCHANGES = 5


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _database_chat_context(
    connection: sqlite3.Connection, session_id: str
) -> list[dict[str, str]]:
    rows = connection.execute(
        """
        SELECT role, content, created_at
        FROM recipe_chat_messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, SHORT_TERM_EXCHANGES * 2),
    ).fetchall()
    return [
        {"role": role, "content": content, "created_at": created_at}
        for role, content, created_at in reversed(rows)
    ]


def _database_general_chat_context(
    connection: sqlite3.Connection, session_id: str
) -> list[dict[str, str]]:
    rows = connection.execute(
        """
        SELECT role, content, created_at
        FROM general_chat_messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, SHORT_TERM_EXCHANGES * 2),
    ).fetchall()
    return [
        {"role": role, "content": content, "created_at": created_at}
        for role, content, created_at in reversed(rows)
    ]


def _extract_explicit_general_name(message: str) -> str | None:
    text = message.strip()
    name_pattern = r"([A-Za-z][A-Za-z'-]{0,39}(?:\s+[A-Za-z][A-Za-z'-]{0,39}){0,2})"
    for prefix in (
        r"(?:hi[, ]+)?my name is\s+",
        r"(?:hi[, ]+)?call me\s+",
        r"(?:hi[, ]+)?(?:i am|i'm|im)\s+",
    ):
        match = re.fullmatch(prefix + name_pattern + r"[.!?]?", text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if name.lower() not in {
                "a",
                "an",
                "fine",
                "good",
                "happy",
                "indian",
                "american",
                "student",
            }:
                return name
    return None


def _store_general_name(
    connection: sqlite3.Connection, user_id: str, name: str
) -> None:
    connection.execute(
        """
        INSERT INTO general_long_term_memories
            (user_id, key, memory_type, value, updated_at)
        VALUES (?, 'user name', 'fact', ?, ?)
        ON CONFLICT(user_id, key) DO UPDATE SET
            memory_type = excluded.memory_type,
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (user_id, name, _timestamp()),
    )


def _remove_legacy_short_term_caches(database_parent: Path) -> None:
    for directory_name in ("short_term_memory", "general_short_term_memory"):
        cache_directory = database_parent / directory_name
        for cache_file in cache_directory.glob("*/*.json"):
            cache_file.unlink(missing_ok=True)


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
                agent_type TEXT NOT NULL DEFAULT 'story',
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

            CREATE TABLE IF NOT EXISTS recipe_chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES story_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_recipe_chat_messages_session
                ON recipe_chat_messages(session_id, id);

            CREATE TABLE IF NOT EXISTS general_chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES story_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_general_chat_messages_session
                ON general_chat_messages(session_id, id);

            CREATE TABLE IF NOT EXISTS general_chat_session_summaries (
                session_id TEXT PRIMARY KEY REFERENCES story_sessions(id) ON DELETE CASCADE,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS general_long_term_memories (
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'fact',
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );

            CREATE INDEX IF NOT EXISTS idx_session_summaries_session
                ON session_summaries(session_id, created_at);

            CREATE TABLE IF NOT EXISTS recipe_long_term_memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(story_sessions)")
        }
        if "agent_type" not in session_columns:
            connection.execute(
                """
                ALTER TABLE story_sessions
                ADD COLUMN agent_type TEXT NOT NULL DEFAULT 'story'
                """
            )
            connection.execute(
                """
                UPDATE story_sessions
                SET agent_type = CASE
                    WHEN (
                        SELECT COUNT(*) FROM general_chat_messages
                        WHERE general_chat_messages.session_id = story_sessions.id
                    ) > (
                        SELECT COUNT(*) FROM recipe_chat_messages
                        WHERE recipe_chat_messages.session_id = story_sessions.id
                    )
                    AND EXISTS (
                        SELECT 1 FROM general_chat_messages
                        WHERE general_chat_messages.session_id = story_sessions.id
                    ) THEN 'general'
                    WHEN EXISTS (
                        SELECT 1 FROM recipe_chat_messages
                        WHERE recipe_chat_messages.session_id = story_sessions.id
                    ) THEN 'recipe'
                    ELSE 'story'
                END
                """
            )
        general_memory_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(general_long_term_memories)"
            )
        }
        if "memory_type" not in general_memory_columns:
            connection.execute(
                """
                ALTER TABLE general_long_term_memories
                ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'fact'
                """
            )
    _remove_legacy_short_term_caches(database_path.parent)


def claim_unowned_data(connection: sqlite3.Connection, user_id: str) -> None:
    session_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(story_sessions)")
    }
    if "owner_user_id" not in session_columns:
        connection.execute(
            "ALTER TABLE story_sessions ADD COLUMN owner_user_id TEXT REFERENCES users(id)"
        )
    connection.execute(
        "UPDATE story_sessions SET owner_user_id = ? WHERE owner_user_id IS NULL",
        (user_id,),
    )

    memory_columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(recipe_long_term_memories)"
        )
    }
    if memory_columns and "user_id" not in memory_columns:
        connection.execute(
            "ALTER TABLE recipe_long_term_memories RENAME TO recipe_long_term_memories_legacy"
        )
        connection.execute(
            """
            CREATE TABLE recipe_long_term_memories (
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
            """
        )
        legacy_rows = connection.execute(
            "SELECT key, value, updated_at FROM recipe_long_term_memories_legacy"
        ).fetchall()
        connection.executemany(
            """
            INSERT INTO recipe_long_term_memories
                (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            [(user_id, key, value, updated_at) for key, value, updated_at in legacy_rows            ],
        )
        connection.execute("DROP TABLE recipe_long_term_memories_legacy")

    memory_columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(recipe_long_term_memories)"
        )
    }
    if "user_id" not in memory_columns:
        connection.execute(
            """
            CREATE TABLE recipe_long_term_memories (
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
            """
        )

def get_chat_history(user_id: str, session_id: str) -> list[dict[str, str]]:
    with sqlite3.connect(get_database_path()) as connection:
        messages = _database_chat_context(connection, session_id)
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]


def get_general_chat_history(
    user_id: str, session_id: str
) -> list[dict[str, str]]:
    with sqlite3.connect(get_database_path()) as connection:
        messages = _database_general_chat_context(connection, session_id)
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]


def get_general_chat_session_summary(user_id: str, session_id: str) -> str:
    with sqlite3.connect(get_database_path()) as connection:
        row = connection.execute(
            """
            SELECT summary
            FROM general_chat_session_summaries
            JOIN story_sessions ON story_sessions.id = general_chat_session_summaries.session_id
            WHERE general_chat_session_summaries.session_id = ?
                AND story_sessions.owner_user_id = ?
            """,
            (session_id, user_id),
        ).fetchone()
        if row is not None and row[0]:
            return str(row[0])

        rows = connection.execute(
            """
            SELECT content
            FROM general_chat_messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()

    prompts = [str(row[0]).replace("\n", " ")[:120] for row in rows]
    if len(prompts) > 20:
        prompts = prompts[:4] + prompts[-16:]
    seed_parts = []
    seed_chars = 0
    for prompt in prompts:
        part = f"- {prompt}"
        if seed_chars + len(part) + 1 > 1500:
            break
        seed_parts.append(part)
        seed_chars += len(part) + 1
    return "Earlier user messages in this session:\n" + "\n".join(seed_parts) if seed_parts else ""


def add_general_chat_exchange(
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_reply: str,
    session_summary: str,
) -> None:
    created_at = _timestamp()
    with sqlite3.connect(get_database_path()) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute(
            "SELECT 1 FROM story_sessions WHERE id = ? AND owner_user_id = ?",
            (session_id, user_id),
        ).fetchone() is None:
            raise LookupError("Session not found")
        connection.executemany(
            """
            INSERT INTO general_chat_messages
                (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (session_id, "user", user_message, created_at),
                (session_id, "assistant", assistant_reply, created_at),
            ],
        )
        name = _extract_explicit_general_name(user_message)
        if name:
            _store_general_name(connection, user_id, name)
        connection.execute(
            "UPDATE story_sessions SET updated_at = ? WHERE id = ?",
            (created_at, session_id),
        )
        connection.execute(
            """
            INSERT INTO general_chat_session_summaries (session_id, summary, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (session_id, session_summary, created_at),
        )


def get_general_long_term_memories(
    user_id: str,
    query: str | None = None,
    limit: int = 8,
) -> list[dict[str, str]]:
    with sqlite3.connect(get_database_path()) as connection:
        connection.row_factory = sqlite3.Row
        has_saved_name = connection.execute(
            """
            SELECT 1 FROM general_long_term_memories
            WHERE user_id = ? AND LOWER(key) LIKE '%name%'
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        asks_about_name = (
            query is None
            or "name" in query.lower()
            or "who am i" in query.lower()
        )
        if not has_saved_name and asks_about_name:
            historical_messages = connection.execute(
                """
                SELECT general_chat_messages.content
                FROM general_chat_messages
                JOIN story_sessions
                    ON story_sessions.id = general_chat_messages.session_id
                WHERE story_sessions.owner_user_id = ?
                    AND general_chat_messages.role = 'user'
                ORDER BY general_chat_messages.id DESC
                """,
                (user_id,),
            ).fetchall()
            for (content,) in historical_messages:
                name = _extract_explicit_general_name(str(content))
                if name:
                    _store_general_name(connection, user_id, name)
                    break
        rows = connection.execute(
            """
            SELECT key, memory_type, value
            FROM general_long_term_memories
            WHERE user_id = ?
            ORDER BY updated_at DESC, key
            """,
            (user_id,),
        ).fetchall()
    memories = [dict(row) for row in rows]
    if query is None or not query.strip():
        return memories

    stop_words = {
        "a", "about", "all", "am", "an", "and", "are", "did", "do", "for",
        "from", "have", "how", "i", "in", "is", "it", "me", "my", "of",
        "on", "our", "please", "tell", "that", "the", "this", "to", "us",
        "was", "we", "what", "when", "where", "which", "who", "why", "with",
        "you", "your",
    }
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 1 and term not in stop_words
    }
    if not query_terms:
        normalized_query = query.lower()
        if "who am i" in normalized_query or "my name" in normalized_query:
            name_memories = [
                memory
                for memory in memories
                if "name" in memory["key"].lower()
                or "name" in memory["value"].lower()
            ]
            if name_memories:
                return name_memories[:limit]
        return memories[:limit]

    ranked = []
    for index, memory in enumerate(memories):
        key_terms = set(re.findall(r"[a-z0-9]+", memory["key"].lower()))
        value_terms = set(re.findall(r"[a-z0-9]+", memory["value"].lower()))
        matched_key_terms = query_terms & key_terms
        matched_value_terms = query_terms & value_terms
        score = (
            len(matched_key_terms) * 4
            + len(matched_value_terms)
            + (2 if memory["memory_type"] in {"decision", "milestone"} else 0)
        )
        if "who am i" in query.lower() and "name" in memory["key"].lower():
            score += 10
        if memory["memory_type"] == "fact" and query_terms <= {"name", "called", "known"}:
            score += 3
        if score:
            ranked.append((score, index, memory))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked[:limit]]


def update_general_long_term_memories(
    user_id: str,
    updates: list[dict[str, str]],
    deletions: list[str],
) -> None:
    updated_at = _timestamp()
    with sqlite3.connect(get_database_path()) as connection:
        connection.executemany(
            """
            INSERT INTO general_long_term_memories
                (user_id, key, memory_type, value, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                memory_type = excluded.memory_type,
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            [
                (
                    user_id,
                    memory["key"],
                    memory.get("memory_type", "fact"),
                    memory["value"],
                    updated_at,
                )
                for memory in updates
            ],
        )
        connection.executemany(
            "DELETE FROM general_long_term_memories WHERE user_id = ? AND key = ?",
            [(user_id, key) for key in deletions],
        )


def get_long_term_memories(user_id: str) -> list[dict[str, str]]:
    with sqlite3.connect(get_database_path()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT key, value
            FROM recipe_long_term_memories
            WHERE user_id = ?
            ORDER BY key
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_long_term_memories(
    user_id: str,
    updates: list[dict[str, str]],
    deletions: list[str],
) -> None:
    updated_at = _timestamp()
    with sqlite3.connect(get_database_path()) as connection:
        connection.executemany(
            """
            INSERT INTO recipe_long_term_memories (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            [
                (user_id, memory["key"], memory["value"], updated_at)
                for memory in updates
            ],
        )
        connection.executemany(
            "DELETE FROM recipe_long_term_memories WHERE user_id = ? AND key = ?",
            [(user_id, key) for key in deletions],
        )


def add_chat_exchange(
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_reply: str,
) -> None:
    created_at = _timestamp()
    with sqlite3.connect(get_database_path()) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute(
            "SELECT 1 FROM story_sessions WHERE id = ? AND owner_user_id = ?",
            (session_id, user_id),
        ).fetchone() is None:
            raise LookupError("Session not found")
        connection.executemany(
            """
            INSERT INTO recipe_chat_messages
                (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (session_id, "user", user_message, created_at),
                (session_id, "assistant", assistant_reply, created_at),
            ],
        )
        connection.execute(
            "UPDATE story_sessions SET updated_at = ? WHERE id = ?",
            (created_at, session_id),
        )


def create_session(
    user_id: str,
    title: str,
    agent_type: str = "story",
) -> dict[str, str]:
    session = {
        "id": str(uuid.uuid4()),
        "title": title,
        "agent_type": agent_type,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
    }
    with sqlite3.connect(get_database_path()) as connection:
        connection.execute(
            """
            INSERT INTO story_sessions
                (id, owner_user_id, title, agent_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session["id"],
                user_id,
                session["title"],
                session["agent_type"],
                session["created_at"],
                session["updated_at"],
            ),
        )
    return session


def update_session_title(user_id: str, session_id: str, title: str) -> None:
    with sqlite3.connect(get_database_path()) as connection:
        connection.execute(
            "UPDATE story_sessions SET title = ? WHERE id = ? AND owner_user_id = ?",
            (title, session_id, user_id),
        )


def list_sessions(
    user_id: str,
    agent_type: str | None = None,
) -> list[dict[str, str]]:
    with sqlite3.connect(get_database_path()) as connection:
        connection.row_factory = sqlite3.Row
        if agent_type is None:
            rows = connection.execute(
                """
                SELECT id, title, agent_type, created_at, updated_at
                FROM story_sessions
                WHERE owner_user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, title, agent_type, created_at, updated_at
                FROM story_sessions
                WHERE owner_user_id = ? AND agent_type = ?
                ORDER BY updated_at DESC
                """,
                (user_id, agent_type),
            ).fetchall()
        sessions = [dict(row) for row in rows]
        default_titles = {
            "story": "Story session",
            "recipe": "Recipe chat",
            "general": "General chat",
        }
        for session in sessions:
            if session["title"] != default_titles.get(session["agent_type"]):
                continue
            if session["agent_type"] == "story":
                row = connection.execute(
                    """
                    SELECT story_name
                    FROM session_summaries
                    WHERE session_id = ?
                    ORDER BY created_at
                    LIMIT 1
                    """,
                    (session["id"],),
                ).fetchone()
            else:
                table = (
                    "recipe_chat_messages"
                    if session["agent_type"] == "recipe"
                    else "general_chat_messages"
                )
                row = connection.execute(
                    f"""
                    SELECT content
                    FROM {table}
                    WHERE session_id = ? AND role = 'user'
                    ORDER BY id
                    LIMIT 1
                    """,
                    (session["id"],),
                ).fetchone()
            if row is None or not row[0]:
                continue
            title = " ".join(str(row[0]).split())
            if len(title) > 80:
                title = f"{title[:77].rstrip()}..."
            connection.execute(
                "UPDATE story_sessions SET title = ? WHERE id = ? AND owner_user_id = ?",
                (title, session["id"], user_id),
            )
            session["title"] = title
    return sessions


def session_exists(
    user_id: str,
    session_id: str,
    agent_type: str | None = None,
) -> bool:
    with sqlite3.connect(get_database_path()) as connection:
        if agent_type is None:
            row = connection.execute(
                "SELECT 1 FROM story_sessions WHERE id = ? AND owner_user_id = ?",
                (session_id, user_id),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT 1 FROM story_sessions
                WHERE id = ? AND owner_user_id = ? AND agent_type = ?
                """,
                (session_id, user_id, agent_type),
            ).fetchone()
    return row is not None


def get_session(user_id: str, session_id: str) -> dict | None:
    with sqlite3.connect(get_database_path()) as connection:
        connection.row_factory = sqlite3.Row
        session = connection.execute(
            """
            SELECT id, title, agent_type, created_at, updated_at
            FROM story_sessions
            WHERE id = ? AND owner_user_id = ?
            """,
            (session_id, user_id),
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
        chat_messages = connection.execute(
            """
            SELECT role, content, created_at
            FROM recipe_chat_messages
            WHERE session_id = ?
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()
        general_chat_messages = connection.execute(
            """
            SELECT role, content, created_at
            FROM general_chat_messages
            WHERE session_id = ?
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()
    return {
        **dict(session),
        "summaries": [dict(row) for row in summaries],
        "recipe_chat_messages": [dict(row) for row in chat_messages],
        "general_chat_messages": [dict(row) for row in general_chat_messages],
    }


def add_summary(
    user_id: str,
    session_id: str,
    story_name: str,
    story_key: str,
    content: str,
) -> dict[str, str]:
    summary = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "story_name": story_name,
        "story_key": story_key,
        "content": content,
        "created_at": _timestamp(),
    }
    with sqlite3.connect(get_database_path()) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute(
            "SELECT 1 FROM story_sessions WHERE id = ? AND owner_user_id = ?",
            (session_id, user_id),
        ).fetchone() is None:
            raise LookupError("Session not found")
        connection.execute(
            """
            INSERT INTO session_summaries
                (id, session_id, story_name, story_key, content, created_at)
            VALUES
                (:id, :session_id, :story_name, :story_key, :content, :created_at)
            """,
            summary,
        )
        connection.execute(
            "UPDATE story_sessions SET updated_at = ? WHERE id = ?",
            (summary["created_at"], session_id),
        )
    return {
        "id": summary["id"],
        "story_name": summary["story_name"],
        "key": summary["story_key"],
        "content": summary["content"],
        "created_at": summary["created_at"],
    }
