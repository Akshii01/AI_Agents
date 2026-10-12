import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app.routes import chat as chat_routes, sessions as session_routes
from app.schemas import (
    GeneratedGeneralResponse,
    GeneratedRecipeResponse,
    GeneralChatRequest,
    RecipeChatRequest,
    SummaryResponse,
)
from app.services import general_chat, recipe_chat, story
from app.tools import general_chat as general_tools
from app.storage.prompt_store import (
    DEFAULT_PROMPT,
    PROMPT_FILE,
    PROMPT_KEY,
    get_prompt,
    initialize_prompt_store,
)
from app.storage.auth_store import initialize_auth_store, register_user
from app.storage.session_store import (
    add_chat_exchange,
    add_general_chat_exchange,
    create_session,
    get_chat_history,
    get_general_chat_history,
    get_general_chat_session_summary,
    get_general_long_term_memories,
    get_long_term_memories,
    initialize_session_store,
    update_long_term_memories,
    update_general_long_term_memories,
)


def authenticated_headers(client, email="one@example.com"):
    result = client.post(
        "/auth/register",
        json={"email": email, "password": "correct-horse-battery"},
    )
    assert result.status_code == 201
    return {"Authorization": f"Bearer {result.json()['access_token']}"}


def test_prompt_is_seeded_and_existing_prompt_is_preserved(tmp_path, monkeypatch):
    database_path = tmp_path / "prompts.db"
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(database_path))

    initialize_prompt_store()
    initial_prompt = get_prompt(PROMPT_KEY)
    assert initial_prompt == PROMPT_FILE.read_text(encoding="utf-8").strip()
    assert "Read the entire story" in initial_prompt
    assert "a few clear sentences" in initial_prompt
    assert "not bold or all caps" in initial_prompt
    assert initial_prompt == DEFAULT_PROMPT

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE prompts SET prompt = ? WHERE key = ?",
            ("custom prompt", PROMPT_KEY),
        )

    initialize_prompt_store()
    assert get_prompt(PROMPT_KEY) == "custom prompt"


def test_summarize_uses_database_prompt_and_returns_only_required_fields(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()
    calls = {}

    class FakeGroq:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"key":"wrong-key","content":"# SUMMARY-THE NECKLACE"}'
                        )
                    )
                ]
            )

    monkeypatch.setattr(story, "Groq", FakeGroq)

    with TestClient(main.app) as client:
        headers = authenticated_headers(client)
        response = client.post(
            "/summarize",
            json={"story_name": "The Necklace", "story": "Complete story text"},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json() == {
        "key": "the-necklace",
        "content": "# SUMMARY-THE NECKLACE",
    }
    assert set(response.json()) == {"key", "content"}
    system_prompt = calls["request"]["messages"][0]["content"]
    assert system_prompt.startswith(get_prompt(PROMPT_KEY))
    assert '{"key":"story-name","content":"formatted summary"}' in system_prompt
    assert "exactly two string fields" in system_prompt
    assert calls["request"]["messages"][1]["content"] == (
        "Story name: The Necklace\n\nStory:\nComplete story text"
    )


def test_session_api_persists_summaries_and_returns_session_history(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    initialize_prompt_store()
    monkeypatch.setattr(
        session_routes,
        "generate_summary",
        lambda request: SummaryResponse(
            key="the-necklace", content="# SUMMARY-THE NECKLACE"
        ),
    )

    with TestClient(main.app) as client:
        headers = authenticated_headers(client)
        created = client.post(
            "/sessions", json={"title": "Class stories"}, headers=headers
        )
        assert created.status_code == 201
        session_id = created.json()["id"]

        summarized = client.post(
            f"/sessions/{session_id}/summaries",
            json={"story_name": "The Necklace", "story": "Full story"},
            headers=headers,
        )
        assert summarized.status_code == 201
        assert summarized.json()["key"] == "the-necklace"
        assert summarized.json()["content"] == "# SUMMARY-THE NECKLACE"

        detail = client.get(f"/sessions/{session_id}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["title"] == "The Necklace"
        assert len(detail.json()["summaries"]) == 1
        assert detail.json()["recipe_chat_messages"] == []
        assert client.get("/sessions", headers=headers).json()[0]["id"] == session_id

        missing = client.get("/sessions/not-a-session", headers=headers)
        assert missing.status_code == 404


def test_sessions_are_filtered_by_agent_type_and_routes_enforce_workspace(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))

    with TestClient(main.app) as client:
        headers = authenticated_headers(client)
        created_sessions = {}
        for agent_type in ("story", "recipe", "general"):
            response = client.post(
                "/sessions",
                json={"title": f"{agent_type} workspace", "agent_type": agent_type},
                headers=headers,
            )
            assert response.status_code == 201
            created_sessions[agent_type] = response.json()

        for agent_type, session in created_sessions.items():
            listed = client.get(
                "/sessions",
                params={"agent_type": agent_type},
                headers=headers,
            )
            assert [item["id"] for item in listed.json()] == [session["id"]]
            assert listed.json()[0]["agent_type"] == agent_type

        assert len(client.get("/sessions", headers=headers).json()) == 3
        wrong_workspace = client.post(
            f"/sessions/{created_sessions['recipe']['id']}/general-chat",
            json={"message": "This is the wrong assistant"},
            headers=headers,
        )
        assert wrong_workspace.status_code == 404


def test_chat_history_contains_only_five_latest_exchanges(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    initialize_auth_store()
    initialize_session_store()
    user, _ = register_user("memory@example.com", "correct-horse-battery")
    session = create_session(user["id"], "Memory test")

    for number in range(6):
        add_chat_exchange(
            user["id"],
            session["id"],
            f"user message {number}",
            f"assistant reply {number}",
        )

    legacy_cache = (
        tmp_path
        / "short_term_memory"
        / user["id"]
        / f"{session['id']}.json"
    )
    legacy_cache.parent.mkdir(parents=True)
    legacy_cache.write_text('{"messages":[]}', encoding="utf-8")
    legacy_general_cache = (
        tmp_path
        / "general_short_term_memory"
        / user["id"]
        / f"{session['id']}.json"
    )
    legacy_general_cache.parent.mkdir(parents=True)
    legacy_general_cache.write_text('{"messages":[]}', encoding="utf-8")
    initialize_session_store()

    history = get_chat_history(user["id"], session["id"])
    assert len(history) == 10
    assert history[0] == {"role": "user", "content": "user message 1"}
    assert history[-1] == {"role": "assistant", "content": "assistant reply 5"}
    assert not legacy_cache.exists()
    assert not legacy_general_cache.exists()
    with sqlite3.connect(tmp_path / "prompts.db") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM recipe_chat_messages WHERE session_id = ?",
            (session["id"],),
        ).fetchone()[0] == 12


def test_long_term_memories_update_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    initialize_auth_store()
    initialize_session_store()
    user, _ = register_user("memory@example.com", "correct-horse-battery")

    update_long_term_memories(
        user["id"],
        [{"key": "diet", "value": "vegetarian"}],
        [],
    )
    update_long_term_memories(
        user["id"],
        [{"key": "diet", "value": "vegan"}],
        [],
    )
    assert get_long_term_memories(user["id"]) == [{"key": "diet", "value": "vegan"}]

    update_long_term_memories(user["id"], [], ["diet"])
    assert get_long_term_memories(user["id"]) == []


def test_sessions_and_long_term_memories_are_isolated_by_user(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    with TestClient(main.app) as client:
        first_headers = authenticated_headers(client, "first@example.com")
        first_user_id = client.get("/auth/me", headers=first_headers).json()["id"]
        first_session = client.post(
            "/sessions",
            json={"title": "First user's session"},
            headers=first_headers,
        ).json()
        update_long_term_memories(
            first_user_id,
            [{"key": "dislikes", "value": "olives"}],
            [],
        )

        second_headers = authenticated_headers(client, "second@example.com")
        second_user_id = client.get("/auth/me", headers=second_headers).json()["id"]

        assert client.get("/sessions", headers=second_headers).json() == []
        assert client.get(
            f"/sessions/{first_session['id']}", headers=second_headers
        ).status_code == 404
        assert get_long_term_memories(second_user_id) == []
        assert get_long_term_memories(first_user_id) == [
            {"key": "dislikes", "value": "olives"}
        ]


def test_protected_endpoints_require_authentication(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    with TestClient(main.app) as client:
        assert client.get("/sessions").status_code == 401


def test_first_account_claims_existing_sessions_and_memories(tmp_path, monkeypatch):
    database_path = tmp_path / "prompts.db"
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(database_path))
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE story_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE session_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                story_name TEXT NOT NULL,
                story_key TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE recipe_chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE recipe_long_term_memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO story_sessions VALUES (?, ?, ?, ?)",
            ("legacy-session", "Legacy chat", "2026-01-01", "2026-01-01"),
        )
        connection.execute(
            "INSERT INTO recipe_chat_messages VALUES (1, ?, ?, ?, ?)",
            ("legacy-session", "user", "I prefer vegetarian food", "2026-01-01"),
        )
        connection.execute(
            "INSERT INTO recipe_long_term_memories VALUES (?, ?, ?)",
            ("diet", "vegetarian", "2026-01-01"),
        )

    with TestClient(main.app) as client:
        headers = authenticated_headers(client)
        user_id = client.get("/auth/me", headers=headers).json()["id"]
        legacy_session = client.get(
            "/sessions/legacy-session", headers=headers
        )
        assert legacy_session.status_code == 200
        assert legacy_session.json()["recipe_chat_messages"][0]["content"] == (
            "I prefer vegetarian food"
        )
        recipe_sessions = client.get(
            "/sessions",
            params={"agent_type": "recipe"},
            headers=headers,
        ).json()
        assert [session["id"] for session in recipe_sessions] == [
            "legacy-session"
        ]
        assert get_long_term_memories(user_id) == [
            {"key": "diet", "value": "vegetarian"}
        ]
