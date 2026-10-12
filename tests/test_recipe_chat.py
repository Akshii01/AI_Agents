import json
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


def test_recipe_chat_is_saved_and_returned_in_session_history(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    initialize_prompt_store()
    initialize_session_store()
    received_history = {}
    received_memories = {}

    def fake_generate_recipe_response(request, history, long_term_memories):
        received_history["value"] = history
        received_memories["value"] = long_term_memories
        return GeneratedRecipeResponse(
            reply=f"Reply to: {request.message}",
            memory_updates=[{"key": "dislikes", "value": "mushrooms"}],
            memory_deletions=[],
        )

    monkeypatch.setattr(
        chat_routes,
        "generate_recipe_response",
        fake_generate_recipe_response,
    )

    with TestClient(main.app) as client:
        headers = authenticated_headers(client)
        user_id = client.get("/auth/me", headers=headers).json()["id"]
        created = client.post(
            "/sessions",
            json={"title": "Recipe ideas", "agent_type": "recipe"},
            headers=headers,
        )
        session_id = created.json()["id"]

        response = client.post(
            f"/sessions/{session_id}/recipe-chat",
            json={"message": "What can I make with lentils?"},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json() == {"reply": "Reply to: What can I make with lentils?"}
        assert received_history["value"] == []
        assert received_memories["value"] == []
        assert get_long_term_memories(user_id) == [
            {"key": "dislikes", "value": "mushrooms"}
        ]
        assert client.get("/sessions", headers=headers).json()[0]["title"] == (
            "What can I make with lentils?"
        )

        detail = client.get(f"/sessions/{session_id}", headers=headers)
        assert detail.status_code == 200
        assert [
            (message["role"], message["content"])
            for message in detail.json()["recipe_chat_messages"]
        ] == [
            ("user", "What can I make with lentils?"),
            ("assistant", "Reply to: What can I make with lentils?"),
        ]
        short_term_path = (
            tmp_path
            / "short_term_memory"
            / user_id
            / f"{session_id}.json"
        )
        assert not short_term_path.exists()
        with sqlite3.connect(tmp_path / "prompts.db") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM recipe_chat_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0] == 2

        client.post(
            f"/sessions/{session_id}/recipe-chat",
            json={"message": "Can I make it spicy?"},
            headers=headers,
        )
        assert received_history["value"] == [
            {"role": "user", "content": "What can I make with lentils?"},
            {"role": "assistant", "content": "Reply to: What can I make with lentils?"},
        ]
        assert received_memories["value"] == [
            {"key": "dislikes", "value": "mushrooms"}
        ]


def test_recipe_response_includes_memories_and_parses_memory_updates(
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
                            content=json.dumps(
                                {
                                    "reply": r"Try a mushroom-free lentil curry.\n\n**Ingredients**",
                                    "memory_updates": [
                                        {"key": "dislikes", "value": "mushrooms"}
                                    ],
                                    "memory_deletions": [],
                                }
                            )
                        )
                    )
                ]
            )

    monkeypatch.setattr(recipe_chat, "Groq", FakeGroq)
    result = recipe_chat.generate_recipe_response(
        RecipeChatRequest(message="I dislike mushrooms."),
        history=[],
        long_term_memories=[{"key": "diet", "value": "vegetarian"}],
    )

    assert result.reply == "Try a mushroom-free lentil curry.\n\n**Ingredients**"
    assert result.memory_updates == [
        {"key": "dislikes", "value": "mushrooms"}
    ]
    request = calls["request"]
    assert request["response_format"] == {"type": "json_object"}
    assert json.dumps(
        [{"key": "diet", "value": "vegetarian"}]
    ) in request["messages"][0]["content"]
