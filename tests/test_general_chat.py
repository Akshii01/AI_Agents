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


def test_general_chat_uses_search_tool_and_passes_results_to_model(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            assert api_key == "test-key"

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                tool_call = SimpleNamespace(
                    id="search-1",
                    function=SimpleNamespace(
                        name="search_web",
                        arguments=json.dumps({"query": "latest science news"}),
                    ),
                )
                message = SimpleNamespace(content=None, tool_calls=[tool_call])
            elif len(calls) == 2:
                message = SimpleNamespace(
                    content="Here is the update ([source](https://example.com)).",
                    tool_calls=[],
                )
            else:
                message = SimpleNamespace(
                    content=json.dumps(
                        {
                            "session_summary": "The user asked about recent science news.",
                            "memory_updates": [],
                            "memory_deletions": [],
                        }
                    ),
                    tool_calls=[],
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )

    monkeypatch.setattr(general_chat, "Groq", FakeGroq)
    executed = {}

    def fake_execute(name, arguments):
        executed["call"] = (name, arguments)
        return {
            "query": arguments["query"],
            "results": [{"title": "A result", "url": "https://example.com"}],
        }

    monkeypatch.setattr(general_chat, "execute_general_chat_tool", fake_execute)
    response = general_chat.generate_general_response(
        GeneralChatRequest(message="What is the latest science news?"),
        history=[],
        long_term_memories=[],
    )

    assert response.reply == "Here is the update ([source](https://example.com))."
    assert response.tools_used == ["search_web"]
    assert executed["call"] == (
        "search_web",
        {"query": "latest science news"},
    )
    assert calls[0]["tools"] == general_chat.GENERAL_CHAT_TOOLS
    assert calls[0]["max_completion_tokens"] == (
        general_chat.GENERAL_CHAT_MAX_COMPLETION_TOKENS
    )
    assert calls[0].get("response_format") is None
    assert calls[2]["response_format"] == {"type": "json_object"}
    assert "tools" not in calls[2]
    assert calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "search-1",
        "content": json.dumps(
            {
                "query": "latest science news",
                "results": [
                    {"title": "A result", "url": "https://example.com"}
                ],
            }
        ),
    }


def test_general_chat_bounds_multiple_tool_results_instead_of_failing(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()
    calls = []
    tool_calls = [
        SimpleNamespace(
            id="weather-1",
            function=SimpleNamespace(
                name="get_weather",
                arguments=json.dumps(
                    {"location": "Dehradun", "forecast_days": 1}
                ),
            ),
        ),
        SimpleNamespace(
            id="search-1",
            function=SimpleNamespace(
                name="search_web",
                arguments=json.dumps({"query": "latest news of Dehradun"}),
            ),
        ),
    ]

    class FakeGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                message = SimpleNamespace(content=None, tool_calls=tool_calls)
            elif len(calls) == 2:
                message = SimpleNamespace(
                    content="It is warm in Dehradun. Here are recent headlines.",
                    tool_calls=[],
                )
            else:
                message = SimpleNamespace(
                    content=json.dumps(
                        {
                            "session_summary": "",
                            "memory_updates": [],
                            "memory_deletions": [],
                        }
                    ),
                    tool_calls=[],
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(general_chat, "Groq", FakeGroq)
    monkeypatch.setattr(
        general_chat,
        "execute_general_chat_tool",
        lambda name, _arguments: {
            "tool": name,
            "data": "weather and search details " * 100,
        },
    )

    response = general_chat.generate_general_response(
        GeneralChatRequest(
            message="What is the weather and latest news in Dehradun?"
        ),
        history=[],
        long_term_memories=[],
    )

    tool_messages = [
        message
        for message in calls[1]["messages"]
        if message["role"] == "tool"
    ]
    assert response.reply == "It is warm in Dehradun. Here are recent headlines."
    assert response.tools_used == ["get_weather", "search_web"]
    assert len(tool_messages) == 2
    assert sum(len(message["content"]) for message in tool_messages) <= (
        general_chat.GENERAL_CHAT_TOOL_RESULT_CHAR_LIMIT
    )
    assert all(
        json.loads(message["content"])["notice"]
        == "Tool output was shortened to fit the model context."
        for message in tool_messages
    )


def test_general_chat_forces_final_answer_after_tool_call_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if "tools" not in kwargs:
                content = json.dumps(
                    {
                        "session_summary": "",
                        "memory_updates": [],
                        "memory_deletions": [],
                    }
                )
                message = SimpleNamespace(content=content, tool_calls=[])
            elif kwargs["tool_choice"] == "none":
                message = SimpleNamespace(
                    content="Here is the answer based on the retrieved results.",
                    tool_calls=[],
                )
            else:
                call_number = sum(1 for call in calls if call.get("tool_choice") == "auto")
                tool_call = SimpleNamespace(
                    id=f"search-{call_number}",
                    function=SimpleNamespace(
                        name="search_web",
                        arguments=json.dumps({"query": f"search {call_number}"}),
                    ),
                )
                message = SimpleNamespace(content=None, tool_calls=[tool_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(general_chat, "Groq", FakeGroq)
    monkeypatch.setattr(
        general_chat,
        "execute_general_chat_tool",
        lambda _name, arguments: {"query": arguments["query"], "results": []},
    )

    result = general_chat.generate_general_response(
        GeneralChatRequest(message="Search and explain this topic"),
        history=[],
        long_term_memories=[],
    )

    assert result.reply == "Here is the answer based on the retrieved results."
    assert [call["tool_choice"] for call in calls[:4]] == [
        "auto",
        "auto",
        "auto",
        "none",
    ]


def test_general_chat_retries_when_model_calls_tool_after_tool_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) <= general_chat.GENERAL_CHAT_MAX_TOOL_CALLS:
                call_number = len(calls)
                tool_call = SimpleNamespace(
                    id=f"search-{call_number}",
                    function=SimpleNamespace(
                        name="search_web",
                        arguments=json.dumps({"query": "Dehradun news"}),
                    ),
                )
                message = SimpleNamespace(content=None, tool_calls=[tool_call])
            elif len(calls) == general_chat.GENERAL_CHAT_MAX_TOOL_CALLS + 1:
                error = general_chat.GroqError(
                    "Error code: 400 - Tool choice is none, but model called a tool"
                )
                error.status_code = 400
                raise error
            elif "tools" not in kwargs and "response_format" not in kwargs:
                message = SimpleNamespace(
                    content="Here is the latest news from Dehradun.",
                    tool_calls=[],
                )
            else:
                message = SimpleNamespace(
                    content=json.dumps(
                        {
                            "session_summary": "",
                            "memory_updates": [],
                            "memory_deletions": [],
                        }
                    ),
                    tool_calls=[],
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(general_chat, "Groq", FakeGroq)
    monkeypatch.setattr(
        general_chat,
        "execute_general_chat_tool",
        lambda _name, arguments: {"query": arguments["query"], "results": []},
    )

    response = general_chat.generate_general_response(
        GeneralChatRequest(message="What is the latest news of Dehradun?"),
        history=[],
        long_term_memories=[],
    )

    fallback_call = calls[general_chat.GENERAL_CHAT_MAX_TOOL_CALLS + 1]
    assert response.reply == "Here is the latest news from Dehradun."
    assert response.tools_used == ["search_web"]
    assert calls[general_chat.GENERAL_CHAT_MAX_TOOL_CALLS]["tool_choice"] == "none"
    assert "tools" not in fallback_call


def test_general_chat_surfaces_groq_provider_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()

    class FakeGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        general_chat.GroqError("model does not support tool calling")
                    )
                )
            )

    monkeypatch.setattr(general_chat, "Groq", FakeGroq)
    with pytest.raises(HTTPException) as error:
        general_chat.generate_general_response(
            GeneralChatRequest(message="Hello"),
            history=[],
            long_term_memories=[],
        )

    assert error.value.status_code == 502
    assert "model does not support tool calling" in error.value.detail


def test_general_chat_retries_invalid_tool_call_without_tools(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                error = general_chat.GroqError(
                    "Error code: 400 - Tool call validation failed: "
                    "attempted to call tool 'open' which was not in request.tools"
                )
                error.status_code = 400
                raise error
            if len(calls) == 2:
                assert "tools" not in kwargs
                message = SimpleNamespace(
                    content="I couldn't fetch live sources, but I can still explain.",
                    tool_calls=[],
                )
            else:
                message = SimpleNamespace(
                    content=json.dumps(
                        {
                            "session_summary": "",
                            "memory_updates": [],
                            "memory_deletions": [],
                        }
                    ),
                    tool_calls=[],
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(general_chat, "Groq", FakeGroq)

    result = general_chat.generate_general_response(
        GeneralChatRequest(message="What is happening today?"),
        history=[],
        long_term_memories=[],
    )

    assert result.reply == "I couldn't fetch live sources, but I can still explain."
    assert "tools" in calls[0]
    assert "tools" not in calls[1]


def test_general_chat_session_history_is_separate_from_recipe_chat(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    initialize_prompt_store()
    def fake_generate_general_response(
        request, history, long_term_memories, session_summary
    ):
        assert long_term_memories == []
        assert session_summary == ""
        return GeneratedGeneralResponse(
            reply=f"General reply to: {request.message}",
            session_summary="Discussed planets.",
            tools_used=["get_weather"],
            memory_updates=[{"key": "preferred detail", "value": "concise answers"}],
            memory_deletions=[],
        )

    monkeypatch.setattr(
        chat_routes,
        "generate_general_response",
        fake_generate_general_response,
    )

    with TestClient(main.app) as client:
        headers = authenticated_headers(client)
        created = client.post(
            "/sessions",
            json={"title": "Separate chats", "agent_type": "general"},
            headers=headers,
        )
        session_id = created.json()["id"]

        response = client.post(
            f"/sessions/{session_id}/general-chat",
            json={"message": "Tell me about planets"},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json() == {
            "reply": "General reply to: Tell me about planets",
            "tools_used": ["get_weather"],
        }
        assert client.get(
            "/sessions",
            params={"agent_type": "general"},
            headers=headers,
        ).json()[0]["title"] == "Tell me about planets"

        detail = client.get(f"/sessions/{session_id}", headers=headers).json()
        assert detail["recipe_chat_messages"] == []
        assert [
            (message["role"], message["content"])
            for message in detail["general_chat_messages"]
        ] == [
            ("user", "Tell me about planets"),
            ("assistant", "General reply to: Tell me about planets"),
        ]
        assert get_general_long_term_memories(
            client.get("/auth/me", headers=headers).json()["id"]
        ) == [
            {
                "key": "preferred detail",
                "memory_type": "fact",
                "value": "concise answers",
            }
        ]
        assert get_general_chat_session_summary(
            client.get("/auth/me", headers=headers).json()["id"],
            session_id,
        ) == "Discussed planets."


def test_general_chat_keeps_five_exchange_short_term_context_and_separate_memories(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    initialize_prompt_store()

    with TestClient(main.app) as client:
        headers = authenticated_headers(client)
        user_id = client.get("/auth/me", headers=headers).json()["id"]
        session_id = client.post(
            "/sessions",
            json={"title": "Memory test", "agent_type": "general"},
            headers=headers,
        ).json()["id"]
        add_general_chat_exchange(
            user_id,
            session_id,
            "Can you tell me today's weather?",
            "Which city?",
            "The user asked about today's weather; location is not known yet.",
        )
        for index in range(6):
            add_general_chat_exchange(
                user_id,
                session_id,
                f"Question {index}",
                f"Answer {index}",
                "The user asked about today's weather and six numbered questions.",
            )

        context = get_general_chat_history(user_id, session_id)
        assert len(context) == 10
        assert context[0]["content"] == "Question 1"
        assert context[-1]["content"] == "Answer 5"
        cache_path = (
            tmp_path
            / "general_short_term_memory"
            / user_id
            / f"{session_id}.json"
        )
        assert not cache_path.exists()

        update_general_long_term_memories(
            user_id,
            [
                {
                    "key": "preferred name",
                    "memory_type": "preference",
                    "value": "Sam",
                }
            ],
            [],
        )
        assert get_general_long_term_memories(user_id) == [
            {
                "key": "preferred name",
                "memory_type": "preference",
                "value": "Sam",
            }
        ]
        assert get_long_term_memories(user_id) == []
        assert get_general_chat_session_summary(user_id, session_id) == (
            "The user asked about today's weather and six numbered questions."
        )


def test_general_long_term_memory_retrieval_selects_relevant_records(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    initialize_prompt_store()
    initialize_auth_store()
    initialize_session_store()
    user, _ = register_user("memory@example.com", "correct-horse-battery")

    update_general_long_term_memories(
        user["id"],
        [
            {
                "key": "preferred name",
                "memory_type": "fact",
                "value": "Anuu",
            },
            {
                "key": "career decision",
                "memory_type": "decision",
                "value": "Plans to study computer science",
            },
            {
                "key": "learning preference",
                "memory_type": "preference",
                "value": "Likes concise explanations",
            },
        ],
        [],
    )

    selected = get_general_long_term_memories(
        user["id"],
        query="What career decision did I make?",
    )
    assert selected == [
        {
            "key": "career decision",
            "memory_type": "decision",
            "value": "Plans to study computer science",
        }
    ]
    assert get_general_long_term_memories(
        user["id"],
        query="Who am I?",
    )[0]["key"] == "preferred name"


def test_general_name_is_saved_and_recovered_across_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    initialize_auth_store()
    initialize_session_store()
    user, _ = register_user("name-memory@example.com", "correct-horse-battery")
    user_id = user["id"]

    previous_session = create_session(user_id, "Previous", agent_type="general")
    with sqlite3.connect(tmp_path / "prompts.db") as connection:
        connection.execute(
            """
            INSERT INTO general_chat_messages (session_id, role, content, created_at)
            VALUES (?, 'user', 'im anuu', '2026-10-10T00:00:00+00:00')
            """,
            (previous_session["id"],),
        )

    recovered = get_general_long_term_memories(user_id, query="Who am I?")
    assert recovered == [
        {"key": "user name", "memory_type": "fact", "value": "anuu"}
    ]

    current_session = create_session(user_id, "Current", agent_type="general")
    add_general_chat_exchange(
        user_id,
        current_session["id"],
        "I'm Anuu",
        "Nice to meet you, Anuu.",
        "",
    )
    assert get_general_long_term_memories(user_id, query="What is my name?") == [
        {"key": "user name", "memory_type": "fact", "value": "Anuu"}
    ]


def test_general_chat_seeds_session_summary_from_older_user_messages(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()
    received = {}

    class FakeGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            received.setdefault("calls", []).append(kwargs)
            if "tools" in kwargs:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="We discussed several topics.",
                                tool_calls=[],
                            )
                        )
                    ]
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "session_summary": (
                                        "Asked about weather in Dehradun, the CJP "
                                        "protest, green-card news, and the user's name Anuu."
                                    ),
                                    "memory_updates": [],
                                    "memory_deletions": [],
                                }
                            ),
                            tool_calls=[],
                        )
                    )
                ]
            )

    monkeypatch.setattr(general_chat, "Groq", FakeGroq)
    with TestClient(main.app) as client:
        headers = authenticated_headers(client)
        user_id = client.get("/auth/me", headers=headers).json()["id"]
        session_id = client.post(
            "/sessions",
            json={"title": "Older topic seed", "agent_type": "general"},
            headers=headers,
        ).json()["id"]
        for user_message, assistant_reply in [
            ("hii", "Hello!"),
            ("do u know me", "I don't have information yet."),
            ("im anuu", "Nice to meet you, Anuu."),
            ("can u teel me weather today", "Which city?"),
            ("dehradun", "Weather in Dehradun is clear."),
            ("recent cjp protest in india", "Here is recent information."),
        ]:
            add_general_chat_exchange(
                user_id,
                session_id,
                user_message,
                assistant_reply,
                "",
            )

        previous_summary = get_general_chat_session_summary(user_id, session_id)
        generated = general_chat.generate_general_response(
            GeneralChatRequest(message="brief about topics we discussed"),
            history=get_general_chat_history(user_id, session_id),
            long_term_memories=[],
            session_summary=previous_summary,
        )

    assert generated.session_summary.startswith("Asked about weather in Dehradun")
    assert "weather today" in received["calls"][0]["messages"][0]["content"]
    assert "dehradun" in received["calls"][0]["messages"][0]["content"]
    assert received["calls"][1]["response_format"] == {"type": "json_object"}


def test_general_chat_bounds_model_context_and_long_term_memories(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMPT_DATABASE_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    initialize_prompt_store()
    received = {}

    class FakeGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            received.setdefault("calls", []).append(kwargs)
            if "tools" in kwargs:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Understood.",
                                tool_calls=[],
                            )
                        )
                    ]
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "session_summary": (
                                        "The user asked about an earlier question."
                                    ),
                                    "memory_updates": [
                                        {
                                            "type": "preference",
                                            "key": "answer detail",
                                            "value": "concise explanations",
                                        }
                                    ],
                                    "memory_deletions": [],
                                }
                            ),
                            tool_calls=[],
                        )
                    )
                ]
            )

    monkeypatch.setattr(general_chat, "Groq", FakeGroq)
    result = general_chat.generate_general_response(
        GeneralChatRequest(message="Please remember this preference."),
        history=[
            {"role": "user", "content": f"Earlier question {i} " + ("x" * 1200)}
            if i % 2 == 0
            else {"role": "assistant", "content": f"Earlier answer {i} " + ("y" * 1200)}
            for i in range(10)
        ],
        long_term_memories=[
            {"key": f"memory {i}", "value": "z" * 500} for i in range(30)
        ],
    )

    assert result.reply == "Understood."
    assert result.tools_used == []
    assert sum(
        len(item.get("content") or "")
        for item in received["calls"][0]["messages"]
    ) <= general_chat.GENERAL_CHAT_BASE_CONTEXT_CHAR_LIMIT
    assert (
        received["calls"][0]["messages"][-1]["content"]
        == "Please remember this preference."
    )
    assert "tools" not in received["calls"][1]
    assert result.memory_updates == [
        {
            "key": "answer detail",
            "memory_type": "preference",
            "value": "concise explanations",
        }
    ]
