import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main
from app.prompt_store import (
    DEFAULT_PROMPT,
    PROMPT_FILE,
    PROMPT_KEY,
    get_prompt,
    initialize_prompt_store,
)


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

    monkeypatch.setattr(main, "Groq", FakeGroq)

    with TestClient(main.app) as client:
        response = client.post(
            "/summarize",
            json={"story_name": "The Necklace", "story": "Complete story text"},
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
        main,
        "generate_summary",
        lambda request: main.SummaryResponse(
            key="the-necklace", content="# SUMMARY-THE NECKLACE"
        ),
    )

    with TestClient(main.app) as client:
        created = client.post("/sessions", json={"title": "Class stories"})
        assert created.status_code == 201
        session_id = created.json()["id"]

        summarized = client.post(
            f"/sessions/{session_id}/summaries",
            json={"story_name": "The Necklace", "story": "Full story"},
        )
        assert summarized.status_code == 201
        assert summarized.json()["key"] == "the-necklace"
        assert summarized.json()["content"] == "# SUMMARY-THE NECKLACE"

        detail = client.get(f"/sessions/{session_id}")
        assert detail.status_code == 200
        assert detail.json()["title"] == "Class stories"
        assert len(detail.json()["summaries"]) == 1
        assert client.get("/sessions").json()[0]["id"] == session_id

        missing = client.get("/sessions/not-a-session")
        assert missing.status_code == 404
