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


def test_weather_tool_resolves_place_and_fetches_forecast(monkeypatch):
    monkeypatch.setenv("WEATHER_API_KEY", "weather-test-key")
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "location": {
                    "name": "Paris",
                    "region": "Ile-de-France",
                    "country": "France",
                },
                "current": {"temp_c": 18},
                "forecast": {
                    "forecastday": [
                        {
                            "day": {"maxtemp_c": 20},
                            "hour": [{"temp_c": 18} for _ in range(24)],
                        }
                    ]
                },
            }

    def fake_get(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(general_tools.httpx, "get", fake_get)
    result = general_tools.execute_general_chat_tool(
        "get_weather", {"location": "Paris", "forecast_days": 1}
    )

    assert result == {
        "location": "Paris, Ile-de-France, France",
        "current": {"temp_c": 18},
        "forecast": [{"date": None, "day": {"maxtemp_c": 20}}],
    }
    assert calls["url"] == "https://api.weatherapi.com/v1/forecast.json"
    assert calls["params"]["key"] == "weather-test-key"
    assert calls["params"]["q"] == "Paris"
    assert calls["params"]["days"] == 1


def test_web_search_uses_tavily_key_and_returns_sources(monkeypatch):
    monkeypatch.setenv("TAVILY_SEARCH_API_KEY", "tavily-test-key")
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "Current information",
                        "url": "https://example.com/latest",
                        "content": "A recent report.",
                    }
                ]
            }

    def fake_post(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(general_tools.httpx, "post", fake_post)
    result = general_tools.execute_general_chat_tool(
        "search_web", {"query": "latest information"}
    )

    assert calls["url"] == "https://api.tavily.com/search"
    assert calls["json"]["api_key"] == "tavily-test-key"
    assert calls["json"]["query"] == "latest information"
    assert result["results"][0]["url"] == "https://example.com/latest"
