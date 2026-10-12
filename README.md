# Story Summarizer

FastAPI provides Groq-powered story summarization, recipe chat, and a separate general-purpose chatbot. The general chatbot can retrieve current weather from WeatherAPI.com and search the web through Tavily. Streamlit is the frontend; both services use the same SQLite database configured by `PROMPT_DATABASE_PATH` (default: `data/prompts.db`).

Users create an account or log in with email and password. Passwords are stored as scrypt hashes, and API routes that access account data require a bearer token. Sessions and long-term cooking memories are scoped to the signed-in user. On an existing single-user database, the first account registered claims the old sessions and memories.

Recipe and general-chat transcripts are stored in SQLite. Each model request reads only the latest five exchanges for that chat directly from the database; no per-session JSON cache files are created. On startup, the app removes cache files left by older versions. Long-term recipe memories remain in SQLite and are user-specific. General chat has three isolated layers: recent exchanges separate from recipe chat, a compact rolling per-session summary of topics and decisions, and user-scoped structured long-term records (`fact`, `preference`, `milestone`, or `decision`) retrieved by relevance. Explicit self-introductions are saved as a user-level name fact, and older general-chat transcripts are checked to recover a name if that fact was missed, so it remains available in new general-chat sessions for the same account. Older sessions without a summary are seeded from their saved user messages. General-chat answers are generated with weather/search tools enabled, followed by a separate tools-disabled request for structured memory updates. The general chatbot bounds history, memory, and tool results sent to Groq to avoid oversized requests. The two assistants do not share history or memories.

## Project structure

- `app/main.py` creates the FastAPI app and connects its route groups.
- `app/routes/` contains authentication, chat, and session endpoints.
- `app/services/` contains story, recipe-chat, and general-chat logic.
- `app/schemas.py` defines the API request and response models.
- `app/storage/` contains database configuration and SQLite-backed stores.
- `app/prompts/` contains the editable story, recipe, and general-chat prompts.
- `app/tools/` contains tools used by the general chatbot.
- `streamlit_app.py` is the frontend entry point.
- `tests/test_story_and_sessions.py` covers prompt setup, authentication, and sessions.
- `tests/test_recipe_chat.py` covers recipe-chat behavior.
- `tests/test_general_chat.py` covers general-chat behavior and memory.
- `tests/test_general_tools.py` covers weather and web-search tools.
- `data/` holds the SQLite database.

## Windows setup

From the project folder, create and install into the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `GROQ_API_KEY`, `TAVILY_SEARCH_API_KEY`, and `WEATHER_API_KEY` in the project `.env` file. The API loads them automatically. `TAVILY_SEARCH_API_KEY` is only needed when the general chatbot performs web searches; `WEATHER_API_KEY` is used for WeatherAPI.com current conditions and forecasts of up to three days. Keep `.env` private.

Run the backend and frontend in separate terminals, both from the project folder:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The frontend uses `http://127.0.0.1:8000` by default. Set `URL` in `.env` to change the API address.

The Streamlit sidebar's **Workspace** selector switches between Story summarizer, Recipe chatbot, and General chatbot. Each workspace has its own clickable session list, active session, and **New session** action; conversations are titled from the first user message (story sessions use the story name) and sessions are tagged by agent type in SQLite.

## Session API

- `POST /auth/register` creates an account and returns an access token.
- `POST /auth/login` signs in and returns an access token.
- Authenticated `POST /sessions` creates a session; JSON `title` and `agent_type` (`story`, `recipe`, or `general`) select its workspace.
- Authenticated `GET /sessions?agent_type=story|recipe|general` lists only that workspace's sessions. Omitting `agent_type` lists all workspaces.
- Authenticated `GET /sessions/{session_id}` returns session details, summaries, and the full saved chat transcript.
- Authenticated `POST /sessions/{session_id}/summaries` generates and saves a summary in that session.
- Authenticated `POST /sessions/{session_id}/recipe-chat` chats in that session and updates that user's long-term cooking memories.
- Authenticated `POST /sessions/{session_id}/general-chat` uses a separate transcript and can call weather and web-search tools. The response's `tools_used` lists which tools ran, and the backend logs each tool invocation.
- Authenticated `POST /summarize` generates a summary without adding it to session history.

For API calls outside the Streamlit frontend, send `Authorization: Bearer <access_token>` with protected requests.

The editable story, recipe, and general chatbot prompts live in `app/prompts/story.txt`, `app/prompts/recipe.txt`, and `app/prompts/general.txt`. They are loaded when first seeding SQLite.
