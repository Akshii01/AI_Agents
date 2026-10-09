# Story Summarizer

FastAPI provides the Groq summarization API and persistent story sessions. Streamlit is the frontend; both services use the same SQLite database configured by `PROMPT_DATABASE_PATH` (default: `data/prompts.db`). Each session has a UUID, and its generated summaries are stored and available through the session API.

## Windows setup

From the project folder, create and install into the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in the project `.env` file. The API loads it automatically. Keep `.env` private.

Run the backend and frontend in separate terminals, both from the project folder:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The frontend uses `http://127.0.0.1:8000` by default. Set `STORY_API_URL` in `.env` to change the API address.

## Session API

- `POST /sessions` creates a session; an optional JSON `title` can name it.
- `GET /sessions` lists sessions.
- `GET /sessions/{session_id}` returns session details and saved summaries.
- `POST /sessions/{session_id}/summaries` generates and saves a summary in that session.
- `POST /summarize` generates a summary without adding it to session history.

The prompt is loaded from `prompt.txt` when first seeding SQLite under the key `Story Summarizer`.
