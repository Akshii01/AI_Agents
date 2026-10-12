# Story Summarizer

A FastAPI and Streamlit app for summarizing stories and chatting about recipes or general topics. The general chatbot can also look up weather and search the web.

## Setup

From the project folder, create the virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Add `GROQ_API_KEY` to `.env`. For general-chat tools, also add `WEATHER_API_KEY` and `TAVILY_SEARCH_API_KEY`. Keep `.env` private.

## Run

Start the API and frontend in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the Streamlit address shown in the terminal. The frontend uses `http://127.0.0.1:8000` for the API by default; set `URL` in `.env` to change it.

## Project layout

- `app/routes/` — API endpoints
- `app/services/` — summarization and chat logic
- `app/storage/` — SQLite database access
- `app/prompts/` — editable chatbot prompts
- `app/tools/` — weather and web-search tools
- `streamlit_app.py` — frontend
- `tests/` — automated tests
- `data/` — SQLite database

Chats, summaries, and memories are stored in SQLite. General-chat replies include a `tools_used` field showing which tools ran.

## API docs

With the API running, open `http://127.0.0.1:8000/docs`. Register or log in, then use **Authorize** with the access token to test protected endpoints.
