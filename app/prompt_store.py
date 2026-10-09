import os
import sqlite3
from pathlib import Path

PROMPT_KEY = "Story Summarizer"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompt.txt"
DEFAULT_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()


def get_database_path() -> Path:
    default_path = Path(__file__).resolve().parent.parent / "data" / "prompts.db"
    return Path(os.getenv("PROMPT_DATABASE_PATH", str(default_path)))


def initialize_prompt_store() -> None:
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                key TEXT PRIMARY KEY,
                prompt TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO prompts (key, prompt) VALUES (?, ?)",
            (PROMPT_KEY, DEFAULT_PROMPT),
        )


def get_prompt(key: str = PROMPT_KEY) -> str:
    database_path = get_database_path()
    if not database_path.is_file():
        raise RuntimeError("Prompt database is not initialized")

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT prompt FROM prompts WHERE key = ?", (key,)
        ).fetchone()

    if row is None:
        raise LookupError(f"No prompt found for key: {key}")
    return str(row[0])
