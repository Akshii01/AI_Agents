import sqlite3
from pathlib import Path

from app.storage.database import get_database_path

PROMPT_KEY = "Story Summarizer"
PROMPT_DIRECTORY = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_FILE = PROMPT_DIRECTORY / "story.txt"
DEFAULT_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()
RECIPE_PROMPT_KEY = "Recipe Chatbot"
RECIPE_PROMPT_FILE = PROMPT_DIRECTORY / "recipe.txt"
DEFAULT_RECIPE_PROMPT = RECIPE_PROMPT_FILE.read_text(encoding="utf-8").strip()
GENERAL_PROMPT_KEY = "General Chatbot"
GENERAL_PROMPT_FILE = PROMPT_DIRECTORY / "general.txt"
DEFAULT_GENERAL_PROMPT = GENERAL_PROMPT_FILE.read_text(encoding="utf-8").strip()


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
        connection.execute(
            "INSERT OR IGNORE INTO prompts (key, prompt) VALUES (?, ?)",
            (RECIPE_PROMPT_KEY, DEFAULT_RECIPE_PROMPT),
        )
        connection.execute(
            "INSERT OR IGNORE INTO prompts (key, prompt) VALUES (?, ?)",
            (GENERAL_PROMPT_KEY, DEFAULT_GENERAL_PROMPT),
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
