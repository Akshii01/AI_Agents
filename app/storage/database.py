import os
from pathlib import Path


def get_database_path() -> Path:
    default_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "prompts.db"
    )
    return Path(os.getenv("PROMPT_DATABASE_PATH", str(default_path)))
