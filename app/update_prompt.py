import os
import sqlite3
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
prompt = (project_dir / "app" / "prompts" / "story.txt").read_text(
    encoding="utf-8"
).strip()
database = Path(
    os.getenv("PROMPT_DATABASE_PATH", str(project_dir / "data" / "prompts.db"))
)

with sqlite3.connect(database) as connection:
    cursor = connection.execute(
        "UPDATE prompts SET prompt = ? WHERE key = ?",
        (prompt, "Story Summarizer"),
    )
    if cursor.rowcount == 0:
        raise RuntimeError("Prompt was not found in the database")

print("Prompt updated. The next summary request will use it.")